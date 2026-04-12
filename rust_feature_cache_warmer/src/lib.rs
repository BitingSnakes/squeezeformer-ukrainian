use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryBuilder, BooleanBuilder, Float64Array, Int32Array,
    Int64Array, LargeBinaryArray, LargeStringArray, StringArray, StringBuilder, StructArray,
    UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use audioadapter_buffers::direct::InterleavedSlice;
use clap::{Parser, ValueEnum};
use env_logger::Env;
use log::{debug, error, info, trace, warn};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rubato::{Fft, FixedSync, Resampler};
use rustfft::num_complex::{Complex32, Complex64};
use rustfft::FftPlanner;
use sha2::{Digest, Sha256};
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

const RUST_PAYLOAD_MAGIC: &[u8; 8] = b"SFCF32L1";

#[cfg(feature = "python")]
mod python;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum FrontendKind {
    Squeezeformer,
    Zipformer,
    W2vBert,
}

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    /// Input parquet manifest. Mutually exclusive with --input-folder.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Directory containing parquet manifests to process recursively.
    #[arg(long)]
    input_folder: Option<PathBuf>,

    /// Split cache root to write. The crate creates feature_shards/features_XX below it.
    #[arg(long)]
    cache_dir: PathBuf,

    /// Resolve relative audio paths against this directory. Defaults to the input parquet parent
    /// for --input, or the folder root for --input-folder.
    #[arg(long)]
    source_base: Option<PathBuf>,

    /// Frontend defaults to mirror.
    #[arg(long, value_enum, default_value_t = FrontendKind::Squeezeformer)]
    frontend: FrontendKind,

    /// Optional row limit for smoke runs.
    #[arg(long)]
    limit: Option<usize>,

    /// Number of cache shards. Must match Python ShardedParquetFeatureCache.
    #[arg(long, default_value_t = 64)]
    num_shards: usize,

    /// Input parquet record batch size.
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,

    /// Flush a parquet part for a shard after this many rows.
    #[arg(long, default_value_t = 256)]
    rows_per_part: usize,

    /// Stop on the first row error instead of skipping bad rows.
    #[arg(long, default_value_t = false)]
    fail_fast: bool,

    /// Parallel feature extraction threads. Use 0 for Rayon's default.
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Disable ffmpeg fallback when Symphonia cannot decode a codec such as Opus.
    #[arg(long, default_value_t = false)]
    no_ffmpeg_fallback: bool,

    #[arg(long)]
    sample_rate: Option<u32>,
    #[arg(long)]
    n_fft: Option<usize>,
    #[arg(long)]
    win_length: Option<usize>,
    #[arg(long)]
    hop_length: Option<usize>,
    #[arg(long)]
    n_mels: Option<usize>,
    #[arg(long)]
    preemphasis: Option<f32>,
    #[arg(long)]
    normalize_signal: Option<bool>,
    #[arg(long)]
    normalize_feature: Option<bool>,
    #[arg(long)]
    normalize_per_frame: Option<bool>,

    /// W2V-BERT model source used only for cache key compatibility.
    #[arg(long, default_value = "facebook/w2v-bert-2.0")]
    w2v_model_source: String,
    #[arg(long, default_value_t = 80)]
    w2v_feature_size: usize,
    #[arg(long, default_value_t = 2)]
    w2v_stride: usize,
    #[arg(long, default_value_t = 1.0)]
    w2v_padding_value: f32,
}

#[derive(Debug, Clone)]
pub struct AudioFrontendConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub preemphasis: f32,
    pub normalize_signal: bool,
    pub normalize_feature: bool,
    pub normalize_per_frame: bool,
}

#[derive(Debug, Clone)]
pub struct W2vBertFrontendConfig {
    pub model_source: String,
    pub sample_rate: u32,
    pub feature_size: usize,
    pub stride: usize,
    pub feature_dim: usize,
    pub padding_value: f32,
}

#[derive(Debug, Clone)]
enum FrontendConfig {
    Audio(AudioFrontendConfig),
    W2vBert(W2vBertFrontendConfig),
}

#[derive(Debug)]
pub struct FeatureMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone)]
enum AudioSource {
    Path(PathBuf),
    Bytes(Vec<u8>, Option<String>),
}

impl AudioSource {
    fn log_label(&self) -> String {
        match self {
            Self::Path(path) => format!("path={}", path.display()),
            Self::Bytes(bytes, Some(path_hint)) => {
                format!("bytes={} path_hint={path_hint}", bytes.len())
            }
            Self::Bytes(bytes, None) => format!("bytes={}", bytes.len()),
        }
    }
}

#[derive(Debug)]
struct CacheRow {
    key: String,
    payload: Vec<u8>,
}

#[derive(Default)]
struct Counters {
    scanned: usize,
    written: usize,
    skipped: usize,
}

pub fn squeezeformer_frontend_config() -> AudioFrontendConfig {
    AudioFrontendConfig {
        sample_rate: 16_000,
        n_fft: 400,
        win_length: 400,
        hop_length: 160,
        n_mels: 80,
        preemphasis: 0.97,
        normalize_signal: true,
        normalize_feature: true,
        normalize_per_frame: false,
    }
}

pub fn zipformer_frontend_config() -> AudioFrontendConfig {
    AudioFrontendConfig {
        sample_rate: 16_000,
        n_fft: 400,
        win_length: 400,
        hop_length: 160,
        n_mels: 80,
        preemphasis: 0.0,
        normalize_signal: false,
        normalize_feature: false,
        normalize_per_frame: false,
    }
}

pub fn w2v_bert_frontend_config(
    model_source: Option<String>,
    sample_rate: Option<u32>,
    feature_size: Option<usize>,
    stride: Option<usize>,
    feature_dim: Option<usize>,
    padding_value: Option<f32>,
) -> W2vBertFrontendConfig {
    let feature_size = feature_size.unwrap_or(80);
    let stride = stride.unwrap_or(2).max(1);
    W2vBertFrontendConfig {
        model_source: model_source.unwrap_or_else(|| "facebook/w2v-bert-2.0".to_string()),
        sample_rate: sample_rate.unwrap_or(16_000),
        feature_size,
        stride,
        feature_dim: feature_dim.unwrap_or(feature_size * stride),
        padding_value: padding_value.unwrap_or(1.0),
    }
}

pub fn extract_audio_features_from_samples(
    waveform: &[f32],
    sample_rate: u32,
    config: &AudioFrontendConfig,
) -> Result<FeatureMatrix> {
    let mut waveform = waveform.to_vec();
    compute_audio_featurizer_features(&mut waveform, sample_rate, config)
}

pub fn extract_w2v_bert_features_from_samples(
    waveform: &[f32],
    sample_rate: u32,
    config: &W2vBertFrontendConfig,
) -> Result<FeatureMatrix> {
    let mut waveform = waveform.to_vec();
    compute_w2v_bert_features(&mut waveform, sample_rate, config)
}

pub fn run_cli() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .try_init()
        .ok();

    let cli = Cli::parse();
    if cli.num_shards == 0 {
        bail!("--num-shards must be greater than zero");
    }
    if cli.rows_per_part == 0 {
        bail!("--rows-per-part must be greater than zero");
    }

    let input_paths = resolve_input_paths(&cli)?;
    let source_base = cli
        .source_base
        .clone()
        .unwrap_or_else(|| default_source_base(&cli, &input_paths));
    let frontend = FrontendConfig::from_cli(&cli);
    let frontend_hash = frontend.frontend_hash();
    info!(
        "starting cache warm inputs={} cache_dir={} source_base={} frontend={:?} frontend_hash={} batch_size={} rows_per_part={} num_shards={} fail_fast={} ffmpeg_fallback={}",
        input_paths.len(),
        cli.cache_dir.display(),
        source_base.display(),
        cli.frontend,
        frontend_hash,
        cli.batch_size,
        cli.rows_per_part,
        cli.num_shards,
        cli.fail_fast,
        !cli.no_ffmpeg_fallback
    );
    debug!("resolved input parquet files: {:?}", input_paths);

    let mut writer = ShardedCacheWriter::new(&cli.cache_dir, cli.num_shards, cli.rows_per_part)?;
    let mut counters = Counters::default();
    let mut pool_builder = ThreadPoolBuilder::new();
    if cli.threads > 0 {
        pool_builder = pool_builder.num_threads(cli.threads);
    }
    let pool = pool_builder
        .build()
        .context("failed to build Rayon feature extraction thread pool")?;
    info!(
        "feature extraction thread pool ready threads={}",
        pool.current_num_threads()
    );

    'inputs: for input_path in input_paths {
        info!("warming input {}", input_path.display());
        let input = File::open(&input_path)
            .with_context(|| format!("failed to open input parquet {}", input_path.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(input)?;
        let reader = builder.with_batch_size(cli.batch_size).build()?;

        for (batch_index, batch_result) in reader.enumerate() {
            let batch = batch_result?;
            let rows_to_process = match cli.limit {
                Some(limit) => limit.saturating_sub(counters.scanned).min(batch.num_rows()),
                None => batch.num_rows(),
            };
            if rows_to_process == 0 {
                info!(
                    "row limit reached limit={} scanned={}",
                    cli.limit.unwrap_or_default(),
                    counters.scanned
                );
                break 'inputs;
            }
            debug!(
                "processing batch input={} batch={} rows={} rows_to_process={}",
                input_path.display(),
                batch_index,
                batch.num_rows(),
                rows_to_process
            );
            let starting_scanned = counters.scanned;
            let results: Vec<(usize, Result<Option<CacheRow>>)> = pool.install(|| {
                (0..rows_to_process)
                    .into_par_iter()
                    .map(|row_index| {
                        let scanned_row = starting_scanned + row_index + 1;
                        let result = process_manifest_row(
                            &batch,
                            row_index,
                            scanned_row,
                            &source_base,
                            &frontend,
                            &frontend_hash,
                            !cli.no_ffmpeg_fallback,
                        );
                        (scanned_row, result)
                    })
                    .collect()
            });
            for (scanned_row, result) in results {
                counters.scanned = scanned_row;
                match result {
                    Ok(Some(cache_row)) => {
                        writer.push(cache_row)?;
                        counters.written += 1;
                    }
                    Ok(None) => {
                        counters.skipped += 1;
                        trace!("skipping row {}: no audio source found", scanned_row);
                    }
                    Err(error) if cli.fail_fast => {
                        error!("failed row {}: {error:#}", scanned_row);
                        return Err(error);
                    }
                    Err(error) => {
                        counters.skipped += 1;
                        warn!("skipping row {}: {error:#}", scanned_row);
                    }
                }
                if scanned_row % 1000 == 0 {
                    info!(
                        "progress scanned={} written={} skipped={}",
                        counters.scanned, counters.written, counters.skipped
                    );
                }
            }
        }
    }

    writer.finish()?;
    info!(
        "complete scanned={} written={} skipped={}",
        counters.scanned, counters.written, counters.skipped
    );
    Ok(())
}

fn resolve_input_paths(cli: &Cli) -> Result<Vec<PathBuf>> {
    match (&cli.input, &cli.input_folder) {
        (Some(_), Some(_)) => bail!("--input and --input-folder are mutually exclusive"),
        (None, None) => bail!("either --input or --input-folder is required"),
        (Some(input), None) => {
            if !input.is_file() {
                bail!("--input must point to a parquet file: {}", input.display());
            }
            Ok(vec![input.clone()])
        }
        (None, Some(input_folder)) => {
            if !input_folder.is_dir() {
                bail!(
                    "--input-folder must point to a directory: {}",
                    input_folder.display()
                );
            }
            let mut paths = Vec::new();
            collect_parquet_paths(input_folder, &mut paths)?;
            paths.sort();
            if paths.is_empty() {
                bail!(
                    "--input-folder contains no parquet files: {}",
                    input_folder.display()
                );
            }
            Ok(paths)
        }
    }
}

fn collect_parquet_paths(directory: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to read input folder {}", directory.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            collect_parquet_paths(&path, paths)?;
        } else if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("parquet"))
        {
            paths.push(path);
        }
    }
    Ok(())
}

fn default_source_base(cli: &Cli, input_paths: &[PathBuf]) -> PathBuf {
    if let Some(input_folder) = &cli.input_folder {
        return input_folder.clone();
    }
    input_paths
        .first()
        .and_then(|path| path.parent())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

impl FrontendConfig {
    fn from_cli(cli: &Cli) -> Self {
        match cli.frontend {
            FrontendKind::Squeezeformer => {
                let mut config = squeezeformer_frontend_config();
                apply_audio_cli_overrides(&mut config, cli);
                Self::Audio(config)
            }
            FrontendKind::Zipformer => {
                let mut config = zipformer_frontend_config();
                apply_audio_cli_overrides(&mut config, cli);
                Self::Audio(config)
            }
            FrontendKind::W2vBert => {
                let feature_size = cli.n_mels.unwrap_or(cli.w2v_feature_size);
                Self::W2vBert(w2v_bert_frontend_config(
                    Some(cli.w2v_model_source.clone()),
                    cli.sample_rate,
                    Some(feature_size),
                    Some(cli.w2v_stride),
                    Some(feature_size * cli.w2v_stride.max(1)),
                    Some(cli.w2v_padding_value),
                ))
            }
        }
    }

    fn feature_dim(&self) -> usize {
        match self {
            Self::Audio(config) => config.n_mels,
            Self::W2vBert(config) => config.feature_dim,
        }
    }

    fn sample_rate(&self) -> u32 {
        match self {
            Self::Audio(config) => config.sample_rate,
            Self::W2vBert(config) => config.sample_rate,
        }
    }

    fn config_repr(&self) -> String {
        match self {
            Self::Audio(config) => format!(
                "{{'featurizer': {{'sample_rate': {}, 'n_fft': {}, 'win_length': {}, 'n_mels': {}, 'backend': 'torchaudio', 'preemphasis': {}, 'normalize_signal': {}, 'normalize_feature': {}, 'normalize_per_frame': {}, 'hop_length': {}}}}}",
                config.sample_rate,
                config.n_fft,
                config.win_length,
                config.n_mels,
                py_float(config.preemphasis),
                py_bool(config.normalize_signal),
                py_bool(config.normalize_feature),
                py_bool(config.normalize_per_frame),
                config.hop_length,
            ),
            Self::W2vBert(config) => format!(
                "{{'featurizer': {{'type': 'w2v_bert', 'model_source': '{}', 'sample_rate': {}, 'feature_size': {}, 'stride': {}, 'feature_dim': {}, 'padding_value': {}}}}}",
                config.model_source.replace('\'', "\\'"),
                config.sample_rate,
                config.feature_size,
                config.stride,
                config.feature_dim,
                py_float(config.padding_value),
            ),
        }
    }

    fn frontend_hash(&self) -> String {
        let digest = Sha256::digest(self.config_repr().as_bytes());
        hex_prefix(&digest, 12)
    }
}

fn apply_audio_cli_overrides(config: &mut AudioFrontendConfig, cli: &Cli) {
    if let Some(sample_rate) = cli.sample_rate {
        config.sample_rate = sample_rate;
    }
    if let Some(n_fft) = cli.n_fft {
        config.n_fft = n_fft;
        if cli.win_length.is_none() {
            config.win_length = n_fft;
        }
    }
    if let Some(win_length) = cli.win_length {
        config.win_length = win_length;
    }
    if let Some(hop_length) = cli.hop_length {
        config.hop_length = hop_length;
    }
    if let Some(n_mels) = cli.n_mels {
        config.n_mels = n_mels;
    }
    if let Some(preemphasis) = cli.preemphasis {
        config.preemphasis = preemphasis;
    }
    if let Some(normalize_signal) = cli.normalize_signal {
        config.normalize_signal = normalize_signal;
    }
    if let Some(normalize_feature) = cli.normalize_feature {
        config.normalize_feature = normalize_feature;
    }
    if let Some(normalize_per_frame) = cli.normalize_per_frame {
        config.normalize_per_frame = normalize_per_frame;
    }
}

fn py_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

fn py_float(value: f32) -> String {
    if value.is_finite() {
        let mut rendered = format!("{:?}", value);
        if !rendered.contains('.') && !rendered.contains('e') && !rendered.contains('E') {
            rendered.push_str(".0");
        }
        rendered
    } else if value.is_nan() {
        "nan".to_string()
    } else if value.is_sign_positive() {
        "inf".to_string()
    } else {
        "-inf".to_string()
    }
}

fn process_manifest_row(
    batch: &RecordBatch,
    row_index: usize,
    scanned_rows: usize,
    source_base: &Path,
    frontend: &FrontendConfig,
    frontend_hash: &str,
    ffmpeg_fallback: bool,
) -> Result<Option<CacheRow>> {
    let row = manifest_audio_row(batch, row_index, scanned_rows, source_base)?;
    let Some((utterance_id, source)) = row else {
        return Ok(None);
    };
    trace!(
        "row {} utterance_id={} source={}",
        scanned_rows,
        utterance_id,
        source.log_label()
    );
    let (waveform, sample_rate) = decode_audio(source, frontend.sample_rate(), ffmpeg_fallback)?;
    let features = compute_features(waveform, sample_rate, frontend)?;
    if features.rows == 0 || features.cols != frontend.feature_dim() {
        bail!(
            "invalid feature matrix for utterance_id={utterance_id}: rows={} cols={} expected_cols={}",
            features.rows,
            features.cols,
            frontend.feature_dim()
        );
    }
    trace!(
        "computed features row={} utterance_id={} frames={} dim={}",
        scanned_rows,
        utterance_id,
        features.rows,
        features.cols
    );
    let key = cache_key(&utterance_id, frontend_hash);
    let payload = encode_feature_payload(&features)?;
    Ok(Some(CacheRow { key, payload }))
}

fn manifest_audio_row(
    batch: &RecordBatch,
    row_index: usize,
    scanned_rows: usize,
    source_base: &Path,
) -> Result<Option<(String, AudioSource)>> {
    let id = column_by_name(batch, &["id", "utterance_id"])
        .and_then(|array| scalar_as_string(array.as_ref(), row_index))
        .filter(|value| !value.is_empty());
    let top_level_path = column_by_name(batch, &["path"])
        .and_then(|array| scalar_as_string(array.as_ref(), row_index))
        .filter(|value| !value.is_empty());

    let mut audio_path = top_level_path.clone();
    let mut audio_bytes = None;
    if let Some(audio_array) = column_by_name(batch, &["audio"]) {
        match audio_array.data_type() {
            DataType::Struct(_) => {
                if let Some(struct_array) = audio_array.as_any().downcast_ref::<StructArray>() {
                    if !struct_array.is_null(row_index) {
                        if audio_bytes.is_none() {
                            audio_bytes = struct_child(struct_array, "bytes")
                                .and_then(|array| scalar_as_bytes(array.as_ref(), row_index));
                        }
                        if audio_path.is_none() {
                            audio_path = struct_child(struct_array, "path")
                                .and_then(|array| scalar_as_string(array.as_ref(), row_index));
                        }
                    }
                }
            }
            DataType::Binary | DataType::LargeBinary => {
                audio_bytes = scalar_as_bytes(audio_array.as_ref(), row_index);
            }
            _ => {}
        }
    }

    let utterance_id = id
        .or_else(|| audio_path.clone())
        .unwrap_or_else(|| scanned_rows.to_string());
    if let Some(bytes) = audio_bytes {
        return Ok(Some((utterance_id, AudioSource::Bytes(bytes, audio_path))));
    }
    if let Some(path) = audio_path {
        if path.starts_with("http://") || path.starts_with("https://") {
            bail!("remote audio URLs are not supported by the Rust warmer: {path}");
        }
        return Ok(Some((
            utterance_id,
            AudioSource::Path(resolve_path(source_base, &path)),
        )));
    }
    Ok(None)
}

fn column_by_name(batch: &RecordBatch, names: &[&str]) -> Option<ArrayRef> {
    for name in names {
        if let Ok(index) = batch.schema().index_of(name) {
            return Some(batch.column(index).clone());
        }
    }
    None
}

fn struct_child(struct_array: &StructArray, name: &str) -> Option<ArrayRef> {
    struct_array
        .column_names()
        .iter()
        .position(|candidate| *candidate == name)
        .map(|index| struct_array.column(index).clone())
}

fn scalar_as_string(array: &dyn Array, row_index: usize) -> Option<String> {
    if array.is_null(row_index) {
        return None;
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Some(values.value(row_index).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Some(values.value(row_index).to_string());
    }
    None
}

fn scalar_as_bytes(array: &dyn Array, row_index: usize) -> Option<Vec<u8>> {
    if array.is_null(row_index) {
        return None;
    }
    if let Some(values) = array.as_any().downcast_ref::<BinaryArray>() {
        return Some(values.value(row_index).to_vec());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        return Some(values.value(row_index).to_vec());
    }
    None
}

fn resolve_path(source_base: &Path, path: &str) -> PathBuf {
    let path_buf = PathBuf::from(path);
    if path_buf.is_absolute() {
        path_buf
    } else {
        source_base.join(path_buf)
    }
}

fn decode_audio(
    source: AudioSource,
    fallback_sample_rate: u32,
    ffmpeg_fallback: bool,
) -> Result<(Vec<f32>, u32)> {
    let source_label = source.log_label();
    debug!(
        "decoding audio source={} fallback_sample_rate={} ffmpeg_fallback={}",
        source_label, fallback_sample_rate, ffmpeg_fallback
    );
    match decode_audio_symphonia(source.clone()) {
        Ok(decoded) => {
            debug!(
                "decoded audio with symphonia source={} samples={} sample_rate={}",
                source_label,
                decoded.0.len(),
                decoded.1
            );
            Ok(decoded)
        }
        Err(symphonia_error) if ffmpeg_fallback => {
            warn!(
                "symphonia decode failed for {}; falling back to ffmpeg: {symphonia_error:#}",
                source_label
            );
            decode_audio_ffmpeg(source, fallback_sample_rate).with_context(|| {
                format!("symphonia decode failed: {symphonia_error:#}; ffmpeg fallback failed")
            })
        }
        Err(error) => Err(error),
    }
}

fn decode_audio_symphonia(source: AudioSource) -> Result<(Vec<f32>, u32)> {
    let (mss, extension) = match source {
        AudioSource::Path(path) => {
            trace!("opening audio file with symphonia path={}", path.display());
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .map(str::to_owned);
            let file = File::open(&path)
                .with_context(|| format!("failed to open audio file {}", path.display()))?;
            (
                MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default()),
                extension,
            )
        }
        AudioSource::Bytes(bytes, path_hint) => {
            let extension = path_hint
                .as_deref()
                .and_then(|path| Path::new(path).extension())
                .and_then(|value| value.to_str())
                .map(str::to_owned);
            (
                MediaSourceStream::new(
                    Box::new(Cursor::new(bytes)),
                    MediaSourceStreamOptions::default(),
                ),
                extension,
            )
        }
    };

    let mut hint = Hint::new();
    if let Some(extension) = extension.as_deref() {
        hint.with_extension(extension);
    }
    let probed = get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("audio container has no default track"))?;
    if track.codec_params.codec == CODEC_TYPE_NULL {
        bail!("unsupported null audio codec");
    }
    let track_id = track.id;
    debug!(
        "symphonia selected track id={} codec={:?} sample_rate={:?}",
        track_id, track.codec_params.codec, track.codec_params.sample_rate
    );
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut mono = Vec::new();
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(16_000);

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(SymphoniaError::ResetRequired) => {
                bail!("decoder reset is not supported for this audio stream");
            }
            Err(error) => return Err(error.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(message)) => {
                trace!("symphonia skipped packet decode error: {message}");
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        append_mono_samples(decoded, &mut mono, &mut sample_rate);
    }

    if mono.is_empty() {
        bail!("decoded audio stream is empty");
    }
    Ok((mono, sample_rate))
}

fn decode_audio_ffmpeg(source: AudioSource, sample_rate: u32) -> Result<(Vec<f32>, u32)> {
    debug!(
        "decoding audio with ffmpeg source={} output_sample_rate={}",
        source.log_label(),
        sample_rate
    );
    let sample_rate_arg = sample_rate.to_string();
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error"]);
    match source {
        AudioSource::Path(path) => {
            command.arg("-i").arg(path);
            let output = command
                .args([
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    &sample_rate_arg,
                    "-f",
                    "f32le",
                    "pipe:1",
                ])
                .output()
                .context("failed to execute ffmpeg")?;
            decode_ffmpeg_output(output, sample_rate)
        }
        AudioSource::Bytes(bytes, _) => {
            let mut child = command
                .args([
                    "-i",
                    "pipe:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    &sample_rate_arg,
                    "-f",
                    "f32le",
                    "pipe:1",
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .context("failed to execute ffmpeg")?;
            {
                let mut stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| anyhow!("failed to open ffmpeg stdin"))?;
                stdin
                    .write_all(&bytes)
                    .context("failed to write audio bytes to ffmpeg stdin")?;
            }
            let output = child
                .wait_with_output()
                .context("failed to wait for ffmpeg")?;
            decode_ffmpeg_output(output, sample_rate)
        }
    }
}

fn decode_ffmpeg_output(output: std::process::Output, sample_rate: u32) -> Result<(Vec<f32>, u32)> {
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "ffmpeg exited with status {}: {}",
            output.status,
            stderr.trim()
        );
    }
    if !output.stdout.len().is_multiple_of(4) {
        bail!(
            "ffmpeg produced {} bytes, which is not divisible by f32 size",
            output.stdout.len()
        );
    }
    let samples = output
        .stdout
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size is exact")))
        .collect::<Vec<_>>();
    if samples.is_empty() {
        bail!("ffmpeg decoded audio stream is empty");
    }
    debug!(
        "decoded audio with ffmpeg samples={} sample_rate={}",
        samples.len(),
        sample_rate
    );
    Ok((samples, sample_rate))
}

fn append_mono_samples(decoded: AudioBufferRef<'_>, output: &mut Vec<f32>, sample_rate: &mut u32) {
    let spec = *decoded.spec();
    *sample_rate = spec.rate;
    let channels = spec.channels.count().max(1);
    let mut sample_buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
    sample_buffer.copy_interleaved_ref(decoded);
    for frame in sample_buffer.samples().chunks(channels) {
        let sum: f32 = frame.iter().copied().sum();
        output.push(sum / channels as f32);
    }
}

fn compute_features(
    mut waveform: Vec<f32>,
    sample_rate: u32,
    frontend: &FrontendConfig,
) -> Result<FeatureMatrix> {
    match frontend {
        FrontendConfig::Audio(config) => {
            compute_audio_featurizer_features(&mut waveform, sample_rate, config)
        }
        FrontendConfig::W2vBert(config) => {
            compute_w2v_bert_features(&mut waveform, sample_rate, config)
        }
    }
}

fn compute_audio_featurizer_features(
    waveform: &mut Vec<f32>,
    sample_rate: u32,
    config: &AudioFrontendConfig,
) -> Result<FeatureMatrix> {
    validate_audio_config(config)?;
    if sample_rate != config.sample_rate {
        *waveform = resample_to_sample_rate(waveform, sample_rate, config.sample_rate)?;
    }
    if config.normalize_signal {
        normalize_signal(waveform);
    }
    if config.preemphasis > 0.0 {
        apply_waveform_preemphasis(waveform, config.preemphasis);
    }
    let required = config.n_fft.max(config.win_length);
    if waveform.len() < required {
        waveform.resize(required, 0.0);
    }

    let window = padded_hann_window(config.win_length, config.n_fft, true);
    let powers = power_spectrogram(
        waveform,
        SpectrogramOptions {
            frame_length: config.n_fft,
            hop_length: config.hop_length,
            fft_length: config.n_fft,
            center: true,
            window: &window,
            remove_dc_offset: false,
            frame_preemphasis: None,
        },
    )?;
    let filters = mel_filter_bank(
        config.n_fft / 2 + 1,
        config.n_mels,
        config.sample_rate,
        0.0,
        config.sample_rate as f32 / 2.0,
        MelScale::Htk,
        false,
    );
    let mut features = log_mel_from_power(&powers, &filters, 1e-5);
    if config.normalize_feature {
        if config.normalize_per_frame {
            normalize_rows(&mut features, 1e-5);
        } else {
            normalize_columns(&mut features, 1e-5, false);
        }
    }
    Ok(features)
}

fn compute_w2v_bert_features(
    waveform: &mut Vec<f32>,
    sample_rate: u32,
    config: &W2vBertFrontendConfig,
) -> Result<FeatureMatrix> {
    if sample_rate != config.sample_rate {
        *waveform = resample_to_sample_rate(waveform, sample_rate, config.sample_rate)?;
    }
    for value in waveform.iter_mut() {
        *value *= 32768.0;
    }
    let mut features = seamless_m4t_log_mel_features(
        waveform,
        SeamlessM4TFbankOptions {
            sample_rate: config.sample_rate,
            frame_length: 400,
            hop_length: 160,
            fft_length: 512,
            num_mel_bins: config.feature_size,
            mel_floor: 1.192_092_955_078_125e-7,
            preemphasis: 0.97,
        },
    )?;
    normalize_columns_with_variance_epsilon(&mut features, 1e-7, true);
    pad_to_stride(&mut features, config.stride, config.padding_value);
    Ok(stack_strided_features(&features, config.stride))
}

fn validate_audio_config(config: &AudioFrontendConfig) -> Result<()> {
    if config.n_fft == 0 || config.win_length == 0 || config.hop_length == 0 {
        bail!("n_fft, win_length and hop_length must be greater than zero");
    }
    if config.win_length > config.n_fft {
        bail!(
            "win_length must be <= n_fft, got win_length={} n_fft={}",
            config.win_length,
            config.n_fft
        );
    }
    Ok(())
}

fn normalize_signal(waveform: &mut [f32]) {
    if waveform.is_empty() {
        return;
    }
    let mean = waveform.iter().copied().sum::<f32>() / waveform.len() as f32;
    let mut max_abs = 0.0f32;
    for value in waveform.iter_mut() {
        *value -= mean;
        max_abs = max_abs.max(value.abs());
    }
    let scale = max_abs.max(1e-6);
    for value in waveform.iter_mut() {
        *value /= scale;
    }
}

fn apply_waveform_preemphasis(waveform: &mut [f32], coefficient: f32) {
    if waveform.len() < 2 {
        return;
    }
    for index in (1..waveform.len()).rev() {
        waveform[index] -= coefficient * waveform[index - 1];
    }
}

fn resample_to_sample_rate(input: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if input.is_empty() || src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        return Ok(input.to_vec());
    }
    debug!(
        "resampling audio with rubato src_rate={} dst_rate={} input_samples={}",
        src_rate,
        dst_rate,
        input.len()
    );

    let mut resampler = Fft::<f32>::new(
        src_rate as usize,
        dst_rate as usize,
        1024,
        2,
        1,
        FixedSync::Both,
    )
    .with_context(|| {
        format!("failed to create Rubato resampler from {src_rate} Hz to {dst_rate} Hz")
    })?;

    let input_adapter = InterleavedSlice::new(input, 1, input.len())
        .context("failed to wrap mono input for Rubato")?;
    let output_capacity = resampler.process_all_needed_output_len(input.len());
    let mut output = vec![0.0f32; output_capacity];
    let mut output_adapter = InterleavedSlice::new_mut(&mut output, 1, output_capacity)
        .context("failed to wrap mono output for Rubato")?;
    let (_input_frames, output_frames) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, input.len(), None)
        .with_context(|| format!("Rubato resampling failed from {src_rate} Hz to {dst_rate} Hz"))?;
    output.truncate(output_frames);
    debug!(
        "resampled audio with rubato src_rate={} dst_rate={} output_samples={}",
        src_rate,
        dst_rate,
        output.len()
    );
    Ok(output)
}

fn padded_hann_window(win_length: usize, frame_length: usize, periodic: bool) -> Vec<f32> {
    let source = hann_window(win_length, periodic);
    if win_length == frame_length {
        return source;
    }
    let mut padded = vec![0.0; frame_length];
    let offset = (frame_length - win_length) / 2;
    padded[offset..offset + win_length].copy_from_slice(&source);
    padded
}

fn hann_window(length: usize, periodic: bool) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    let denominator = if periodic {
        length as f32
    } else {
        (length - 1) as f32
    };
    (0..length)
        .map(|index| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * index as f32) / denominator).cos())
        .collect()
}

struct SpectrogramOptions<'a> {
    frame_length: usize,
    hop_length: usize,
    fft_length: usize,
    center: bool,
    window: &'a [f32],
    remove_dc_offset: bool,
    frame_preemphasis: Option<f32>,
}

fn power_spectrogram(waveform: &[f32], options: SpectrogramOptions<'_>) -> Result<Vec<Vec<f32>>> {
    let SpectrogramOptions {
        frame_length,
        hop_length,
        fft_length,
        center,
        window,
        remove_dc_offset,
        frame_preemphasis,
    } = options;
    if frame_length == 0 || hop_length == 0 || fft_length < frame_length {
        bail!("invalid spectrogram dimensions");
    }
    if window.len() != frame_length {
        bail!(
            "window length must equal frame_length, got {} vs {}",
            window.len(),
            frame_length
        );
    }
    let padded = if center {
        reflect_pad(waveform, frame_length / 2)
    } else {
        waveform.to_vec()
    };
    if padded.len() < frame_length {
        return Ok(Vec::new());
    }
    let num_frames = 1 + (padded.len() - frame_length) / hop_length;
    let num_bins = fft_length / 2 + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_length);
    let mut output = Vec::with_capacity(num_frames);
    let mut frame = vec![0.0f32; frame_length];
    let mut buffer = vec![Complex32::new(0.0, 0.0); fft_length];

    for frame_index in 0..num_frames {
        let start = frame_index * hop_length;
        frame.copy_from_slice(&padded[start..start + frame_length]);
        if remove_dc_offset {
            let mean = frame.iter().copied().sum::<f32>() / frame.len() as f32;
            for value in frame.iter_mut() {
                *value -= mean;
            }
        }
        if let Some(coefficient) = frame_preemphasis {
            for index in (1..frame.len()).rev() {
                frame[index] -= coefficient * frame[index - 1];
            }
            frame[0] *= 1.0 - coefficient;
        }
        for value in buffer.iter_mut() {
            *value = Complex32::new(0.0, 0.0);
        }
        for index in 0..frame_length {
            buffer[index].re = frame[index] * window[index];
        }
        fft.process(&mut buffer);
        let mut bins = Vec::with_capacity(num_bins);
        for value in buffer.iter().take(num_bins) {
            bins.push(value.re.mul_add(value.re, value.im * value.im));
        }
        output.push(bins);
    }

    Ok(output)
}

fn reflect_pad(input: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 || input.is_empty() {
        return input.to_vec();
    }
    if input.len() == 1 {
        let mut output = vec![input[0]; input.len() + pad * 2];
        output[pad] = input[0];
        return output;
    }
    let len = input.len() as isize;
    let mut output = Vec::with_capacity(input.len() + pad * 2);
    for padded_index in 0..output.capacity() {
        let source_index = padded_index as isize - pad as isize;
        output.push(input[reflect_index(source_index, len) as usize]);
    }
    output
}

fn reflect_index(mut index: isize, len: isize) -> isize {
    while index < 0 || index >= len {
        if index < 0 {
            index = -index;
        } else {
            index = 2 * len - 2 - index;
        }
    }
    index
}

#[derive(Debug, Clone, Copy)]
enum MelScale {
    Htk,
    Kaldi,
}

fn mel_filter_bank(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    sample_rate: u32,
    min_frequency: f32,
    max_frequency: f32,
    mel_scale: MelScale,
    triangularize_in_mel_space: bool,
) -> Vec<Vec<f32>> {
    let min_mel = hz_to_mel(min_frequency, mel_scale);
    let max_mel = hz_to_mel(max_frequency, mel_scale);
    let mel_points: Vec<f32> = (0..num_mel_filters + 2)
        .map(|index| min_mel + (max_mel - min_mel) * index as f32 / (num_mel_filters + 1) as f32)
        .collect();
    let freq_points: Vec<f32> = mel_points
        .iter()
        .map(|mel| mel_to_hz(*mel, mel_scale))
        .collect();
    let all_freqs: Vec<f32> = (0..num_frequency_bins)
        .map(|index| {
            index as f32 * (sample_rate as f32 / 2.0) / (num_frequency_bins - 1).max(1) as f32
        })
        .collect();
    let all_mels: Vec<f32> = all_freqs
        .iter()
        .map(|frequency| hz_to_mel(*frequency, mel_scale))
        .collect();
    let mut filters = vec![vec![0.0; num_frequency_bins]; num_mel_filters];

    for mel_index in 0..num_mel_filters {
        let (left, center, right, coordinates) = if triangularize_in_mel_space {
            (
                mel_points[mel_index],
                mel_points[mel_index + 1],
                mel_points[mel_index + 2],
                &all_mels,
            )
        } else {
            (
                freq_points[mel_index],
                freq_points[mel_index + 1],
                freq_points[mel_index + 2],
                &all_freqs,
            )
        };
        for (bin_index, coordinate) in coordinates.iter().enumerate() {
            let lower = (*coordinate - left) / (center - left).max(f32::MIN_POSITIVE);
            let upper = (right - *coordinate) / (right - center).max(f32::MIN_POSITIVE);
            filters[mel_index][bin_index] = lower.min(upper).max(0.0);
        }
    }
    filters
}

fn hz_to_mel(frequency: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + frequency / 700.0).log10(),
        MelScale::Kaldi => 1127.0 * (1.0 + frequency / 700.0).ln(),
    }
}

fn hz_to_mel_f64(frequency: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + frequency / 700.0).log10(),
        MelScale::Kaldi => 1127.0 * (1.0 + frequency / 700.0).ln(),
    }
}

fn mel_to_hz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 700.0 * (10f32.powf(mel / 2595.0) - 1.0),
        MelScale::Kaldi => 700.0 * ((mel / 1127.0).exp() - 1.0),
    }
}

fn log_mel_from_power(powers: &[Vec<f32>], filters: &[Vec<f32>], floor: f32) -> FeatureMatrix {
    let rows = powers.len();
    let cols = filters.len();
    let mut values = Vec::with_capacity(rows * cols);
    for frame in powers {
        for filter in filters {
            let mel_energy = frame
                .iter()
                .zip(filter.iter())
                .map(|(power, weight)| power * weight)
                .sum::<f32>()
                .max(floor);
            values.push(mel_energy.ln());
        }
    }
    FeatureMatrix { rows, cols, values }
}

struct SeamlessM4TFbankOptions {
    sample_rate: u32,
    frame_length: usize,
    hop_length: usize,
    fft_length: usize,
    num_mel_bins: usize,
    mel_floor: f64,
    preemphasis: f64,
}

fn seamless_m4t_log_mel_features(
    waveform: &[f32],
    options: SeamlessM4TFbankOptions,
) -> Result<FeatureMatrix> {
    if options.frame_length == 0
        || options.hop_length == 0
        || options.fft_length < options.frame_length
    {
        bail!("invalid SeamlessM4T fbank dimensions");
    }
    if waveform.len() < options.frame_length {
        return Ok(FeatureMatrix {
            rows: 0,
            cols: options.num_mel_bins,
            values: Vec::new(),
        });
    }

    let powers = seamless_m4t_power_spectrogram(waveform, &options)?;
    let filters = seamless_m4t_mel_filter_bank(
        options.fft_length / 2 + 1,
        options.num_mel_bins,
        options.sample_rate,
        20.0,
        (options.sample_rate / 2) as f64,
    );
    let rows = powers.len();
    let cols = options.num_mel_bins;
    let mut values = Vec::with_capacity(rows * cols);
    for frame in &powers {
        for filter in &filters {
            let mel_energy = frame
                .iter()
                .zip(filter.iter())
                .map(|(power, weight)| power * weight)
                .sum::<f64>()
                .max(options.mel_floor);
            values.push(mel_energy.ln() as f32);
        }
    }
    Ok(FeatureMatrix { rows, cols, values })
}

fn seamless_m4t_power_spectrogram(
    waveform: &[f32],
    options: &SeamlessM4TFbankOptions,
) -> Result<Vec<Vec<f64>>> {
    let num_frames = 1 + (waveform.len() - options.frame_length) / options.hop_length;
    let num_bins = options.fft_length / 2 + 1;
    let window = povey_window_f64(options.frame_length);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(options.fft_length);
    let mut output = Vec::with_capacity(num_frames);
    let mut buffer = vec![Complex64::new(0.0, 0.0); options.fft_length];

    for frame_index in 0..num_frames {
        let start = frame_index * options.hop_length;
        for value in buffer.iter_mut() {
            *value = Complex64::new(0.0, 0.0);
        }
        for index in 0..options.frame_length {
            buffer[index].re = waveform[start + index] as f64;
        }
        let mean = buffer[..options.frame_length]
            .iter()
            .map(|value| value.re)
            .sum::<f64>()
            / options.frame_length as f64;
        for value in &mut buffer[..options.frame_length] {
            value.re -= mean;
        }
        for index in (1..options.frame_length).rev() {
            buffer[index].re -= options.preemphasis * buffer[index - 1].re;
        }
        buffer[0].re *= 1.0 - options.preemphasis;
        for index in 0..options.frame_length {
            buffer[index].re *= window[index];
        }

        fft.process(&mut buffer);
        let mut bins = Vec::with_capacity(num_bins);
        for value in buffer.iter().take(num_bins) {
            // Transformers stores the FFT result in complex64 before taking power.
            let real = value.re as f32 as f64;
            let imaginary = value.im as f32 as f64;
            bins.push(real.mul_add(real, imaginary * imaginary));
        }
        output.push(bins);
    }

    Ok(output)
}

fn povey_window_f64(length: usize) -> Vec<f64> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    let denominator = (length - 1) as f64;
    (0..length)
        .map(|index| {
            let hann =
                0.5 - 0.5 * ((2.0 * std::f64::consts::PI * index as f64) / denominator).cos();
            hann.powf(0.85)
        })
        .collect()
}

fn seamless_m4t_mel_filter_bank(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    sample_rate: u32,
    min_frequency: f64,
    max_frequency: f64,
) -> Vec<Vec<f64>> {
    let min_mel = hz_to_mel_f64(min_frequency, MelScale::Kaldi);
    let max_mel = hz_to_mel_f64(max_frequency, MelScale::Kaldi);
    let filter_freqs: Vec<f64> = (0..num_mel_filters + 2)
        .map(|index| min_mel + (max_mel - min_mel) * index as f64 / (num_mel_filters + 1) as f64)
        .collect();
    let fft_bin_width = sample_rate as f64 / ((num_frequency_bins - 1) * 2) as f64;
    let fft_freqs: Vec<f64> = (0..num_frequency_bins)
        .map(|index| hz_to_mel_f64(fft_bin_width * index as f64, MelScale::Kaldi))
        .collect();
    let mut filters = vec![vec![0.0; num_frequency_bins]; num_mel_filters];

    for mel_index in 0..num_mel_filters {
        let left = filter_freqs[mel_index];
        let center = filter_freqs[mel_index + 1];
        let right = filter_freqs[mel_index + 2];
        let left_width = center - left;
        let right_width = right - center;
        for (bin_index, fft_freq) in fft_freqs.iter().enumerate() {
            let down_slope = (*fft_freq - left) / left_width;
            let up_slope = (right - *fft_freq) / right_width;
            filters[mel_index][bin_index] = down_slope.min(up_slope).max(0.0);
        }
    }
    filters
}

fn normalize_columns(features: &mut FeatureMatrix, min_std: f32, unbiased: bool) {
    if features.rows == 0 || features.cols == 0 {
        return;
    }
    for col in 0..features.cols {
        let mean = (0..features.rows)
            .map(|row| features.values[row * features.cols + col])
            .sum::<f32>()
            / features.rows as f32;
        let divisor = if unbiased && features.rows > 1 {
            (features.rows - 1) as f32
        } else {
            features.rows as f32
        };
        let variance = (0..features.rows)
            .map(|row| {
                let delta = features.values[row * features.cols + col] - mean;
                delta * delta
            })
            .sum::<f32>()
            / divisor.max(1.0);
        let std = variance.sqrt().max(min_std);
        for row in 0..features.rows {
            let index = row * features.cols + col;
            features.values[index] = (features.values[index] - mean) / std;
        }
    }
}

fn normalize_columns_with_variance_epsilon(
    features: &mut FeatureMatrix,
    variance_epsilon: f32,
    unbiased: bool,
) {
    if features.rows == 0 || features.cols == 0 {
        return;
    }
    for col in 0..features.cols {
        let mean = (0..features.rows)
            .map(|row| features.values[row * features.cols + col])
            .sum::<f32>()
            / features.rows as f32;
        let divisor = if unbiased && features.rows > 1 {
            (features.rows - 1) as f32
        } else {
            features.rows as f32
        };
        let variance = (0..features.rows)
            .map(|row| {
                let delta = features.values[row * features.cols + col] - mean;
                delta * delta
            })
            .sum::<f32>()
            / divisor.max(1.0);
        let std = (variance + variance_epsilon).sqrt();
        for row in 0..features.rows {
            let index = row * features.cols + col;
            features.values[index] = (features.values[index] - mean) / std;
        }
    }
}

fn normalize_rows(features: &mut FeatureMatrix, min_std: f32) {
    if features.rows == 0 || features.cols == 0 {
        return;
    }
    for row in 0..features.rows {
        let start = row * features.cols;
        let end = start + features.cols;
        let mean = features.values[start..end].iter().copied().sum::<f32>() / features.cols as f32;
        let variance = features.values[start..end]
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / features.cols as f32;
        let std = variance.sqrt().max(min_std);
        for value in &mut features.values[start..end] {
            *value = (*value - mean) / std;
        }
    }
}

fn pad_to_stride(features: &mut FeatureMatrix, stride: usize, padding_value: f32) {
    if stride <= 1 || features.rows.is_multiple_of(stride) {
        return;
    }
    let missing = stride - (features.rows % stride);
    features
        .values
        .extend(std::iter::repeat_n(padding_value, missing * features.cols));
    features.rows += missing;
}

fn stack_strided_features(features: &FeatureMatrix, stride: usize) -> FeatureMatrix {
    if stride <= 1 {
        return FeatureMatrix {
            rows: features.rows,
            cols: features.cols,
            values: features.values.clone(),
        };
    }
    let rows = features.rows / stride;
    let cols = features.cols * stride;
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for stride_index in 0..stride {
            let source_row = row * stride + stride_index;
            let start = source_row * features.cols;
            values.extend_from_slice(&features.values[start..start + features.cols]);
        }
    }
    FeatureMatrix { rows, cols, values }
}

fn encode_feature_payload(features: &FeatureMatrix) -> Result<Vec<u8>> {
    let rows: u32 = features
        .rows
        .try_into()
        .context("feature row count does not fit into u32")?;
    let cols: u32 = features
        .cols
        .try_into()
        .context("feature column count does not fit into u32")?;
    let mut payload = Vec::with_capacity(RUST_PAYLOAD_MAGIC.len() + 8 + features.values.len() * 4);
    payload.extend_from_slice(RUST_PAYLOAD_MAGIC);
    payload.extend_from_slice(&rows.to_le_bytes());
    payload.extend_from_slice(&cols.to_le_bytes());
    for value in &features.values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    Ok(payload)
}

fn cache_key(utterance_id: &str, frontend_hash: &str) -> String {
    let digest = Sha256::digest(format!("{utterance_id}:{frontend_hash}").as_bytes());
    hex_full(&digest)
}

fn hex_prefix(bytes: &[u8], chars: usize) -> String {
    hex_full(bytes).chars().take(chars).collect()
}

fn hex_full(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

struct ShardedCacheWriter {
    shard_dir: PathBuf,
    num_shards: usize,
    rows_per_part: usize,
    pending: HashMap<usize, Vec<CacheRow>>,
    counters: HashMap<usize, usize>,
    schema: Arc<Schema>,
}

impl ShardedCacheWriter {
    fn new(root: &Path, num_shards: usize, rows_per_part: usize) -> Result<Self> {
        let shard_dir = root.join("feature_shards");
        fs::create_dir_all(&shard_dir)
            .with_context(|| format!("failed to create {}", shard_dir.display()))?;
        debug!(
            "initialized sharded cache writer shard_dir={} num_shards={} rows_per_part={}",
            shard_dir.display(),
            num_shards,
            rows_per_part
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("payload", DataType::Binary, true),
            Field::new("deleted", DataType::Boolean, false),
        ]));
        Ok(Self {
            shard_dir,
            num_shards,
            rows_per_part,
            pending: HashMap::new(),
            counters: HashMap::new(),
            schema,
        })
    }

    fn push(&mut self, row: CacheRow) -> Result<()> {
        let shard_index = shard_index(&row.key, self.num_shards)?;
        let rows = self.pending.entry(shard_index).or_default();
        rows.push(row);
        if rows.len() >= self.rows_per_part {
            self.flush_shard(shard_index)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        let shard_indices: Vec<usize> = self.pending.keys().copied().collect();
        debug!("flushing pending shards count={}", shard_indices.len());
        for shard_index in shard_indices {
            self.flush_shard(shard_index)?;
        }
        Ok(())
    }

    fn flush_shard(&mut self, shard_index: usize) -> Result<()> {
        let rows = self.pending.remove(&shard_index).unwrap_or_default();
        if rows.is_empty() {
            return Ok(());
        }
        let row_count = rows.len();
        let output_dir = self.shard_dir.join(format!("features_{shard_index:02}"));
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("failed to create {}", output_dir.display()))?;
        let part_counter = self.counters.entry(shard_index).or_insert(0);
        *part_counter += 1;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let output_path = output_dir.join(format!(
            "part_rust_{}_{}_{:06}.parquet",
            std::process::id(),
            now,
            *part_counter
        ));

        let mut key_builder = StringBuilder::new();
        let mut payload_builder = BinaryBuilder::new();
        let mut deleted_builder = BooleanBuilder::new();
        for row in rows {
            key_builder.append_value(row.key);
            payload_builder.append_value(row.payload);
            deleted_builder.append_value(false);
        }
        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(key_builder.finish()) as ArrayRef,
                Arc::new(payload_builder.finish()) as ArrayRef,
                Arc::new(deleted_builder.finish()) as ArrayRef,
            ],
        )?;
        let file = File::create(&output_path)
            .with_context(|| format!("failed to create {}", output_path.display()))?;
        let mut writer = ArrowWriter::try_new(file, self.schema.clone(), None)?;
        writer.write(&batch)?;
        writer.close()?;
        debug!(
            "flushed cache shard={} rows={} path={}",
            shard_index,
            row_count,
            output_path.display()
        );
        Ok(())
    }
}

fn shard_index(key: &str, num_shards: usize) -> Result<usize> {
    if key.len() < 8 {
        bail!("cache key is shorter than 8 hex characters: {key}");
    }
    let prefix = u32::from_str_radix(&key[..8], 16)?;
    Ok(prefix as usize % num_shards)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squeezeformer_config_hash_matches_python_repr_contract() {
        let cli = Cli::parse_from(["test", "--input", "in.parquet", "--cache-dir", "cache"]);
        let config = FrontendConfig::from_cli(&cli);
        assert_eq!(
            config.config_repr(),
            "{'featurizer': {'sample_rate': 16000, 'n_fft': 400, 'win_length': 400, 'n_mels': 80, 'backend': 'torchaudio', 'preemphasis': 0.97, 'normalize_signal': True, 'normalize_feature': True, 'normalize_per_frame': False, 'hop_length': 160}}"
        );
        assert_eq!(config.frontend_hash(), "0a48384efcf3");
    }

    #[test]
    fn zipformer_config_uses_paper_defaults() {
        let cli = Cli::parse_from([
            "test",
            "--input",
            "in.parquet",
            "--cache-dir",
            "cache",
            "--frontend",
            "zipformer",
        ]);
        let config = FrontendConfig::from_cli(&cli);
        assert!(config.config_repr().contains("'preemphasis': 0.0"));
        assert!(config.config_repr().contains("'normalize_signal': False"));
        assert!(config.config_repr().contains("'normalize_feature': False"));
        assert_eq!(config.frontend_hash(), "4d9c4bc8f09a");
    }

    #[test]
    fn w2v_bert_config_hash_matches_python_repr_contract() {
        let cli = Cli::parse_from([
            "test",
            "--input",
            "in.parquet",
            "--cache-dir",
            "cache",
            "--frontend",
            "w2v-bert",
        ]);
        let config = FrontendConfig::from_cli(&cli);
        assert_eq!(
            config.config_repr(),
            "{'featurizer': {'type': 'w2v_bert', 'model_source': 'facebook/w2v-bert-2.0', 'sample_rate': 16000, 'feature_size': 80, 'stride': 2, 'feature_dim': 160, 'padding_value': 1.0}}"
        );
        assert_eq!(config.frontend_hash(), "c62e513533e1");
    }

    #[test]
    fn payload_roundtrip_header_is_stable() {
        let features = FeatureMatrix {
            rows: 2,
            cols: 3,
            values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let payload = encode_feature_payload(&features).unwrap();
        assert_eq!(&payload[..8], RUST_PAYLOAD_MAGIC);
        assert_eq!(u32::from_le_bytes(payload[8..12].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(payload[12..16].try_into().unwrap()), 3);
        assert_eq!(payload.len(), 16 + 6 * 4);
    }

    #[test]
    fn input_folder_discovers_parquet_files_recursively() {
        let root = std::env::temp_dir().join(format!(
            "sfcw_input_folder_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let nested = root.join("nested");
        fs::create_dir_all(&nested).unwrap();
        File::create(root.join("a.parquet")).unwrap();
        File::create(nested.join("b.parquet")).unwrap();
        File::create(root.join("ignored.txt")).unwrap();
        let cli = Cli::parse_from(vec![
            "test".to_string(),
            "--input-folder".to_string(),
            root.to_string_lossy().into_owned(),
            "--cache-dir".to_string(),
            "cache".to_string(),
        ]);

        let paths = resolve_input_paths(&cli).unwrap();

        assert_eq!(paths.len(), 2);
        assert!(paths[0].ends_with("a.parquet"));
        assert!(paths[1].ends_with("b.parquet"));
        assert_eq!(default_source_base(&cli, &paths), root);
        fs::remove_dir_all(default_source_base(&cli, &paths)).unwrap();
    }

    #[test]
    fn rubato_resampler_converts_sample_rate() {
        let waveform: Vec<f32> = (0..4_800)
            .map(|index| (index as f32 * 0.01).sin())
            .collect();

        let resampled = resample_to_sample_rate(&waveform, 48_000, 16_000).unwrap();

        assert_eq!(resampled.len(), 1_600);
        assert!(resampled.iter().any(|value| value.abs() > 1e-5));
    }

    #[test]
    fn audio_frontend_produces_time_by_mel_matrix() {
        let config = AudioFrontendConfig {
            sample_rate: 16_000,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            n_mels: 80,
            preemphasis: 0.97,
            normalize_signal: true,
            normalize_feature: true,
            normalize_per_frame: false,
        };
        let mut waveform = vec![0.0f32; 320];
        waveform[10] = 0.5;
        let features = compute_audio_featurizer_features(&mut waveform, 16_000, &config).unwrap();
        assert_eq!(features.cols, 80);
        assert_eq!(features.rows, 3);
        assert_eq!(features.values.len(), features.rows * features.cols);
    }

    #[test]
    fn w2v_frontend_stacks_stride_frames() {
        let config = W2vBertFrontendConfig {
            model_source: "facebook/w2v-bert-2.0".to_string(),
            sample_rate: 16_000,
            feature_size: 80,
            stride: 2,
            feature_dim: 160,
            padding_value: 1.0,
        };
        let mut waveform = vec![0.0f32; 800];
        waveform[20] = 0.5;
        let features = compute_w2v_bert_features(&mut waveform, 16_000, &config).unwrap();
        assert_eq!(features.cols, 160);
        assert!(features.rows >= 1);
        assert_eq!(features.values.len(), features.rows * features.cols);
    }
}
