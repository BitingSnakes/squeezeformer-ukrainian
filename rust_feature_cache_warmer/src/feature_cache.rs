use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use arrow::array::{Array, StructArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use clap::{ArgAction, Parser, ValueEnum};
use log::{debug, error, info, trace, warn};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::arrow_utils::{column_by_name, scalar_as_bytes, scalar_as_string, struct_child};
use crate::audio::{decode_audio, AudioSource};
use crate::cache::{cache_key, encode_feature_payload, CacheRow, ShardedCacheWriter};
use crate::frontend::{
    compute_features, squeezeformer_frontend_config, w2v_bert_frontend_config,
    zipformer_frontend_config, AudioFrontendConfig, FrontendConfig,
};
use crate::util::resolve_path;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum FrontendKind {
    Squeezeformer,
    Zipformer,
    W2vBert,
}

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about,
    after_help = "Subcommands:\n  record-cache        Build Python-compatible train/validation record cache files\n\nRun `feature-cache-warmer record-cache --help` for record-cache options."
)]
pub(crate) struct Cli {
    /// Input parquet manifest. Mutually exclusive with --input-folder.
    #[arg(long)]
    pub(crate) input: Option<PathBuf>,

    /// Directory containing parquet manifests to process recursively. Repeat to combine folders.
    #[arg(long = "input-folder", action = ArgAction::Append)]
    pub(crate) input_folders: Vec<PathBuf>,

    /// Python-compatible disk-backed record cache JSONL file, such as train.jsonl.
    #[arg(long)]
    pub(crate) input_record_cache: Option<PathBuf>,

    /// Split cache root to write. The crate creates feature_shards/features_XX below it.
    #[arg(long)]
    pub(crate) cache_dir: PathBuf,

    /// Resolve relative audio paths against this directory. Defaults to the input parquet parent
    /// for --input, or each folder root for --input-folder.
    #[arg(long)]
    pub(crate) source_base: Option<PathBuf>,

    /// Frontend defaults to mirror.
    #[arg(long, value_enum, default_value_t = FrontendKind::Squeezeformer)]
    pub(crate) frontend: FrontendKind,

    /// Optional row limit for smoke runs.
    #[arg(long)]
    pub(crate) limit: Option<usize>,

    /// Number of cache shards. Must match Python ShardedParquetFeatureCache.
    #[arg(long, default_value_t = 64)]
    pub(crate) num_shards: usize,

    /// Input parquet record batch size.
    #[arg(long, default_value_t = 1024)]
    pub(crate) batch_size: usize,

    /// Flush a parquet part for a shard after this many rows.
    #[arg(long, default_value_t = 256)]
    pub(crate) rows_per_part: usize,

    /// Stop on the first row error instead of skipping bad rows.
    #[arg(long, default_value_t = false)]
    pub(crate) fail_fast: bool,

    /// Parallel feature extraction threads. Use 0 for Rayon's default.
    #[arg(long, default_value_t = 0)]
    pub(crate) threads: usize,

    /// Disable ffmpeg fallback when Symphonia cannot decode a codec such as Opus.
    #[arg(long, default_value_t = false)]
    pub(crate) no_ffmpeg_fallback: bool,

    #[arg(long)]
    pub(crate) sample_rate: Option<u32>,
    #[arg(long)]
    pub(crate) n_fft: Option<usize>,
    #[arg(long)]
    pub(crate) win_length: Option<usize>,
    #[arg(long)]
    pub(crate) hop_length: Option<usize>,
    #[arg(long)]
    pub(crate) n_mels: Option<usize>,
    #[arg(long)]
    pub(crate) preemphasis: Option<f32>,
    #[arg(long)]
    pub(crate) normalize_signal: Option<bool>,
    #[arg(long)]
    pub(crate) normalize_feature: Option<bool>,
    #[arg(long)]
    pub(crate) normalize_per_frame: Option<bool>,

    /// W2V-BERT model source used only for cache key compatibility.
    #[arg(long, default_value = "facebook/w2v-bert-2.0")]
    pub(crate) w2v_model_source: String,
    #[arg(long, default_value_t = 80)]
    pub(crate) w2v_feature_size: usize,
    #[arg(long, default_value_t = 2)]
    pub(crate) w2v_stride: usize,
    #[arg(long, default_value_t = 1.0)]
    pub(crate) w2v_padding_value: f32,
}

#[derive(Default)]
struct Counters {
    scanned: usize,
    written: usize,
    skipped: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct InputManifest {
    pub(crate) path: PathBuf,
    pub(crate) source_base: PathBuf,
}

pub(crate) fn run_feature_cache_cli(cli: Cli) -> Result<()> {
    if cli.num_shards == 0 {
        bail!("--num-shards must be greater than zero");
    }
    if cli.rows_per_part == 0 {
        bail!("--rows-per-part must be greater than zero");
    }

    let frontend = FrontendConfig::from_cli(&cli);
    let frontend_hash = frontend.frontend_hash();
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

    if let Some(record_cache_path) = &cli.input_record_cache {
        info!(
            "starting record-cache feature warm input_record_cache={} cache_dir={} frontend={:?} frontend_hash={} batch_size={} rows_per_part={} num_shards={} fail_fast={} ffmpeg_fallback={}",
            record_cache_path.display(),
            cli.cache_dir.display(),
            cli.frontend,
            frontend_hash,
            cli.batch_size,
            cli.rows_per_part,
            cli.num_shards,
            cli.fail_fast,
            !cli.no_ffmpeg_fallback
        );
        warm_record_cache_features(
            record_cache_path,
            &cli,
            &frontend,
            &frontend_hash,
            &pool,
            &mut writer,
            &mut counters,
        )?;
        writer.finish()?;
        info!(
            "complete scanned={} written={} skipped={}",
            counters.scanned, counters.written, counters.skipped
        );
        return Ok(());
    }

    let input_manifests = resolve_input_manifests(&cli)?;
    let explicit_source_base = cli
        .source_base
        .clone()
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|| "per-input".to_string());
    info!(
        "starting cache warm inputs={} cache_dir={} source_base={} frontend={:?} frontend_hash={} batch_size={} rows_per_part={} num_shards={} fail_fast={} ffmpeg_fallback={}",
        input_manifests.len(),
        cli.cache_dir.display(),
        explicit_source_base,
        cli.frontend,
        frontend_hash,
        cli.batch_size,
        cli.rows_per_part,
        cli.num_shards,
        cli.fail_fast,
        !cli.no_ffmpeg_fallback
    );
    debug!("resolved input parquet files: {:?}", input_manifests);

    'inputs: for input_manifest in input_manifests {
        let input_path = input_manifest.path;
        let source_base = cli
            .source_base
            .clone()
            .unwrap_or(input_manifest.source_base);
        info!(
            "warming input {} source_base={}",
            input_path.display(),
            source_base.display()
        );
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

fn warm_record_cache_features(
    record_cache_path: &Path,
    cli: &Cli,
    frontend: &FrontendConfig,
    frontend_hash: &str,
    pool: &rayon::ThreadPool,
    writer: &mut ShardedCacheWriter,
    counters: &mut Counters,
) -> Result<()> {
    if !record_cache_path.is_file() {
        bail!(
            "--input-record-cache must point to a JSONL file: {}",
            record_cache_path.display()
        );
    }
    let input = File::open(record_cache_path).with_context(|| {
        format!(
            "failed to open record cache {}",
            record_cache_path.display()
        )
    })?;
    let reader = BufReader::new(input);
    let mut rows = Vec::with_capacity(cli.batch_size.max(1));
    let mut scanned_before_batch = counters.scanned;
    for line_result in reader.lines() {
        let line = line_result?;
        counters.scanned += 1;
        if cli.limit.is_some_and(|limit| counters.scanned > limit) {
            counters.scanned -= 1;
            break;
        }
        rows.push((counters.scanned, line));
        if rows.len() >= cli.batch_size.max(1) {
            process_record_cache_feature_batch(
                std::mem::take(&mut rows),
                record_cache_path,
                cli,
                frontend,
                frontend_hash,
                pool,
                writer,
                counters,
            )?;
            scanned_before_batch = counters.scanned;
        }
    }
    if !rows.is_empty() {
        process_record_cache_feature_batch(
            rows,
            record_cache_path,
            cli,
            frontend,
            frontend_hash,
            pool,
            writer,
            counters,
        )?;
    }
    debug!(
        "record cache feature warm consumed input={} scanned_before_last_batch={}",
        record_cache_path.display(),
        scanned_before_batch
    );
    Ok(())
}

fn process_record_cache_feature_batch(
    rows: Vec<(usize, String)>,
    record_cache_path: &Path,
    cli: &Cli,
    frontend: &FrontendConfig,
    frontend_hash: &str,
    pool: &rayon::ThreadPool,
    writer: &mut ShardedCacheWriter,
    counters: &mut Counters,
) -> Result<()> {
    let results: Vec<(usize, Result<Option<CacheRow>>)> = pool.install(|| {
        rows.into_par_iter()
            .map(|(scanned_row, line)| {
                let result = process_record_cache_feature_line(
                    &line,
                    scanned_row,
                    record_cache_path,
                    frontend,
                    frontend_hash,
                    !cli.no_ffmpeg_fallback,
                );
                (scanned_row, result)
            })
            .collect()
    });
    for (scanned_row, result) in results {
        match result {
            Ok(Some(cache_row)) => {
                writer.push(cache_row)?;
                counters.written += 1;
            }
            Ok(None) => {
                counters.skipped += 1;
                trace!(
                    "skipping record-cache row {}: no audio source found",
                    scanned_row
                );
            }
            Err(error) if cli.fail_fast => {
                error!("failed record-cache row {}: {error:#}", scanned_row);
                return Err(error);
            }
            Err(error) => {
                counters.skipped += 1;
                warn!("skipping record-cache row {}: {error:#}", scanned_row);
            }
        }
        if scanned_row % 1000 == 0 {
            info!(
                "progress scanned={} written={} skipped={}",
                scanned_row, counters.written, counters.skipped
            );
        }
    }
    Ok(())
}

fn process_record_cache_feature_line(
    line: &str,
    scanned_row: usize,
    record_cache_path: &Path,
    frontend: &FrontendConfig,
    frontend_hash: &str,
    ffmpeg_fallback: bool,
) -> Result<Option<CacheRow>> {
    let value: serde_json::Value = serde_json::from_str(line)
        .with_context(|| format!("failed to parse record-cache JSON row {scanned_row}"))?;
    let utterance_id = value
        .get("utterance_id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| scanned_row.to_string());
    let audio_path = value
        .get("audio_path")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let audio_blob_path = value
        .get("audio_blob_path")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty());
    let source = if let Some(blob_path) = audio_blob_path {
        let blob_path = resolve_record_cache_blob_path(record_cache_path, blob_path);
        let bytes = fs::read(&blob_path)
            .with_context(|| format!("failed to read audio blob {}", blob_path.display()))?;
        Some(AudioSource::Bytes(bytes, audio_path.clone()))
    } else if let Some(path) = audio_path {
        if path.starts_with("http://") || path.starts_with("https://") {
            bail!("remote audio URLs are not supported by the Rust warmer: {path}");
        }
        Some(AudioSource::Path(PathBuf::from(path)))
    } else {
        None
    };
    let Some(source) = source else {
        return Ok(None);
    };
    trace!(
        "record-cache row {} utterance_id={} source={}",
        scanned_row,
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
    Ok(Some(CacheRow {
        key: cache_key(&utterance_id, frontend_hash),
        payload: encode_feature_payload(&features)?,
    }))
}

fn resolve_record_cache_blob_path(record_cache_path: &Path, blob_path: &str) -> PathBuf {
    let path = PathBuf::from(blob_path);
    if path.is_absolute() {
        path
    } else {
        record_cache_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    }
}

pub(crate) fn resolve_input_manifests(cli: &Cli) -> Result<Vec<InputManifest>> {
    let input_modes = usize::from(cli.input.is_some())
        + usize::from(!cli.input_folders.is_empty())
        + usize::from(cli.input_record_cache.is_some());
    if input_modes != 1 {
        bail!("exactly one of --input, --input-folder, or --input-record-cache is required");
    }
    if let Some(input) = &cli.input {
        if !input.is_file() {
            bail!("--input must point to a parquet file: {}", input.display());
        }
        let source_base = input
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        return Ok(vec![InputManifest {
            path: input.clone(),
            source_base,
        }]);
    }
    if cli.input_folders.is_empty() {
        bail!("--input-record-cache is handled before parquet input resolution");
    }
    let mut inputs = Vec::new();
    for input_folder in &cli.input_folders {
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
        inputs.extend(paths.into_iter().map(|path| InputManifest {
            path,
            source_base: input_folder.clone(),
        }));
    }
    inputs.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(inputs)
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

impl FrontendConfig {
    pub(crate) fn from_cli(cli: &Cli) -> Self {
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
