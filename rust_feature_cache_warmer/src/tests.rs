use std::fs::{self, File};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;

use crate::cache::{encode_feature_payload, RUST_PAYLOAD_MAGIC};
use crate::dsp::{
    compute_audio_featurizer_features, compute_w2v_bert_features, resample_to_sample_rate,
};
use crate::feature_cache::{resolve_input_manifests, Cli};
use crate::frontend::{AudioFrontendConfig, FeatureMatrix, FrontendConfig, W2vBertFrontendConfig};

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

    let inputs = resolve_input_manifests(&cli).unwrap();
    let paths: Vec<_> = inputs.iter().map(|input| input.path.as_path()).collect();

    assert_eq!(paths.len(), 2);
    assert!(paths[0].ends_with("a.parquet"));
    assert!(paths[1].ends_with("b.parquet"));
    assert!(inputs.iter().all(|input| input.source_base == root));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn repeated_input_folders_keep_per_folder_source_base() {
    let root = std::env::temp_dir().join(format!(
        "sfcw_multi_input_folder_test_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let first = root.join("first");
    let second = root.join("second");
    let nested = second.join("nested");
    fs::create_dir_all(&first).unwrap();
    fs::create_dir_all(&nested).unwrap();
    File::create(first.join("a.parquet")).unwrap();
    File::create(nested.join("b.parquet")).unwrap();
    let cli = Cli::parse_from(vec![
        "test".to_string(),
        "--input-folder".to_string(),
        first.to_string_lossy().into_owned(),
        "--input-folder".to_string(),
        second.to_string_lossy().into_owned(),
        "--cache-dir".to_string(),
        "cache".to_string(),
    ]);

    let inputs = resolve_input_manifests(&cli).unwrap();
    let paths: Vec<_> = inputs.iter().map(|input| input.path.as_path()).collect();

    assert_eq!(paths.len(), 2);
    assert_eq!(inputs.len(), 2);
    let first_input = inputs
        .iter()
        .find(|input| input.path.ends_with("a.parquet"))
        .unwrap();
    let second_input = inputs
        .iter()
        .find(|input| input.path.ends_with("b.parquet"))
        .unwrap();
    assert_eq!(first_input.source_base, first);
    assert_eq!(second_input.source_base, second);
    fs::remove_dir_all(root).unwrap();
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
