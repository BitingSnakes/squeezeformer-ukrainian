use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::{
    extract_audio_features_from_samples, extract_w2v_bert_features_from_samples,
    squeezeformer_frontend_config, w2v_bert_frontend_config, zipformer_frontend_config,
    FeatureMatrix,
};

fn py_error(error: anyhow::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn mono_waveform(waveform: PyReadonlyArrayDyn<'_, f32>) -> PyResult<Vec<f32>> {
    let array = waveform.as_array();
    match array.ndim() {
        1 => Ok(array.iter().copied().collect()),
        2 => {
            let shape = array.shape();
            let channels = shape[0];
            let frames = shape[1];
            if channels == 0 || frames == 0 {
                return Ok(Vec::new());
            }
            let mut mono = Vec::with_capacity(frames);
            for frame in 0..frames {
                let mut sum = 0.0f32;
                for channel in 0..channels {
                    sum += *array
                        .get([channel, frame])
                        .ok_or_else(|| PyValueError::new_err("invalid waveform shape"))?;
                }
                mono.push(sum / channels as f32);
            }
            Ok(mono)
        }
        _ => Err(PyValueError::new_err(format!(
            "expected waveform with shape [time] or [channels, time], got {:?}",
            array.shape()
        ))),
    }
}

fn feature_matrix_to_pyarray<'py>(
    py: Python<'py>,
    features: FeatureMatrix,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let array = Array2::from_shape_vec((features.rows, features.cols), features.values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    waveform,
    sample_rate,
    *,
    target_sample_rate=16_000,
    n_fft=400,
    win_length=None,
    hop_length=160,
    n_mels=80,
    preemphasis=0.97,
    normalize_signal=true,
    normalize_feature=true,
    normalize_per_frame=false
))]
fn extract_squeezeformer<'py>(
    py: Python<'py>,
    waveform: PyReadonlyArrayDyn<'py, f32>,
    sample_rate: u32,
    target_sample_rate: u32,
    n_fft: usize,
    win_length: Option<usize>,
    hop_length: usize,
    n_mels: usize,
    preemphasis: f32,
    normalize_signal: bool,
    normalize_feature: bool,
    normalize_per_frame: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let waveform = mono_waveform(waveform)?;
    let mut config = squeezeformer_frontend_config();
    config.sample_rate = target_sample_rate;
    config.n_fft = n_fft;
    config.win_length = win_length.unwrap_or(n_fft);
    config.hop_length = hop_length;
    config.n_mels = n_mels;
    config.preemphasis = preemphasis;
    config.normalize_signal = normalize_signal;
    config.normalize_feature = normalize_feature;
    config.normalize_per_frame = normalize_per_frame;
    let features =
        extract_audio_features_from_samples(&waveform, sample_rate, &config).map_err(py_error)?;
    feature_matrix_to_pyarray(py, features)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    waveform,
    sample_rate,
    *,
    target_sample_rate=16_000,
    n_fft=400,
    win_length=None,
    hop_length=160,
    n_mels=80,
    preemphasis=0.0,
    normalize_signal=false,
    normalize_feature=false,
    normalize_per_frame=false
))]
fn extract_zipformer<'py>(
    py: Python<'py>,
    waveform: PyReadonlyArrayDyn<'py, f32>,
    sample_rate: u32,
    target_sample_rate: u32,
    n_fft: usize,
    win_length: Option<usize>,
    hop_length: usize,
    n_mels: usize,
    preemphasis: f32,
    normalize_signal: bool,
    normalize_feature: bool,
    normalize_per_frame: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let waveform = mono_waveform(waveform)?;
    let mut config = zipformer_frontend_config();
    config.sample_rate = target_sample_rate;
    config.n_fft = n_fft;
    config.win_length = win_length.unwrap_or(n_fft);
    config.hop_length = hop_length;
    config.n_mels = n_mels;
    config.preemphasis = preemphasis;
    config.normalize_signal = normalize_signal;
    config.normalize_feature = normalize_feature;
    config.normalize_per_frame = normalize_per_frame;
    let features =
        extract_audio_features_from_samples(&waveform, sample_rate, &config).map_err(py_error)?;
    feature_matrix_to_pyarray(py, features)
}

#[pyfunction]
#[pyo3(signature = (
    waveform,
    sample_rate,
    *,
    target_sample_rate=16_000,
    feature_size=80,
    stride=2,
    padding_value=1.0
))]
fn extract_w2v_bert<'py>(
    py: Python<'py>,
    waveform: PyReadonlyArrayDyn<'py, f32>,
    sample_rate: u32,
    target_sample_rate: u32,
    feature_size: usize,
    stride: usize,
    padding_value: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let waveform = mono_waveform(waveform)?;
    let config = w2v_bert_frontend_config(
        None,
        Some(target_sample_rate),
        Some(feature_size),
        Some(stride),
        Some(feature_size * stride.max(1)),
        Some(padding_value),
    );
    let features = extract_w2v_bert_features_from_samples(&waveform, sample_rate, &config)
        .map_err(py_error)?;
    feature_matrix_to_pyarray(py, features)
}

#[pymodule]
fn feature_cache_warmer_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_squeezeformer, m)?)?;
    m.add_function(wrap_pyfunction!(extract_zipformer, m)?)?;
    m.add_function(wrap_pyfunction!(extract_w2v_bert, m)?)?;
    Ok(())
}
