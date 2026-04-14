use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use arrow::array::Array;
use numpy::ndarray::Array2;
use numpy::IntoPyArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::arrow_utils::{column_by_name, scalar_as_bytes, scalar_as_string};
use crate::cache::RUST_PAYLOAD_MAGIC;

#[derive(Clone, Debug)]
struct FeatureLocation {
    path: PathBuf,
    row_index: usize,
}

#[pyclass]
pub(crate) struct RustParquetFeatureCacheReader {
    index: HashMap<String, FeatureLocation>,
}

#[pymethods]
impl RustParquetFeatureCacheReader {
    #[new]
    #[pyo3(signature = (cache_root, *, num_shards=64))]
    fn new(cache_root: PathBuf, num_shards: usize) -> PyResult<Self> {
        let index = build_feature_index(&cache_root, num_shards).map_err(py_error)?;
        Ok(Self { index })
    }

    fn len(&self) -> usize {
        self.index.len()
    }

    fn fetch_many<'py>(&self, py: Python<'py>, keys: Vec<String>) -> PyResult<Bound<'py, PyList>> {
        let mut output: Vec<Option<FeatureMatrixPayload>> = vec![None; keys.len()];
        let mut positions_by_path: HashMap<&Path, Vec<(usize, usize)>> = HashMap::new();
        for (position, key) in keys.iter().enumerate() {
            if let Some(location) = self.index.get(key) {
                positions_by_path
                    .entry(location.path.as_path())
                    .or_default()
                    .push((position, location.row_index));
            }
        }
        for (path, positions) in positions_by_path {
            let requested: HashMap<usize, Vec<usize>> =
                positions
                    .into_iter()
                    .fold(HashMap::new(), |mut acc, (position, row_index)| {
                        acc.entry(row_index).or_default().push(position);
                        acc
                    });
            let payloads =
                read_payloads_from_part(path, requested.keys().copied()).map_err(py_error)?;
            for (row_index, positions) in requested {
                let Some(payload) = payloads.get(&row_index) else {
                    continue;
                };
                let decoded = decode_feature_payload(payload).map_err(py_error)?;
                for position in positions {
                    output[position] = Some(decoded.clone());
                }
            }
        }

        let py_items = PyList::empty(py);
        for item in output {
            match item {
                Some(features) => {
                    let array =
                        Array2::from_shape_vec((features.rows, features.cols), features.values)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?;
                    py_items.append(array.into_pyarray(py))?;
                }
                None => py_items.append(py.None())?,
            }
        }
        Ok(py_items)
    }
}

#[derive(Clone)]
struct FeatureMatrixPayload {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

fn py_error(error: anyhow::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn build_feature_index(
    cache_root: &Path,
    num_shards: usize,
) -> Result<HashMap<String, FeatureLocation>> {
    let shard_root = cache_root.join("feature_shards");
    let mut index = HashMap::new();
    for shard_index in 0..num_shards {
        let shard_path = shard_root.join(format!("features_{shard_index:02}"));
        if !shard_path.is_dir() {
            continue;
        }
        let mut part_paths = fs::read_dir(&shard_path)
            .with_context(|| format!("failed to read {}", shard_path.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("part_") && name.ends_with(".parquet"))
            })
            .collect::<Vec<_>>();
        part_paths.sort_by_key(|path| {
            path.metadata()
                .and_then(|metadata| metadata.modified())
                .ok()
        });
        for part_path in part_paths {
            apply_part_to_index(&part_path, &mut index)?;
        }
    }
    Ok(index)
}

fn apply_part_to_index(path: &Path, index: &mut HashMap<String, FeatureLocation>) -> Result<()> {
    let input = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)?
        .with_batch_size(8192)
        .build()?;
    let mut global_row_index = 0usize;
    for batch_result in reader {
        let batch = batch_result?;
        let key_column = column_by_name(&batch, &["key"])
            .ok_or_else(|| anyhow!("{} has no key column", path.display()))?;
        let deleted_column = column_by_name(&batch, &["deleted"])
            .ok_or_else(|| anyhow!("{} has no deleted column", path.display()))?;
        for row_index in 0..batch.num_rows() {
            let Some(key) = scalar_as_string(key_column.as_ref(), row_index) else {
                global_row_index += 1;
                continue;
            };
            let deleted = scalar_as_bool(deleted_column.as_ref(), row_index).unwrap_or(false);
            if deleted {
                index.remove(&key);
            } else {
                index.insert(
                    key,
                    FeatureLocation {
                        path: path.to_path_buf(),
                        row_index: global_row_index,
                    },
                );
            }
            global_row_index += 1;
        }
    }
    Ok(())
}

fn read_payloads_from_part(
    path: &Path,
    requested_rows: impl Iterator<Item = usize>,
) -> Result<HashMap<usize, Vec<u8>>> {
    let requested = requested_rows.collect::<HashSet<_>>();
    let input = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)?
        .with_batch_size(8192)
        .build()?;
    let mut payloads = HashMap::new();
    let mut global_row_index = 0usize;
    for batch_result in reader {
        let batch = batch_result?;
        let payload_column = column_by_name(&batch, &["payload"])
            .ok_or_else(|| anyhow!("{} has no payload column", path.display()))?;
        for row_index in 0..batch.num_rows() {
            if !requested.contains(&global_row_index) || payloads.contains_key(&global_row_index) {
                global_row_index += 1;
                continue;
            }
            if let Some(payload) = scalar_as_bytes(payload_column.as_ref(), row_index) {
                payloads.insert(global_row_index, payload);
            }
            global_row_index += 1;
        }
        if payloads.len() == requested.len() {
            break;
        }
    }
    Ok(payloads)
}

fn scalar_as_bool(array: &dyn Array, row_index: usize) -> Option<bool> {
    if array.is_null(row_index) {
        return None;
    }
    array
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .map(|values| values.value(row_index))
}

fn decode_feature_payload(payload: &[u8]) -> Result<FeatureMatrixPayload> {
    if !payload.starts_with(RUST_PAYLOAD_MAGIC) {
        bail!("feature payload is not in Rust SFCF32L1 format");
    }
    if payload.len() < RUST_PAYLOAD_MAGIC.len() + 8 {
        bail!("feature payload is shorter than its header");
    }
    let rows = u32::from_le_bytes(
        payload[RUST_PAYLOAD_MAGIC.len()..RUST_PAYLOAD_MAGIC.len() + 4]
            .try_into()
            .expect("slice length checked"),
    ) as usize;
    let cols = u32::from_le_bytes(
        payload[RUST_PAYLOAD_MAGIC.len() + 4..RUST_PAYLOAD_MAGIC.len() + 8]
            .try_into()
            .expect("slice length checked"),
    ) as usize;
    let data = &payload[RUST_PAYLOAD_MAGIC.len() + 8..];
    let expected_len = rows
        .checked_mul(cols)
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(|| anyhow!("feature payload dimensions overflow"))?;
    if data.len() != expected_len {
        bail!(
            "feature payload has invalid size: got {} bytes, expected {}",
            data.len(),
            expected_len
        );
    }
    let values = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size is 4")))
        .collect();
    Ok(FeatureMatrixPayload { rows, cols, values })
}
