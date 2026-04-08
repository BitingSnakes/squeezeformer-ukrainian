from __future__ import annotations

import argparse
from functools import partial
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import torch
from tqdm.auto import tqdm

from squeezeformer_pytorch.data import (
    AudioFeaturizer,
    SpecAugment,
    WaveformAugment,
    feature_tensor_is_plausible,
    feature_cache_path,
    iter_records_from_source,
    load_audio,
    max_reasonable_feature_frames,
    normalize_feature_tensor,
    prevalidate_records,
)

_WORKER_FEATURIZER: AudioFeaturizer | None = None
_WORKER_SPECAUGMENT: SpecAugment | None = None
_WORKER_WAVEFORM_AUGMENT: WaveformAugment | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and cache log-mel features for a dataset split without training."
    )
    parser.add_argument("--dataset-repo", default="speech-uk/cv22")
    parser.add_argument(
        "--dataset-source",
        action="append",
        default=None,
        help=(
            "Dataset source to load. Repeat to combine multiple sources. Each source may be a "
            "Hugging Face dataset repo, a direct TSV/Parquet file path or URL, or a local "
            "directory with Common Voice-style TSV or Parquet manifests plus audio files. "
            "If omitted, --dataset-repo is used."
        ),
    )
    parser.add_argument(
        "--validation-dataset-source",
        action="append",
        default=None,
        help=(
            "Validation-only dataset source. Repeat to combine multiple sources. Each source may "
            "be a Hugging Face dataset repo, a direct TSV/Parquet file path or URL, or a local "
            "directory with Common Voice-style TSV or Parquet manifests plus audio files. When "
            "provided, validation extraction uses the full set of records from these sources, "
            "matching train.py behavior."
        ),
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Deterministic internal split to cache.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, (os.cpu_count() or 1)),
        help="Number of worker threads or processes to use for audio loading and feature extraction.",
    )
    parser.add_argument(
        "--parallelism",
        default="process",
        choices=["process", "thread"],
        help=(
            "Worker backend for feature extraction. Use 'process' for CPU-bound multi-core "
            "servers, or 'thread' when process spawning is undesirable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute features even if the cache file already exists.",
    )
    parser.add_argument(
        "--prevalidate-audio",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prevalidate-workers", type=int, default=4)
    parser.add_argument("--min-transcript-chars", type=int, default=1)
    parser.add_argument("--max-transcript-chars", type=int, default=5000)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--frontend-backend",
        default="torchaudio",
        choices=["torchaudio", "audioflux"],
    )
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--preemphasis", type=float, default=0.97)
    parser.add_argument(
        "--normalize-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-per-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-freq-masks", type=int, default=0)
    parser.add_argument("--freq-mask-param", type=int, default=27)
    parser.add_argument("--num-time-masks", type=int, default=0)
    parser.add_argument("--time-mask-max-ratio", type=float, default=0.05)
    parser.add_argument(
        "--no-data-augmentation",
        action="store_true",
        help=(
            "Disable SpecAugment and waveform augmentation regardless of the configured "
            "augmentation parameters."
        ),
    )
    parser.add_argument("--speed-perturb-prob", type=float, default=0.0)
    parser.add_argument("--speed-factors", default="0.9,1.0,1.1")
    parser.add_argument("--noise-prob", type=float, default=0.0)
    parser.add_argument("--noise-snr-db-min", type=float, default=10.0)
    parser.add_argument("--noise-snr-db-max", type=float, default=30.0)
    parser.add_argument("--reverb-prob", type=float, default=0.0)
    parser.add_argument("--reverb-decay-min", type=float, default=0.15)
    parser.add_argument("--reverb-decay-max", type=float, default=0.5)
    parser.add_argument("--reverb-delay-ms-min", type=float, default=8.0)
    parser.add_argument("--reverb-delay-ms-max", type=float, default=35.0)
    return parser.parse_args()


def _resolve_sources(
    raw_sources: list[str] | None, *, fallback: str | None = None
) -> list[str | Path]:
    sources = list(raw_sources or [])
    if not sources:
        if fallback is None:
            return []
        sources = [fallback]

    resolved_sources: list[str | Path] = []
    seen: set[str] = set()
    for source in sources:
        source_path = Path(source).expanduser()
        resolved_source: str | Path = source_path.resolve() if source_path.exists() else source
        dedupe_key = str(resolved_source)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        resolved_sources.append(resolved_source)
    return resolved_sources


def _selected_records(
    args: argparse.Namespace,
    dataset_sources: list[str | Path],
    *,
    split: str,
    val_fraction: float,
    test_fraction: float,
):
    selected = 0
    for dataset_source in dataset_sources:
        remaining_samples = None
        if args.max_samples is not None:
            remaining_samples = args.max_samples - selected
            if remaining_samples <= 0:
                break
        for record in iter_records_from_source(
            dataset_source,
            split=split,
            seed=args.seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            max_samples=remaining_samples,
            min_transcript_chars=args.min_transcript_chars,
            max_transcript_chars=args.max_transcript_chars,
            max_symbol_ratio=args.max_symbol_ratio,
            hf_token=args.hf_token,
        ):
            selected += 1
            yield record


def _resolve_validation_dataset_sources(args: argparse.Namespace) -> list[str | Path]:
    return _resolve_sources(args.validation_dataset_source)


def _resolve_float_tuple(values: str) -> tuple[float, ...]:
    parts = [part.strip() for part in values.split(",")]
    parsed = tuple(float(part) for part in parts if part)
    if not parsed:
        raise ValueError("Expected at least one comma-separated float value.")
    return parsed


def _resolve_extraction_sources(
    args: argparse.Namespace,
    dataset_sources: list[str | Path],
    validation_dataset_sources: list[str | Path],
) -> tuple[list[str | Path], str, float, float]:
    explicit_train_sources = bool(args.dataset_source)
    if args.split == "train" and explicit_train_sources:
        return dataset_sources, "train", 0.0, 0.0
    if args.split == "validation" and validation_dataset_sources:
        return validation_dataset_sources, "train", 0.0, 0.0
    return dataset_sources, args.split, args.val_fraction, args.test_fraction


def _specaugment_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "num_freq_masks": args.num_freq_masks,
        "freq_mask_param": args.freq_mask_param,
        "num_time_masks": args.num_time_masks,
        "time_mask_max_ratio": args.time_mask_max_ratio,
    }


def _waveform_augment_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "speed_perturb_prob": args.speed_perturb_prob,
        "speed_factors": _resolve_float_tuple(args.speed_factors),
        "noise_prob": args.noise_prob,
        "noise_snr_db_range": (args.noise_snr_db_min, args.noise_snr_db_max),
        "reverb_prob": args.reverb_prob,
        "reverb_decay_range": (args.reverb_decay_min, args.reverb_decay_max),
        "reverb_delay_ms_range": (args.reverb_delay_ms_min, args.reverb_delay_ms_max),
    }


def _augmentation_modules(
    args: argparse.Namespace,
) -> tuple[SpecAugment | None, WaveformAugment | None, dict[str, object] | None]:
    if args.no_data_augmentation:
        return None, None, None
    specaugment = None
    waveform_augment = None
    augmentation_config: dict[str, object] = {}
    specaugment_kwargs = _specaugment_kwargs(args)
    if args.num_freq_masks > 0 or args.num_time_masks > 0:
        specaugment = SpecAugment(**specaugment_kwargs)
        augmentation_config["specaugment"] = specaugment_kwargs
    waveform_augment_kwargs = _waveform_augment_kwargs(args)
    waveform_augment = WaveformAugment(**waveform_augment_kwargs)
    if waveform_augment.is_enabled():
        augmentation_config["waveform_augment"] = {
            "speed_perturb_prob": args.speed_perturb_prob,
            "speed_factors": list(waveform_augment_kwargs["speed_factors"]),
            "noise_prob": args.noise_prob,
            "noise_snr_db_range": list(waveform_augment_kwargs["noise_snr_db_range"]),
            "reverb_prob": args.reverb_prob,
            "reverb_decay_range": list(waveform_augment_kwargs["reverb_decay_range"]),
            "reverb_delay_ms_range": list(waveform_augment_kwargs["reverb_delay_ms_range"]),
        }
    else:
        waveform_augment = None
    return specaugment, waveform_augment, augmentation_config or None


def extract_record_features(
    record,
    split_cache_dir: str | Path,
    featurizer_kwargs: dict[str, object],
    augmentation_config: dict[str, object] | None,
    overwrite: bool,
) -> str:
    featurizer = _WORKER_FEATURIZER or AudioFeaturizer(**featurizer_kwargs)
    specaugment = _WORKER_SPECAUGMENT
    waveform_augment = _WORKER_WAVEFORM_AUGMENT
    cache_path = feature_cache_path(
        Path(split_cache_dir),
        record.utterance_id,
        featurizer,
        cache_key_extra=augmentation_config,
    )
    if cache_path is None:
        raise RuntimeError("feature cache path could not be resolved")
    if cache_path.exists() and not overwrite:
        return "cache_hit"
    waveform, sample_rate = load_audio(record.audio_path, record.audio_bytes)
    if waveform_augment is not None:
        waveform, sample_rate = waveform_augment(waveform, sample_rate)
    features = featurizer(waveform, sample_rate)
    if specaugment is not None:
        features = specaugment(features)
    features = normalize_feature_tensor(features, featurizer.n_mels)
    if features is None or not feature_tensor_is_plausible(
        record,
        features,
        expected_feature_bins=featurizer.n_mels,
    ):
        raise RuntimeError(
            "Refusing to cache invalid features for "
            f"{record.utterance_id}: shape={tuple(features.shape) if features is not None else None} "
            f"estimated_frames={int(record.estimated_frames)} "
            f"max_reasonable_frames={max_reasonable_feature_frames(record)}"
        )
    torch.save(features, cache_path)
    return "written"


def _initialize_process_worker(
    featurizer_kwargs: dict[str, object],
    specaugment_kwargs: dict[str, object] | None,
    waveform_augment_kwargs: dict[str, object] | None,
) -> None:
    global _WORKER_FEATURIZER, _WORKER_SPECAUGMENT, _WORKER_WAVEFORM_AUGMENT
    _configure_torch_threads()
    _WORKER_FEATURIZER = AudioFeaturizer(**featurizer_kwargs)
    _WORKER_SPECAUGMENT = SpecAugment(**specaugment_kwargs) if specaugment_kwargs else None
    _WORKER_WAVEFORM_AUGMENT = (
        WaveformAugment(**waveform_augment_kwargs) if waveform_augment_kwargs else None
    )


def submit_extract_record_features(
    record,
    split_cache_dir: str | Path,
    featurizer_kwargs: dict[str, object],
    augmentation_config: dict[str, object] | None,
    overwrite: bool,
) -> str:
    return extract_record_features(
        record,
        split_cache_dir,
        featurizer_kwargs,
        augmentation_config,
        overwrite,
    )


def iter_completed_futures(executor, records, num_workers: int, submit_task):
    record_iter = iter(records)
    pending = set()
    max_pending = max(num_workers * 2, 1)

    while len(pending) < max_pending:
        try:
            pending.add(executor.submit(submit_task, next(record_iter)))
        except StopIteration:
            break

    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future
        while len(pending) < max_pending:
            try:
                pending.add(executor.submit(submit_task, next(record_iter)))
            except StopIteration:
                break


def _configure_torch_threads() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            if "cannot set number of interop threads" not in str(exc):
                raise


def main() -> None:
    global _WORKER_FEATURIZER, _WORKER_SPECAUGMENT, _WORKER_WAVEFORM_AUGMENT
    args = parse_args()
    _configure_torch_threads()
    dataset_sources = _resolve_sources(args.dataset_source, fallback=args.dataset_repo)
    validation_dataset_sources = _resolve_validation_dataset_sources(args)
    selected_sources, selected_split, selected_val_fraction, selected_test_fraction = (
        _resolve_extraction_sources(args, dataset_sources, validation_dataset_sources)
    )
    specaugment, waveform_augment, augmentation_config = _augmentation_modules(args)
    featurizer = AudioFeaturizer(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        backend=args.frontend_backend,
        preemphasis=args.preemphasis,
        normalize_signal=args.normalize_signal,
        normalize_feature=args.normalize_feature,
        normalize_per_frame=args.normalize_per_frame,
    )
    _WORKER_FEATURIZER = featurizer
    _WORKER_SPECAUGMENT = specaugment
    _WORKER_WAVEFORM_AUGMENT = waveform_augment
    split_cache_dir = Path(args.feature_cache_dir) / args.split
    selection_progress = tqdm(
        desc=f"Selecting {args.split} records",
        total=args.max_samples,
        unit="utt",
    )

    def selected_records():
        try:
            for record in _selected_records(
                args,
                selected_sources,
                split=selected_split,
                val_fraction=selected_val_fraction,
                test_fraction=selected_test_fraction,
            ):
                selection_progress.update()
                yield record
        finally:
            selection_progress.close()

    records = selected_records()
    if args.prevalidate_audio:
        records = prevalidate_records(list(records), num_workers=args.prevalidate_workers)
        if not records:
            raise RuntimeError("Audio prevalidation removed every sample from the selected split.")

    counters = {"processed": 0, "written": 0, "cache_hits": 0}
    progress = tqdm(desc=f"Extracting {args.split} features", total=args.max_samples, unit="utt")
    featurizer_kwargs = {
        "sample_rate": args.sample_rate,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "backend": args.frontend_backend,
        "preemphasis": args.preemphasis,
        "normalize_signal": args.normalize_signal,
        "normalize_feature": args.normalize_feature,
        "normalize_per_frame": args.normalize_per_frame,
    }
    specaugment_kwargs = _specaugment_kwargs(args) if specaugment is not None else None
    waveform_augment_kwargs = _waveform_augment_kwargs(args) if waveform_augment is not None else None
    num_workers = max(1, args.num_workers)
    split_cache_dir_str = str(split_cache_dir)
    try:
        if num_workers == 1:
            saw_record = False
            for record in records:
                saw_record = True
                status = extract_record_features(
                    record=record,
                    split_cache_dir=split_cache_dir,
                    featurizer_kwargs=featurizer_kwargs,
                    augmentation_config=augmentation_config,
                    overwrite=args.overwrite,
                )
                counters["processed"] += 1
                counters["cache_hits"] += int(status == "cache_hit")
                counters["written"] += int(status == "written")
                progress.update()
            if not saw_record:
                raise RuntimeError("No records were selected for feature extraction.")
        else:
            executor_cls = ProcessPoolExecutor if args.parallelism == "process" else ThreadPoolExecutor
            executor_kwargs = {"max_workers": num_workers}
            if args.parallelism == "process":
                executor_kwargs["initializer"] = _initialize_process_worker
                executor_kwargs["initargs"] = (
                    featurizer_kwargs,
                    specaugment_kwargs,
                    waveform_augment_kwargs,
                )
            with executor_cls(**executor_kwargs) as executor:
                submit_task = partial(
                    submit_extract_record_features,
                    split_cache_dir=split_cache_dir_str,
                    featurizer_kwargs=featurizer_kwargs,
                    augmentation_config=augmentation_config,
                    overwrite=args.overwrite,
                )
                saw_record = False
                for future in iter_completed_futures(executor, records, num_workers, submit_task):
                    saw_record = True
                    status = future.result()
                    counters["processed"] += 1
                    counters["cache_hits"] += int(status == "cache_hit")
                    counters["written"] += int(status == "written")
                    progress.update()
                if not saw_record:
                    raise RuntimeError("No records were selected for feature extraction.")
    finally:
        progress.close()

    summary = {
        "dataset_sources": [str(source) for source in dataset_sources],
        "validation_dataset_sources": [str(source) for source in validation_dataset_sources],
        "selected_sources": [str(source) for source in selected_sources],
        "split": args.split,
        "selection_split": selected_split,
        "selection_val_fraction": selected_val_fraction,
        "selection_test_fraction": selected_test_fraction,
        "feature_cache_dir": str(split_cache_dir),
        "processed": counters["processed"],
        "written": counters["written"],
        "cache_hits": counters["cache_hits"],
        "overwrite": args.overwrite,
        "max_samples": args.max_samples,
        "parallelism": args.parallelism,
        "num_workers": num_workers,
        "augmentation_config": augmentation_config,
        "featurizer_config": featurizer.config_dict(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
