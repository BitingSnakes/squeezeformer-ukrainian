from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import trackio
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from squeezeformer_pytorch import data as _data
from squeezeformer_pytorch.asr import (
    CharacterTokenizer,
    SentencePieceTokenizer,
    SqueezeformerCTC,
    load_lm_scorer,
    load_tokenizer,
    tokenizer_from_dict,
)
from squeezeformer_pytorch.data import (
    ASRDataset,
    AudioFeaturizer,
    SpecAugment,
    WaveformAugment,
    create_dataloader,
)
from squeezeformer_pytorch.lm import NGramLanguageModel
from squeezeformer_pytorch.model import (
    SqueezeformerConfig,
    squeezeformer_variant,
    transformer_engine_available,
)
from squeezeformer_pytorch.runtime_types import DecodeStrategy, DTypeChoice, ValidationModelSource
from squeezeformer_pytorch.training import data_loading as _training_data_loading
from squeezeformer_pytorch.training import runtime as _training_runtime
from squeezeformer_pytorch.training.cli import (
    _resolve_block_pattern,
    _resolve_float_tuple,
    _resolve_scheduler_kwargs,
    _validate_startup_args,
    parse_args,
)
from squeezeformer_pytorch.training.cli import (
    _validate_device_argument as _cli_validate_device_argument,
)
from squeezeformer_pytorch.training.data_loading import (
    DiskBackedRecordStore as _DiskBackedRecordStore,
)
from squeezeformer_pytorch.training.data_loading import (
    _build_disk_backed_record_store as _data_loading_build_disk_backed_record_store,
)
from squeezeformer_pytorch.training.data_loading import (
    _build_split_audit,
    _frames_to_minutes,
    _record_store_duration_hours,
    _resolve_dataset_sources,
    _resolve_validation_dataset_sources,
)
from squeezeformer_pytorch.training.data_loading import (
    _load_records_from_dataset_roots as _data_loading_load_records_from_dataset_roots,
)
from squeezeformer_pytorch.training.evaluation import (
    _aed_cross_entropy_loss,
    _build_aed_targets,
    _evaluate_and_checkpoint,
    ctc_batch_diagnostics,
    ctc_logit_diagnostics,
    encoder_output_diagnostics,
    summarize_ctc_batch_diagnostics,
    summarize_encoder_output_diagnostics,
    summarize_ctc_logit_diagnostics,
    top_emitted_token_histogram,
)
from squeezeformer_pytorch.training.evaluation import (
    decode_batch as _evaluation_decode_batch,
)
from squeezeformer_pytorch.training.evaluation import (
    evaluate as _evaluation_evaluate,
)
from squeezeformer_pytorch.training.evaluation import (
    speaker_level_metrics as _evaluation_speaker_level_metrics,
)
from squeezeformer_pytorch.training.runtime import (
    ExponentialMovingAverage,
    FrozenAudioTeacher,
    FrozenLibertaTeacher,
    _autocast_context,
    _build_trackio_grouped_metrics,
    _compute_grad_norm,
    _configure_console_logger,
    _configure_trackio_storage,
    _format_elapsed_seconds,
    _format_memory_snapshot,
    _log_batch_autotune_snapshot,
    _peak_process_memory_bytes,
    _read_proc_status_memory_bytes,
    _resolve_aed_settings,
    _resolve_audio_teacher_settings,
    _resolve_blank_pruning_settings,
    _resolve_intermediate_ctc_settings,
    _resolve_liberta_settings,
    _resolve_model_load_dtype,
    _resolve_resume_checkpoint_path,
    _validate_resume_checkpoint_payload,
    _variant_defaults,
    build_paper_scheduler,
    resolve_device,
)
from squeezeformer_pytorch.training.runtime import (
    _validate_device_ready as _runtime_validate_device_ready,
)

DiskBackedRecordStore = _DiskBackedRecordStore
_build_disk_backed_record_store = _data_loading_build_disk_backed_record_store
_load_records_from_dataset_roots = _data_loading_load_records_from_dataset_roots
_validate_device_argument = _cli_validate_device_argument
_validate_device_ready = _runtime_validate_device_ready
decode_batch = _evaluation_decode_batch
evaluate = _evaluation_evaluate
speaker_level_metrics = _evaluation_speaker_level_metrics
download_dataset = _data.download_dataset
load_audio = _data.load_audio
te = _training_runtime.te
Format = _training_runtime.Format
DelayedScaling = _training_runtime.DelayedScaling
ExternalMuon = _training_runtime.ExternalMuon
_export_inference_checkpoint = _training_runtime._export_inference_checkpoint


def _truncate_for_log(value: str, *, limit: int = 120) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 3)]}..."


def _build_trackio_run_name(
    *,
    trackio_project: str,
    output_dir: Path,
    start_epoch: int,
    global_step: int,
    process_start_timestamp: float,
) -> str:
    base_name = trackio_project.strip() or output_dir.name or "training"
    normalized_base_name = re.sub(r"[^A-Za-z0-9._-]+", "-", base_name).strip("-")
    if not normalized_base_name:
        normalized_base_name = "training"
    timestamp = datetime.fromtimestamp(process_start_timestamp, tz=timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    run_name = f"{normalized_base_name}_{timestamp}"
    if global_step > 0 or start_epoch > 1:
        run_name = f"{run_name}_resume-e{start_epoch:04d}-s{global_step:08d}"
    return run_name


def _looks_like_numeric_cli_value(token: str) -> bool:
    return bool(re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token))


def _build_trackio_cli_arguments_table(argv: list[str]) -> trackio.Table | None:
    rows: list[dict[str, object]] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("-") or token == "-":
            index += 1
            continue

        value: str | bool = True
        if token.startswith("--") and "=" in token:
            argument, value = token.split("=", 1)
        else:
            argument = token
            next_index = index + 1
            if next_index < len(argv):
                next_token = argv[next_index]
                if not next_token.startswith("-") or _looks_like_numeric_cli_value(next_token):
                    value = next_token
                    index = next_index

        rows.append(
            {
                "position": len(rows) + 1,
                "argument": argument,
                "value": value,
            }
        )
        index += 1

    if not rows:
        return None
    return trackio.Table(data=rows)


def _launch_trackio_ui(
    *,
    trackio_dir: Path,
    logger: logging.Logger,
    project: str | None = None,
) -> tuple[object, str, str | None, str]:
    original_trackio_dir = os.environ.get("TRACKIO_DIR")
    original_gradio_share = os.environ.get("GRADIO_SHARE")
    try:
        os.environ["TRACKIO_DIR"] = str(trackio_dir)
        os.environ["GRADIO_SHARE"] = "True"
        app, url, share_url, full_url = trackio.show(
            project=project,
            open_browser=False,
            block_thread=False,
        )
    finally:
        if original_trackio_dir is None:
            os.environ.pop("TRACKIO_DIR", None)
        else:
            os.environ["TRACKIO_DIR"] = original_trackio_dir
        if original_gradio_share is None:
            os.environ.pop("GRADIO_SHARE", None)
        else:
            os.environ["GRADIO_SHARE"] = original_gradio_share
    logger.info(
        "trackio ui launched trackio_dir=%s local_url=%s share_url=%s full_url=%s",
        trackio_dir,
        url,
        share_url,
        full_url,
    )
    return app, url, share_url, full_url


def _validate_resume_tokenizer_configuration(
    *,
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    requested_tokenizer_type: str,
    tokenizer_path: str | None,
) -> None:
    checkpoint_tokenizer = checkpoint.get("tokenizer")
    if not isinstance(checkpoint_tokenizer, dict):
        raise RuntimeError(
            f"Resume checkpoint '{checkpoint_path}' is missing tokenizer metadata or it is malformed."
        )
    checkpoint_tokenizer_type = str(checkpoint_tokenizer.get("type", "character"))
    if checkpoint_tokenizer_type != requested_tokenizer_type:
        raise RuntimeError(
            f"Resume checkpoint '{checkpoint_path}' uses tokenizer type "
            f"'{checkpoint_tokenizer_type}', but the current run requested "
            f"'--tokenizer {requested_tokenizer_type}'. Start a fresh run or resume from a "
            "checkpoint created with the same tokenizer type."
        )
    if tokenizer_path is None:
        return
    requested_tokenizer = load_tokenizer(tokenizer_path)
    if requested_tokenizer.to_dict() != checkpoint_tokenizer:
        raise RuntimeError(
            f"Resume checkpoint '{checkpoint_path}' tokenizer metadata does not match the "
            f"tokenizer loaded from '{tokenizer_path}'. Use the tokenizer artifact that created "
            "this checkpoint or start a fresh run."
        )


def _distributed_mean(value: float, *, device: torch.device, distributed: bool) -> float:
    if not distributed or not dist.is_initialized():
        return value
    reduced = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    return float(reduced.item())


def _distributed_sum_float(value: float, *, device: torch.device, distributed: bool) -> float:
    if not distributed or not dist.is_initialized():
        return value
    reduced = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return float(reduced.item())


def _distributed_weighted_mean(
    value: float,
    *,
    weight: float,
    device: torch.device,
    distributed: bool,
) -> float:
    if weight <= 0.0:
        return 0.0
    if not distributed or not dist.is_initialized():
        return value
    reduced = torch.tensor(
        [value * weight, weight],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    denominator = max(float(reduced[1].item()), 1e-12)
    return float(reduced[0].item() / denominator)


def _synchronize_best_val_wer(
    best_val_wer: float, *, device: torch.device, distributed: bool
) -> float:
    if not distributed or not dist.is_initialized():
        return best_val_wer
    payload = torch.tensor(best_val_wer, device=device, dtype=torch.float64)
    dist.broadcast(payload, src=0)
    return float(payload.item())


def _distributed_min_int(value: int, *, device: torch.device, distributed: bool) -> int:
    if not distributed or not dist.is_initialized():
        return value
    reduced = torch.tensor(value, device=device, dtype=torch.int64)
    dist.all_reduce(reduced, op=dist.ReduceOp.MIN)
    return int(reduced.item())


def _distributed_max_int(value: int, *, device: torch.device, distributed: bool) -> int:
    if not distributed or not dist.is_initialized():
        return value
    reduced = torch.tensor(value, device=device, dtype=torch.int64)
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return int(reduced.item())


def _set_dataloader_epoch(dataloader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


def _classifier_head_row_diagnostics(model: nn.Module) -> dict[str, float]:
    classifier = getattr(model, "classifier", None)
    weight = getattr(classifier, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] < 2:
        return {
            "blank_bias": 0.0,
            "blank_weight_norm": 0.0,
            "nonblank_weight_norm_mean": 0.0,
            "blank_weight_grad_norm": 0.0,
            "nonblank_weight_grad_norm_mean": 0.0,
            "blank_bias_grad": 0.0,
            "nonblank_bias_grad_mean": 0.0,
        }

    weight_float = weight.detach().float()
    blank_weight = weight_float[0]
    nonblank_weight = weight_float[1:]
    weight_grad = weight.grad
    if isinstance(weight_grad, torch.Tensor):
        weight_grad_float = weight_grad.detach().float()
        blank_weight_grad_norm = float(weight_grad_float[0].norm().item())
        nonblank_weight_grad_norm_mean = float(
            weight_grad_float[1:].norm(dim=1).mean().item()
        )
    else:
        blank_weight_grad_norm = 0.0
        nonblank_weight_grad_norm_mean = 0.0

    bias = getattr(classifier, "bias", None)
    blank_bias = 0.0
    blank_bias_grad = 0.0
    nonblank_bias_grad_mean = 0.0
    if isinstance(bias, torch.Tensor) and bias.ndim == 1 and bias.shape[0] == weight.shape[0]:
        bias_float = bias.detach().float()
        blank_bias = float(bias_float[0].item())
        bias_grad = bias.grad
        if isinstance(bias_grad, torch.Tensor):
            bias_grad_float = bias_grad.detach().float()
            blank_bias_grad = float(bias_grad_float[0].item())
            nonblank_bias_grad_mean = float(bias_grad_float[1:].mean().item())

    return {
        "blank_bias": blank_bias,
        "blank_weight_norm": float(blank_weight.norm().item()),
        "nonblank_weight_norm_mean": float(nonblank_weight.norm(dim=1).mean().item()),
        "blank_weight_grad_norm": blank_weight_grad_norm,
        "nonblank_weight_grad_norm_mean": nonblank_weight_grad_norm_mean,
        "blank_bias_grad": blank_bias_grad,
        "nonblank_bias_grad_mean": nonblank_bias_grad_mean,
    }


def _divide_gradients_in_place(parameters, denominator: float) -> None:
    if denominator <= 0.0:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(float(denominator))


def _should_warn_on_blank_starvation(
    *,
    global_step: int,
    avg_blank_probability: float,
    argmax_blank_fraction: float,
    avg_top_nonblank_probability: float,
) -> bool:
    if global_step <= 0 or global_step > 500:
        return False
    if argmax_blank_fraction > 0.001:
        return False
    if avg_top_nonblank_probability <= 0.0:
        return False
    return avg_blank_probability <= (0.5 * avg_top_nonblank_probability)


def _decode_train_preview_hypotheses(
    *,
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer,
    beam_size: int,
    lm_scorer,
    lm_weight: float,
    beam_length_bonus: float,
) -> tuple[str, str]:
    preview_log_probs = log_probs[:1]
    preview_output_lengths = output_lengths[:1]
    greedy_hypothesis = decode_batch(
        preview_log_probs,
        preview_output_lengths,
        tokenizer=tokenizer,
        strategy=DecodeStrategy.GREEDY,
    )[0]
    beam_hypothesis = decode_batch(
        preview_log_probs,
        preview_output_lengths,
        tokenizer=tokenizer,
        strategy=DecodeStrategy.BEAM,
        beam_size=beam_size,
        lm_scorer=lm_scorer,
        lm_weight=lm_weight,
        beam_length_bonus=beam_length_bonus,
    )[0]
    return greedy_hypothesis, beam_hypothesis


def _distributed_barrier() -> None:
    if not dist.is_initialized():
        return
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
        return
    dist.barrier()


def _maybe_downshift_adaptive_batch_budget(
    args,
    *,
    device: torch.device,
    distributed: bool,
    liberta_distill_enabled: bool,
    logger,
) -> None:
    if (
        not distributed
        or not liberta_distill_enabled
        or device.type != "cuda"
        or args.adaptive_batch_unit is None
        or args.adaptive_batch_budget is None
    ):
        return
    original_budget = int(args.adaptive_batch_budget)
    adjusted_budget = original_budget
    reason = "distributed LiBERTa distillation safety margin"
    if args.adaptive_batch_unit == "frames":
        total_memory_gib = float(torch.cuda.get_device_properties(device).total_memory) / float(
            1024**3
        )
        free_memory_gib = total_memory_gib
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            free_memory_gib = min(free_memory_gib, float(free_bytes) / float(1024**3))
        except RuntimeError:
            pass
        adjusted_budget = max(1000, int(free_memory_gib * 2500.0))
        adjusted_budget = max(1000, (adjusted_budget // 1000) * 1000)
        reason = (
            f"{reason} total_gpu_memory={total_memory_gib:.2f}GiB "
            f"free_gpu_memory={free_memory_gib:.2f}GiB"
        )
    else:
        adjusted_budget = max(1, int(original_budget * 0.75))
    adjusted_budget = _distributed_min_int(
        adjusted_budget,
        device=device,
        distributed=distributed,
    )
    if adjusted_budget >= original_budget:
        return
    args.adaptive_batch_budget = adjusted_budget
    logger.warning(
        "downshifting adaptive batch budget from %s to %s for %s",
        original_budget,
        adjusted_budget,
        reason,
    )


def _resolve_dataset_roots(args) -> list[Path]:
    sources = list(args.dataset_source or [])
    if not sources:
        sources = [args.dataset_repo]

    dataset_roots: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        dataset_root = download_dataset(
            repo_id=source,
            token=args.hf_token,
            cache_dir=args.cache_dir,
        ).resolve()
        if dataset_root in seen:
            continue
        seen.add(dataset_root)
        dataset_roots.append(dataset_root)
    return dataset_roots


def _ensure_opus_decode_support(records, *, split: str) -> None:
    original_load_audio = _training_data_loading.load_audio
    _training_data_loading.load_audio = load_audio
    try:
        return _training_data_loading._ensure_opus_decode_support(records, split=split)
    finally:
        _training_data_loading.load_audio = original_load_audio


def _load_train_val_records(
    args,
    train_dataset_sources,
    *,
    validation_dataset_sources=None,
    lowercase_transcripts: bool,
    output_dir: Path,
    distributed: bool = False,
    is_main_process: bool = True,
):
    original_load_records = _training_data_loading._load_records_from_dataset_roots
    original_build_store = _training_data_loading._build_disk_backed_record_store
    _training_data_loading._load_records_from_dataset_roots = _load_records_from_dataset_roots
    _training_data_loading._build_disk_backed_record_store = _build_disk_backed_record_store
    try:
        return _training_data_loading._load_train_val_records(
            args,
            train_dataset_sources,
            validation_dataset_sources=validation_dataset_sources,
            lowercase_transcripts=lowercase_transcripts,
            output_dir=output_dir,
            distributed=distributed,
            is_main_process=is_main_process,
        )
    finally:
        _training_data_loading._load_records_from_dataset_roots = original_load_records
        _training_data_loading._build_disk_backed_record_store = original_build_store


def _validate_fp8_runtime(device: torch.device, encoder_config: SqueezeformerConfig) -> None:
    original_te = _training_runtime.te
    original_transformer_engine_available = _training_runtime.transformer_engine_available
    _training_runtime.te = te
    _training_runtime.transformer_engine_available = transformer_engine_available
    try:
        return _training_runtime._validate_fp8_runtime(device, encoder_config)
    finally:
        _training_runtime.te = original_te
        _training_runtime.transformer_engine_available = original_transformer_engine_available


def _build_fp8_recipe(args):
    original_format = _training_runtime.Format
    original_delayed_scaling = _training_runtime.DelayedScaling
    _training_runtime.Format = Format
    _training_runtime.DelayedScaling = DelayedScaling
    try:
        return _training_runtime._build_fp8_recipe(args)
    finally:
        _training_runtime.Format = original_format
        _training_runtime.DelayedScaling = original_delayed_scaling


def build_optimizer(*args, **kwargs):
    original_external_muon = _training_runtime.ExternalMuon
    _training_runtime.ExternalMuon = ExternalMuon
    try:
        return _training_runtime.build_optimizer(*args, **kwargs)
    finally:
        _training_runtime.ExternalMuon = original_external_muon


def _average_topk_checkpoints(output_dir: Path) -> Path | None:
    original_export = _training_runtime._export_inference_checkpoint
    _training_runtime._export_inference_checkpoint = _export_inference_checkpoint
    try:
        return _training_runtime._average_topk_checkpoints(output_dir)
    finally:
        _training_runtime._export_inference_checkpoint = original_export


def _update_top_checkpoints(
    output_dir: Path,
    checkpoint: dict[str, object],
    epoch: int,
    val_wer: float,
    keep_top_k: int,
    global_step: int | None = None,
) -> None:
    if global_step is not None:
        return _training_runtime._update_top_checkpoints(
            output_dir=output_dir,
            checkpoint=checkpoint,
            epoch=epoch,
            global_step=global_step,
            val_wer=val_wer,
            keep_top_k=keep_top_k,
        )

    topk_dir = output_dir / "checkpoints_topk"
    topk_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = topk_dir / "metadata.json"
    compatibility_signature, metadata = _training_runtime._load_topk_metadata(metadata_path)
    current_signature = _training_runtime._checkpoint_compatibility_signature(
        checkpoint["model_state_dict"]
    )
    logger = logging.getLogger("train")

    if compatibility_signature is not None and compatibility_signature != current_signature:
        for item in metadata:
            checkpoint_path = topk_dir / str(item["path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        metadata = []

    compatible_metadata: list[dict[str, object]] = []
    removed_incompatible = 0
    for item in metadata:
        checkpoint_path = topk_dir / str(item["path"])
        if not checkpoint_path.exists():
            continue
        saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_signature = _training_runtime._checkpoint_compatibility_signature(
            saved_checkpoint["model_state_dict"]
        )
        if saved_signature != current_signature:
            checkpoint_path.unlink()
            removed_incompatible += 1
            continue
        compatible_metadata.append(item)
    if removed_incompatible:
        logger.info(
            "Removed %s incompatible top-k checkpoint artifact(s) from %s.",
            removed_incompatible,
            topk_dir,
            extra={"rank": 0},
        )
    metadata = compatible_metadata

    filename = f"checkpoint_epoch={epoch:04d}_valwer={val_wer:.6f}.pt"
    checkpoint_path = topk_dir / filename
    _training_runtime.save_checkpoint(checkpoint, checkpoint_path)

    metadata.append(
        {
            "epoch": epoch,
            "val_wer": val_wer,
            "path": str(checkpoint_path.name),
        }
    )
    metadata.sort(key=lambda item: (float(item["val_wer"]), int(item["epoch"])))

    removed = metadata[keep_top_k:]
    metadata = metadata[:keep_top_k]
    for item in removed:
        stale_path = topk_dir / str(item["path"])
        if stale_path.exists():
            stale_path.unlink()

    _training_runtime._write_topk_metadata(metadata_path, current_signature, metadata)


def main() -> None:
    args = parse_args()
    process_start_monotonic = time.perf_counter()
    process_start_timestamp = time.time()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _validate_startup_args(args, world_size=world_size)
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested_device = resolve_device(args.device)
    if distributed and requested_device.type == "cuda":
        if requested_device.index not in {None, local_rank}:
            raise ValueError(
                f"--device {args.device} conflicts with LOCAL_RANK={local_rank}. "
                "Use --device cuda or the matching cuda:<local_rank>."
            )
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = requested_device
    sync_device = device
    is_main_process = rank == 0
    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / (
        f"training_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = _configure_console_logger(
        rank=rank,
        is_main_process=is_main_process,
        log_path=training_log_path,
    )
    trackio_dir = _configure_trackio_storage(output_dir)
    if is_main_process and args.run_trackio_ui:
        _launch_trackio_ui(
            trackio_dir=trackio_dir,
            logger=logger,
            project=args.trackio_project,
        )
    resume_path = _resolve_resume_checkpoint_path(args, output_dir=output_dir, logger=logger)
    logger.info(
        "starting training variant=%s device=%s distributed=%s world_size=%s output_dir=%s",
        args.variant,
        requested_device,
        distributed,
        world_size,
        output_dir,
    )
    variant_defaults = _variant_defaults(args.variant)
    stage_start_time = time.perf_counter()
    logger.info("loading LM scorer source=%s", args.lm_scorer or "disabled")
    lm_scorer = load_lm_scorer(args.lm_scorer)
    logger.info(
        "LM scorer ready source=%s elapsed=%s",
        args.lm_scorer or "disabled",
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
    )

    stage_start_time = time.perf_counter()
    logger.info("resolving dataset sources")
    train_dataset_sources = _resolve_dataset_sources(args)
    validation_dataset_sources = _resolve_validation_dataset_sources(args)
    lowercase_transcripts = args.tokenizer != "sentencepiece"
    logger.info(
        "dataset sources resolved train_count=%s validation_count=%s elapsed=%s",
        len(train_dataset_sources),
        len(validation_dataset_sources),
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
    )
    stage_start_time = time.perf_counter()
    logger.info(
        "loading dataset records train_sources=%s validation_sources=%s validation_mode=%s record_cache=%s prevalidate_audio=%s",
        [str(dataset_source) for dataset_source in train_dataset_sources],
        [str(dataset_source) for dataset_source in validation_dataset_sources]
        if validation_dataset_sources
        else [str(dataset_source) for dataset_source in train_dataset_sources],
        "external" if validation_dataset_sources else "split",
        args.record_cache,
        args.prevalidate_audio,
    )
    train_records, val_records = _load_train_val_records(
        args,
        train_dataset_sources,
        validation_dataset_sources=validation_dataset_sources,
        lowercase_transcripts=lowercase_transcripts,
        output_dir=output_dir,
        distributed=distributed,
        is_main_process=is_main_process,
    )
    _ensure_opus_decode_support(train_records, split="train")
    _ensure_opus_decode_support(val_records, split="validation")
    split_audit = _build_split_audit(
        {"train": train_records, "validation": val_records},
        hop_length=args.hop_length,
    )
    if is_main_process:
        (output_dir / "split_audit.json").write_text(
            json.dumps(split_audit, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "loaded datasets train_sources=%s validation_sources=%s train_samples=%s val_samples=%s speaker_balance_ratio=%.3f elapsed=%s",
            [str(dataset_source) for dataset_source in train_dataset_sources],
            [str(dataset_source) for dataset_source in validation_dataset_sources]
            if validation_dataset_sources
            else [str(dataset_source) for dataset_source in train_dataset_sources],
            len(train_records),
            len(val_records),
            float(split_audit["speaker_balance_ratio"]),
            _format_elapsed_seconds(time.perf_counter() - stage_start_time),
        )

    stage_start_time = time.perf_counter()
    checkpoint = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path is not None
        else None
    )
    if resume_path is not None:
        _validate_resume_checkpoint_payload(checkpoint, checkpoint_path=resume_path)
        _validate_resume_tokenizer_configuration(
            checkpoint=checkpoint,
            checkpoint_path=resume_path,
            requested_tokenizer_type=args.tokenizer,
            tokenizer_path=args.tokenizer_path,
        )
        logger.info(
            "resume checkpoint loaded path=%s elapsed=%s",
            resume_path,
            _format_elapsed_seconds(time.perf_counter() - stage_start_time),
        )
    (
        aed_decoder_enabled,
        aed_decoder_layers,
        aed_decoder_heads,
        aed_decoder_dropout,
        aed_loss_weight,
    ) = _resolve_aed_settings(args, checkpoint)
    (
        liberta_distill_enabled,
        liberta_model_name,
        liberta_model_path,
        liberta_distill_weight,
        liberta_max_length,
    ) = _resolve_liberta_settings(
        args,
        checkpoint,
        aed_enabled=aed_decoder_enabled,
    )
    (
        audio_teacher_enabled,
        audio_teacher_model_name,
        audio_teacher_model_path,
        audio_teacher_weight,
        audio_teacher_objective,
        audio_teacher_target,
        audio_teacher_layer,
        audio_teacher_sample_rate,
        audio_teacher_max_seconds,
    ) = _resolve_audio_teacher_settings(
        args,
        checkpoint,
    )
    stage_start_time = time.perf_counter()
    logger.info(
        "preparing tokenizer mode=%s resume=%s tokenizer_path=%s",
        args.tokenizer,
        checkpoint is not None,
        args.tokenizer_path or "auto",
    )
    tokenizer_path = output_dir / "tokenizer.json"
    if checkpoint is not None:
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    elif args.tokenizer_path is not None:
        tokenizer = load_tokenizer(args.tokenizer_path)
    elif args.tokenizer == "sentencepiece":
        sentencepiece_model_path = output_dir / "tokenizer.model"
        if not distributed or is_main_process:
            tokenizer = SentencePieceTokenizer.train(
                (record.transcript for record in train_records),
                model_prefix=output_dir / "tokenizer",
                vocab_size=args.spm_vocab_size,
                model_type=args.spm_model_type,
            )
            tokenizer.save(sentencepiece_model_path)
            tokenizer.save(tokenizer_path)
        if distributed:
            _distributed_barrier()
        if distributed and not is_main_process:
            tokenizer = load_tokenizer(sentencepiece_model_path)
    else:
        if not distributed or is_main_process:
            tokenizer = CharacterTokenizer.build(record.transcript for record in train_records)
            tokenizer.save(tokenizer_path)
        if distributed:
            _distributed_barrier()
        if distributed and not is_main_process:
            tokenizer = load_tokenizer(tokenizer_path)
    if (checkpoint is not None or args.tokenizer_path is not None) and (
        not distributed or is_main_process
    ):
        tokenizer.save(tokenizer_path)
    if distributed and (checkpoint is not None or args.tokenizer_path is not None):
        _distributed_barrier()
    logger.info(
        "tokenizer ready vocab_size=%s elapsed=%s",
        tokenizer.vocab_size,
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
    )
    if args.fit_shallow_fusion_lm:
        stage_start_time = time.perf_counter()
        shallow_fusion_lm_path = output_dir / "shallow_fusion_lm.json"
        if not distributed or is_main_process:
            logger.info(
                "training shallow-fusion LM order=%s alpha=%.3f",
                args.shallow_fusion_lm_order,
                args.shallow_fusion_lm_alpha,
            )
            shallow_fusion_lm = NGramLanguageModel.train(
                (record.transcript for record in train_records),
                order=args.shallow_fusion_lm_order,
                alpha=args.shallow_fusion_lm_alpha,
            )
            shallow_fusion_lm.save(shallow_fusion_lm_path)
        if distributed:
            _distributed_barrier()
        if distributed and not is_main_process:
            shallow_fusion_lm = NGramLanguageModel.load(shallow_fusion_lm_path)
        if lm_scorer is None:
            lm_scorer = shallow_fusion_lm.score_extension
        logger.info(
            "shallow-fusion LM ready path=%s elapsed=%s",
            shallow_fusion_lm_path,
            _format_elapsed_seconds(time.perf_counter() - stage_start_time),
        )
    featurizer = AudioFeaturizer(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        backend=args.frontend_backend,
        preemphasis=args.preemphasis,
        normalize_signal=args.normalize_signal,
        normalize_feature=args.normalize_feature,
        normalize_per_frame=args.normalize_per_frame,
    )
    specaugment = None
    waveform_augment = None
    if not args.no_data_augmentation:
        specaugment = SpecAugment(
            num_freq_masks=args.num_freq_masks,
            freq_mask_param=args.freq_mask_param,
            num_time_masks=args.num_time_masks or variant_defaults.num_time_masks,
            time_mask_max_ratio=args.time_mask_max_ratio,
        )
        waveform_augment = WaveformAugment(
            speed_perturb_prob=args.speed_perturb_prob,
            speed_factors=_resolve_float_tuple(args.speed_factors),
            noise_prob=args.noise_prob,
            noise_snr_db_range=(args.noise_snr_db_min, args.noise_snr_db_max),
            reverb_prob=args.reverb_prob,
            reverb_decay_range=(args.reverb_decay_min, args.reverb_decay_max),
            reverb_delay_ms_range=(args.reverb_delay_ms_min, args.reverb_delay_ms_max),
        )
    train_feature_cache_dir = (
        Path(args.feature_cache_dir) / "train" if args.feature_cache_dir is not None else None
    )
    val_feature_cache_dir = (
        Path(args.feature_cache_dir) / "validation" if args.feature_cache_dir is not None else None
    )
    _maybe_downshift_adaptive_batch_budget(
        args,
        device=device,
        distributed=distributed,
        liberta_distill_enabled=liberta_distill_enabled,
        logger=logger,
    )
    train_dataset = ASRDataset(
        train_records,
        tokenizer=tokenizer,
        featurizer=featurizer,
        specaugment=specaugment,
        waveform_augment=waveform_augment,
        feature_cache_dir=train_feature_cache_dir,
        return_waveforms=audio_teacher_enabled,
    )
    val_dataset = ASRDataset(
        val_records,
        tokenizer=tokenizer,
        featurizer=featurizer,
        feature_cache_dir=val_feature_cache_dir,
        return_waveforms=audio_teacher_enabled,
    )
    stage_start_time = time.perf_counter()
    logger.info(
        "building dataloaders train_samples=%s val_samples=%s distributed=%s world_size=%s train_hours=%.2f val_hours=%.2f num_workers=%s metadata_workers=%s persistent_workers=%s prefetch_factor=%s",
        len(train_records),
        len(val_records),
        distributed,
        world_size,
        _record_store_duration_hours(train_records, hop_length=args.hop_length),
        _record_store_duration_hours(val_records, hop_length=args.hop_length),
        args.num_workers,
        args.metadata_workers,
        args.persistent_workers,
        args.prefetch_factor,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        bucket_by_length=args.bucket_by_length,
        max_batch_duration_sec=args.max_batch_duration_sec,
        max_batch_frames=args.max_batch_frames,
        adaptive_batch_unit=args.adaptive_batch_unit,
        adaptive_batch_budget=args.adaptive_batch_budget,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        metadata_workers=args.metadata_workers,
        longest_batches_first=args.longest_batches_first,
        multiprocessing_context=args.dataloader_mp_context,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        pad_distributed_batches=distributed,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        bucket_by_length=args.bucket_by_length,
        max_batch_duration_sec=args.max_batch_duration_sec,
        max_batch_frames=args.max_batch_frames,
        adaptive_batch_unit=args.adaptive_batch_unit,
        adaptive_batch_budget=args.adaptive_batch_budget,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        metadata_workers=args.metadata_workers,
        longest_batches_first=False,
        multiprocessing_context=args.dataloader_mp_context,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        pad_distributed_batches=False,
    )
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    logger.info(
        "dataloaders ready train_batches=%s val_batches=%s elapsed=%s",
        train_batches,
        val_batches,
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
    )

    encoder_config = (
        SqueezeformerConfig(**checkpoint["encoder_config"])
        if checkpoint is not None
        else squeezeformer_variant(args.variant)
    )
    if checkpoint is None:
        attention_backend = (
            "relative" if args.disable_flash_attention else args.attention_backend
        )
        encoder_config = replace(
            deepcopy(encoder_config),
            block_pattern=_resolve_block_pattern(args.block_pattern),
            activation_checkpointing=args.activation_checkpointing,
            attention_backend=attention_backend,
        )
    if args.disable_flash_attn2_kernels:
        encoder_config = replace(
            deepcopy(encoder_config),
            flash_attn2_enabled=False,
        )
    intermediate_ctc_layers, intermediate_ctc_weight = _resolve_intermediate_ctc_settings(
        args,
        encoder_config,
        checkpoint,
    )
    blank_prune_layer, blank_prune_threshold, blank_prune_min_keep_frames = (
        _resolve_blank_pruning_settings(args, encoder_config, checkpoint)
    )
    args.intermediate_ctc_layers = list(intermediate_ctc_layers)
    args.intermediate_ctc_layer = (
        intermediate_ctc_layers[0] if len(intermediate_ctc_layers) == 1 else None
    )
    args.intermediate_ctc = bool(intermediate_ctc_layers) and intermediate_ctc_weight > 0.0
    args.intermediate_ctc_weight = intermediate_ctc_weight
    args.aed_decoder = aed_decoder_enabled
    args.aed_decoder_layers = aed_decoder_layers
    args.aed_decoder_heads = aed_decoder_heads
    args.aed_decoder_dropout = aed_decoder_dropout
    args.aed_loss_weight = aed_loss_weight
    args.liberta_distill = liberta_distill_enabled
    args.liberta_model_name = liberta_model_name
    args.liberta_model_path = liberta_model_path
    args.liberta_distill_weight = liberta_distill_weight
    args.liberta_max_length = liberta_max_length
    args.audio_teacher = audio_teacher_enabled
    args.audio_teacher_model_name = audio_teacher_model_name
    args.audio_teacher_model_path = audio_teacher_model_path
    args.audio_teacher_weight = audio_teacher_weight
    args.audio_teacher_objective = audio_teacher_objective
    args.audio_teacher_target = audio_teacher_target
    args.audio_teacher_layer = audio_teacher_layer
    args.audio_teacher_sample_rate = audio_teacher_sample_rate
    args.audio_teacher_max_seconds = audio_teacher_max_seconds
    args.blank_prune = blank_prune_layer is not None and blank_prune_threshold > 0.0
    args.blank_prune_layer = blank_prune_layer
    args.blank_prune_threshold = blank_prune_threshold
    args.blank_prune_min_keep_frames = blank_prune_min_keep_frames
    fp8_recipe = _build_fp8_recipe(args)
    if args.dtype == DTypeChoice.FP8:
        _validate_fp8_runtime(device, encoder_config)
    requested_audio_teacher_device = resolve_device(args.audio_teacher_device)
    if distributed and requested_audio_teacher_device.type == "cuda":
        if requested_audio_teacher_device.index not in {None, local_rank}:
            logger.warning(
                "--audio-teacher-device %s conflicts with LOCAL_RANK=%s; using cuda:%s for this rank.",
                args.audio_teacher_device,
                local_rank,
                local_rank,
            )
        audio_teacher_device = torch.device(f"cuda:{local_rank}")
    else:
        audio_teacher_device = requested_audio_teacher_device
    audio_teacher = (
        FrozenAudioTeacher(
            str(Path(audio_teacher_model_path).expanduser().resolve())
            if audio_teacher_model_path is not None
            else audio_teacher_model_name,
            device=audio_teacher_device,
            dtype=_resolve_model_load_dtype(args.dtype),
            sample_rate=audio_teacher_sample_rate,
            layer=audio_teacher_layer,
            max_seconds=audio_teacher_max_seconds,
        )
        if audio_teacher_enabled
        else None
    )
    stage_start_time = time.perf_counter()
    logger.info(
        "building model variant=%s dtype=%s compile=%s intermediate_ctc_layers=%s identical_initial_ctc_heads=%s aed=%s liberta=%s audio_teacher=%s blank_prune_layer=%s",
        args.variant,
        args.dtype,
        args.compile,
        list(intermediate_ctc_layers),
        args.identical_initial_ctc_heads,
        aed_decoder_enabled,
        liberta_distill_enabled,
        audio_teacher_enabled,
        blank_prune_layer,
    )
    model = SqueezeformerCTC(
        encoder_config=encoder_config,
        vocab_size=tokenizer.vocab_size,
        intermediate_ctc_layers=intermediate_ctc_layers,
        blank_prune_layer=blank_prune_layer,
        blank_prune_threshold=blank_prune_threshold,
        blank_prune_min_keep_frames=blank_prune_min_keep_frames,
        aed_decoder_enabled=aed_decoder_enabled,
        aed_decoder_layers=aed_decoder_layers,
        aed_decoder_heads=aed_decoder_heads,
        aed_decoder_dropout=aed_decoder_dropout,
        liberta_distill_enabled=liberta_distill_enabled,
        audio_teacher_enabled=audio_teacher is not None,
        audio_teacher_hidden_size=(
            audio_teacher.hidden_size if audio_teacher is not None else encoder_config.d_model
        ),
        audio_teacher_target=audio_teacher_target,
        initial_ctc_blank_bias=args.initial_ctc_blank_bias,
        identical_initial_ctc_heads=args.identical_initial_ctc_heads,
        blank_logit_offset=args.blank_logit_offset,
        blank_logit_regularization_weight=args.blank_logit_regularization_weight,
        use_transformer_engine=args.dtype == DTypeChoice.FP8,
    )
    model.to(device)
    if checkpoint is not None:
        model.load_state_dict(
            checkpoint.get("resume_model_state_dict", checkpoint["model_state_dict"])
        )
    ddp_find_unused_parameters = (
        blank_prune_layer is not None
        and blank_prune_threshold > 0.0
        and blank_prune_layer not in intermediate_ctc_layers
    )
    if distributed:
        forward_model: nn.Module = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=ddp_find_unused_parameters,
        )
    else:
        forward_model = torch.compile(model) if args.compile else model
    use_grad_scaler = device.type == "cuda" and args.dtype == DTypeChoice.FLOAT16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    peak_lr = args.learning_rate if args.learning_rate is not None else variant_defaults.peak_lr
    muon_lr = args.muon_learning_rate if args.muon_learning_rate is not None else peak_lr
    adamw_lr = args.adamw_learning_rate if args.adamw_learning_rate is not None else peak_lr
    muon_weight_decay = (
        args.muon_weight_decay if args.muon_weight_decay is not None else args.weight_decay
    )
    adamw_weight_decay = (
        args.adamw_weight_decay if args.adamw_weight_decay is not None else args.weight_decay
    )
    optimizer_steps_per_epoch = max(
        1,
        (train_batches + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps,
    )
    optimizers, optimizer_names = build_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        muon_weight_decay=muon_weight_decay,
        adamw_weight_decay=adamw_weight_decay,
    )
    schedulers = []
    for optimizer, optimizer_group_name in zip(optimizers, optimizer_names, strict=True):
        schedulers.append(
            build_paper_scheduler(
                optimizer,
                steps_per_epoch=optimizer_steps_per_epoch,
                **_resolve_scheduler_kwargs(args, optimizer_group_name),
            )
        )
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    requested_liberta_device = resolve_device(args.liberta_device)
    if distributed and requested_liberta_device.type == "cuda":
        if requested_liberta_device.index not in {None, local_rank}:
            logger.warning(
                "--liberta-device %s conflicts with LOCAL_RANK=%s; using cuda:%s for this rank.",
                args.liberta_device,
                local_rank,
                local_rank,
            )
        liberta_device = torch.device(f"cuda:{local_rank}")
    else:
        liberta_device = requested_liberta_device
    liberta_teacher = (
        FrozenLibertaTeacher(
            str(Path(liberta_model_path).expanduser().resolve())
            if liberta_model_path is not None
            else liberta_model_name,
            device=liberta_device,
            dtype=_resolve_model_load_dtype(args.dtype),
            max_length=liberta_max_length,
        )
        if liberta_distill_enabled
        else None
    )
    logger.info(
        "model and auxiliaries ready params=%s liberta=%s audio_teacher=%s elapsed=%s",
        sum(parameter.numel() for parameter in model.parameters()),
        liberta_teacher is not None,
        audio_teacher is not None,
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
    )
    ema = (
        ExponentialMovingAverage(
            model,
            decay=args.ema_decay,
            warmup_steps=args.ema_warmup_steps,
        )
        if args.ema_decay > 0
        else None
    )
    if args.validation_model_source == ValidationModelSource.EMA and ema is None:
        raise ValueError(
            "--validation-model-source ema requires EMA to be enabled; set --ema-decay > 0 "
            "or use --validation-model-source raw."
        )
    start_epoch = 1
    global_step = 0
    best_val_wer = float("inf")
    if checkpoint is not None:
        optimizer_states = checkpoint.get("optimizer_state_dicts", [])
        scheduler_states = checkpoint.get("scheduler_state_dicts", [])
        if len(optimizer_states) != len(optimizers) or len(scheduler_states) != len(schedulers):
            raise RuntimeError(
                "Resume checkpoint optimizer/scheduler layout does not match current setup."
            )
        for optimizer, state_dict in zip(optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(state_dict)
        for scheduler, state_dict in zip(schedulers, scheduler_states, strict=True):
            scheduler.load_state_dict(state_dict)
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_wer = float(checkpoint.get("best_val_wer", float("inf")))
        logger.info(
            "resumed from %s starting_epoch=%s global_step=%s best_val_wer=%.4f intermediate_ctc_layers=%s intermediate_ctc_weight=%.3f blank_prune_layer=%s blank_prune_threshold=%.3f",
            resume_path,
            start_epoch,
            global_step,
            best_val_wer,
            list(intermediate_ctc_layers),
            intermediate_ctc_weight,
            blank_prune_layer,
            blank_prune_threshold,
        )

    if is_main_process:
        cli_arguments_table = _build_trackio_cli_arguments_table(sys.argv[1:])
        trackio_run_name = _build_trackio_run_name(
            trackio_project=args.trackio_project,
            output_dir=output_dir,
            start_epoch=start_epoch,
            global_step=global_step,
            process_start_timestamp=process_start_timestamp,
        )
        trackio.init(
            project=args.trackio_project,
            name=trackio_run_name,
            space_id=args.trackio_space_id,
            config={
                **vars(args),
                "encoder_config": asdict(encoder_config),
                "train_samples": len(train_records),
                "val_samples": len(val_records),
                "active_optimizers": optimizer_names,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "featurizer_config": featurizer.config_dict(),
                "intermediate_ctc_layers": list(intermediate_ctc_layers),
                "intermediate_ctc_layer": (
                    intermediate_ctc_layers[0] if len(intermediate_ctc_layers) == 1 else None
                ),
                "intermediate_ctc_weight": intermediate_ctc_weight,
                "blank_prune_layer": blank_prune_layer,
                "blank_prune_threshold": blank_prune_threshold,
                "blank_prune_min_keep_frames": blank_prune_min_keep_frames,
                "split_audit": split_audit,
                "distributed": distributed,
                "world_size": world_size,
            },
        )
        if cli_arguments_table is not None:
            trackio.log({"cli_arguments": cli_arguments_table})
        logger.info(
            "trackio initialized project=%s run_name=%s trackio_dir=%s elapsed_since_start=%s",
            args.trackio_project,
            trackio_run_name,
            trackio_dir,
            _format_elapsed_seconds(time.perf_counter() - process_start_monotonic),
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        _set_dataloader_epoch(train_loader, epoch)
        logger.info(
            "epoch %s/%s started train_batches=%s val_batches=%s grad_accumulation=%s",
            epoch,
            args.epochs,
            train_batches,
            val_batches,
            args.gradient_accumulation_steps,
        )
        forward_model.train()
        running_loss = 0.0
        running_main_ctc_loss = 0.0
        running_intermediate_ctc_loss = 0.0
        running_blank_logit_regularization_loss = 0.0
        running_aed_loss = 0.0
        running_liberta_distill_loss = 0.0
        running_audio_teacher_loss = 0.0
        running_sample_count = 0.0
        blank_starvation_warning_logged = False
        tune_effective_frames = 0
        tune_padded_frames = 0
        tune_target_tokens = 0
        accumulated_global_batch_size = 0.0
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            if batch is None:
                logger.warning("skipping empty training batch after dataset filtering")
                continue
            should_step = (
                batch_index % args.gradient_accumulation_steps == 0 or batch_index == train_batches
            )
            if (batch_index - 1) % args.gradient_accumulation_steps == 0:
                tune_effective_frames = 0
                tune_padded_frames = 0
                tune_target_tokens = 0
                accumulated_global_batch_size = 0.0
                if (
                    is_main_process
                    and args.memory_tune_steps > 0
                    and device.type == "cuda"
                    and torch.cuda.is_available()
                    and global_step < args.memory_tune_steps
                ):
                    torch.cuda.reset_peak_memory_stats(device)
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            local_batch_size = float(features.size(0))
            global_batch_size = _distributed_sum_float(
                local_batch_size,
                device=device,
                distributed=distributed,
            )
            accumulated_global_batch_size += global_batch_size
            tune_effective_frames += int(feature_lengths.sum().item())
            tune_padded_frames += int(features.size(0) * features.size(1))
            tune_target_tokens += int(target_lengths.sum().item())
            if is_main_process and batch_index == 1:
                logger.info(
                    "epoch=%s first_train_batch_ready elapsed=%s batch_size=%s max_feature_frames=%s target_tokens=%s",
                    epoch,
                    _format_elapsed_seconds(time.perf_counter() - epoch_start_time),
                    int(features.size(0)),
                    int(feature_lengths.max().item()),
                    int(target_lengths.sum().item()),
                )
            if aed_decoder_enabled:
                if model.aed_decoder is None:
                    raise RuntimeError("AED decoder was enabled but not constructed on the model.")
                decoder_inputs, decoder_targets, decoder_target_lengths = _build_aed_targets(
                    targets,
                    target_lengths,
                    bos_id=model.aed_decoder.bos_id,
                    eos_id=model.aed_decoder.eos_id,
                    token_offset=model.aed_decoder.token_offset,
                    pad_id=model.aed_decoder.pad_id,
                )
                decoder_inputs = decoder_inputs.to(device)
                decoder_targets = decoder_targets.to(device)
                decoder_target_lengths = decoder_target_lengths.to(device)
            else:
                decoder_inputs = None
                decoder_targets = None
                decoder_target_lengths = None
            next_global_step = global_step + 1 if should_step else global_step
            should_log_step = should_step and next_global_step % args.log_every == 0
            should_collect_log_probs = should_log_step and is_main_process

            sync_context = (
                forward_model.no_sync()
                if distributed and not should_step and isinstance(forward_model, DDP)
                else nullcontext()
            )
            with sync_context:
                with _autocast_context(device, args.dtype, fp8_recipe=fp8_recipe):
                    forward_outputs = forward_model(
                        features,
                        feature_lengths,
                        return_training_outputs=True,
                        targets=targets,
                        target_lengths=target_lengths,
                        blank_id=tokenizer.blank_id,
                        return_main_log_probs=should_collect_log_probs,
                        decoder_inputs=decoder_inputs,
                        liberta_lengths=decoder_target_lengths,
                    )
                    main_ctc_loss = forward_outputs["main_ctc_loss"]
                    intermediate_ctc_losses_map = forward_outputs["intermediate_ctc_losses"]
                    intermediate_ctc_diagnostics_map = forward_outputs.get(
                        "intermediate_ctc_diagnostics",
                        {},
                    )
                    output_lengths = forward_outputs["output_lengths"]
                    encoded = forward_outputs["encoded"]
                    log_probs = forward_outputs.get("main_log_probs")
                    logits = forward_outputs.get("main_logits")
                    blank_logit_regularization_loss = forward_outputs.get(
                        "blank_logit_regularization_loss"
                    )
                    aed_logits = forward_outputs.get("aed_logits")
                    liberta_student_embeddings = forward_outputs.get("liberta_student_embeddings")
                    audio_teacher_student_states = forward_outputs.get(
                        "audio_teacher_student_states"
                    )
                    if intermediate_ctc_losses_map:
                        intermediate_ctc_loss = torch.stack(
                            [
                                intermediate_ctc_losses_map[layer_index]
                                for layer_index in intermediate_ctc_layers
                            ]
                        ).mean()
                        loss = (
                            1.0 - intermediate_ctc_weight
                        ) * main_ctc_loss + intermediate_ctc_weight * intermediate_ctc_loss
                    else:
                        intermediate_ctc_loss = None
                        loss = main_ctc_loss
                    if blank_logit_regularization_loss is not None:
                        loss = loss + (
                            args.blank_logit_regularization_weight
                            * blank_logit_regularization_loss
                        )
                    if aed_logits is not None and decoder_targets is not None:
                        aed_loss = _aed_cross_entropy_loss(
                            aed_logits,
                            decoder_targets,
                            pad_id=model.aed_decoder.pad_id,
                        )
                        loss = (1.0 - args.aed_loss_weight) * loss + args.aed_loss_weight * aed_loss
                    else:
                        aed_loss = None
                if liberta_teacher is not None and liberta_student_embeddings is not None:
                    teacher_embeddings = liberta_teacher.encode(batch["transcripts"]).to(
                        device=liberta_student_embeddings.device,
                        dtype=liberta_student_embeddings.dtype,
                    )
                    liberta_distill_loss = F.mse_loss(
                        F.normalize(liberta_student_embeddings.float(), dim=-1),
                        F.normalize(teacher_embeddings.float(), dim=-1),
                    )
                    loss = loss + (args.liberta_distill_weight * liberta_distill_loss)
                else:
                    liberta_distill_loss = None
                if audio_teacher is not None and audio_teacher_student_states is not None:
                    teacher_outputs = audio_teacher.encode_waveforms(
                        batch["waveforms"],
                        batch["waveform_lengths"],
                        sample_rates=batch.get("sample_rates"),
                    )
                    teacher_embeddings = teacher_outputs["pooled_hidden"].to(
                        device=audio_teacher_student_states.device,
                        dtype=audio_teacher_student_states.dtype,
                    )
                    if args.audio_teacher_objective == "hidden_cosine":
                        audio_teacher_loss = (
                            1.0
                            - F.cosine_similarity(
                                audio_teacher_student_states.float(),
                                teacher_embeddings.float(),
                                dim=-1,
                            )
                        ).mean()
                    else:
                        audio_teacher_loss = F.mse_loss(
                            F.normalize(audio_teacher_student_states.float(), dim=-1),
                            F.normalize(teacher_embeddings.float(), dim=-1),
                        )
                    loss = loss + (args.audio_teacher_weight * audio_teacher_loss)
                else:
                    audio_teacher_loss = None
                running_loss += float(loss.item()) * local_batch_size
                running_main_ctc_loss += float(main_ctc_loss.item()) * local_batch_size
                running_intermediate_ctc_loss += float(
                    intermediate_ctc_loss.item() if intermediate_ctc_loss is not None else 0.0
                ) * local_batch_size
                running_blank_logit_regularization_loss += float(
                    blank_logit_regularization_loss.item()
                    if blank_logit_regularization_loss is not None
                    else 0.0
                ) * local_batch_size
                running_aed_loss += float(aed_loss.item() if aed_loss is not None else 0.0) * local_batch_size
                running_liberta_distill_loss += float(
                    liberta_distill_loss.item() if liberta_distill_loss is not None else 0.0
                ) * local_batch_size
                running_audio_teacher_loss += float(
                    audio_teacher_loss.item() if audio_teacher_loss is not None else 0.0
                ) * local_batch_size
                running_sample_count += local_batch_size
                backward_loss = loss * local_batch_size
                if distributed and dist.is_initialized():
                    backward_loss = backward_loss * float(world_size)
                scaler.scale(backward_loss).backward()
            if should_step:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                _divide_gradients_in_place(
                    model.parameters(),
                    max(1.0, accumulated_global_batch_size),
                )
                grad_norm = float(_compute_grad_norm(model.parameters()).item())
                head_row_stats = _classifier_head_row_diagnostics(model)
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()
                global_step += 1
                if ema is not None:
                    ema.update(model)

                if (
                    is_main_process
                    and args.memory_tune_steps > 0
                    and global_step <= args.memory_tune_steps
                ):
                    _log_batch_autotune_snapshot(
                        logger,
                        epoch=epoch,
                        global_step=global_step,
                        batch_index=batch_index,
                        total_batches=train_batches,
                        optimizer_step_index=global_step,
                        effective_frames=tune_effective_frames,
                        padded_frames=tune_padded_frames,
                        target_tokens=tune_target_tokens,
                        device=device,
                    )

                if should_log_step:
                    train_loss_step = _distributed_weighted_mean(
                        float(loss.item()),
                        weight=local_batch_size,
                        device=device,
                        distributed=distributed,
                    )
                    train_main_ctc_loss_step = _distributed_weighted_mean(
                        float(main_ctc_loss.item()),
                        weight=local_batch_size,
                        device=device,
                        distributed=distributed,
                    )
                    train_intermediate_ctc_loss_step = _distributed_weighted_mean(
                        float(
                            intermediate_ctc_loss.item()
                            if intermediate_ctc_loss is not None
                            else 0.0
                        ),
                        weight=local_batch_size,
                        device=device,
                        distributed=distributed,
                    )
                    train_blank_logit_regularization_loss_step = _distributed_weighted_mean(
                        float(
                            blank_logit_regularization_loss.item()
                            if blank_logit_regularization_loss is not None
                            else 0.0
                        ),
                        weight=local_batch_size,
                        device=device,
                        distributed=distributed,
                    )
                    train_aed_loss_step = _distributed_weighted_mean(
                        float(aed_loss.item() if aed_loss is not None else 0.0),
                        weight=local_batch_size,
                        device=device,
                        distributed=distributed,
                    )
                    train_liberta_distill_loss_step = _distributed_weighted_mean(
                        float(
                            liberta_distill_loss.item() if liberta_distill_loss is not None else 0.0
                        ),
                        weight=local_batch_size,
                        device=device,
                        distributed=distributed,
                    )
                    train_audio_teacher_loss_step = _distributed_weighted_mean(
                        float(audio_teacher_loss.item() if audio_teacher_loss is not None else 0.0),
                        weight=local_batch_size,
                        device=device,
                        distributed=distributed,
                    )
                    grad_norm = _distributed_mean(
                        grad_norm,
                        device=device,
                        distributed=distributed,
                    )
                    head_row_stats = {
                        name: _distributed_mean(
                            value,
                            device=device,
                            distributed=distributed,
                        )
                        for name, value in head_row_stats.items()
                    }
                    batch_audio_minutes = _distributed_mean(
                        _frames_to_minutes(
                            tune_effective_frames,
                            hop_length=args.hop_length,
                        ),
                        device=device,
                        distributed=distributed,
                    )
                    max_feature_frames = _distributed_max_int(
                        int(feature_lengths.max().item()),
                        device=device,
                        distributed=distributed,
                    )
                    if log_probs is not None:
                        ctc_diagnostics = summarize_ctc_batch_diagnostics(
                            ctc_batch_diagnostics(
                                log_probs,
                                output_lengths,
                                tokenizer,
                                targets=targets,
                                target_lengths=target_lengths,
                            )
                        )
                        ctc_logit_stats = summarize_ctc_logit_diagnostics(
                            ctc_logit_diagnostics(
                                logits if logits is not None else log_probs,
                                output_lengths,
                                tokenizer,
                            )
                        )
                        encoder_stats = summarize_encoder_output_diagnostics(
                            encoder_output_diagnostics(
                                encoded,
                                output_lengths,
                            )
                        )
                        top_token_histogram = top_emitted_token_histogram(
                            log_probs,
                            output_lengths,
                            tokenizer,
                            top_k=5,
                        )
                    else:
                        ctc_diagnostics = {
                            "avg_blank_probability": 0.0,
                            "argmax_blank_fraction": 0.0,
                            "avg_top_nonblank_probability": 0.0,
                            "avg_output_frames": 0.0,
                            "avg_target_tokens": 0.0,
                            "target_tokens_per_frame": 0.0,
                            "impossible_sample_fraction": 0.0,
                            "tight_sample_fraction": 0.0,
                        }
                        ctc_logit_stats = {
                            "avg_blank_logit": 0.0,
                            "avg_top_logit": 0.0,
                            "avg_top2_margin": 0.0,
                            "avg_blank_nonblank_margin": 0.0,
                            "avg_entropy": 0.0,
                        }
                        encoder_stats = {
                            "avg_mean": 0.0,
                            "avg_std": 0.0,
                            "avg_token_l2_norm": 0.0,
                        }
                        top_token_histogram = []
                if is_main_process and should_log_step:
                    learning_rates = {
                        f"learning_rate_{name}": optimizer.param_groups[0]["lr"]
                        for name, optimizer in zip(optimizer_names, optimizers, strict=True)
                    }
                    memory_snapshot = _format_memory_snapshot(device)
                    intermediate_loss_detail = (
                        " ".join(
                            f"layer{layer_index}_ctc={float(intermediate_ctc_losses_map[layer_index].item()):.4f}"
                            for layer_index in sorted(intermediate_ctc_losses_map)
                        )
                        if intermediate_ctc_losses_map
                        else ""
                    )
                    intermediate_length_detail = (
                        " ".join(
                            (
                                f"layer{layer_index}_ctc_impossible="
                                f"{(intermediate_ctc_diagnostics_map[layer_index].get('impossible_sample_count', 0.0) / max(1.0, intermediate_ctc_diagnostics_map[layer_index].get('sample_count', 1.0))):.4f} "
                                f"layer{layer_index}_ctc_tight="
                                f"{(intermediate_ctc_diagnostics_map[layer_index].get('tight_sample_count', 0.0) / max(1.0, intermediate_ctc_diagnostics_map[layer_index].get('sample_count', 1.0))):.4f}"
                            )
                            for layer_index in sorted(intermediate_ctc_diagnostics_map)
                        )
                        if intermediate_ctc_diagnostics_map
                        else ""
                    )
                    intermediate_blank_detail = (
                        " ".join(
                            (
                                f"layer{layer_index}_avg_blank_prob="
                                f"{summarize_ctc_batch_diagnostics(intermediate_ctc_diagnostics_map[layer_index])['avg_blank_probability']:.4f} "
                                f"layer{layer_index}_argmax_blank_frac="
                                f"{summarize_ctc_batch_diagnostics(intermediate_ctc_diagnostics_map[layer_index])['argmax_blank_fraction']:.4f} "
                                f"layer{layer_index}_avg_top_nonblank_prob="
                                f"{summarize_ctc_batch_diagnostics(intermediate_ctc_diagnostics_map[layer_index])['avg_top_nonblank_probability']:.4f} "
                                f"layer{layer_index}_avg_blank_nonblank_margin="
                                f"{summarize_ctc_logit_diagnostics(intermediate_ctc_diagnostics_map[layer_index])['avg_blank_nonblank_margin']:.4f}"
                            )
                            for layer_index in sorted(intermediate_ctc_diagnostics_map)
                        )
                        if intermediate_ctc_diagnostics_map
                        else ""
                    )
                    logger.info(
                        (
                            "epoch=%s step=%s/%s global_step=%s train_loss=%.4f "
                            "train_main_ctc_loss=%.4f train_intermediate_ctc_loss=%.4f "
                            "train_blank_logit_regularization_loss=%.4f "
                            "train_aed_loss=%.4f train_liberta_distill_loss=%.4f "
                            "train_audio_teacher_loss=%.4f "
                            "batch_audio_minutes=%.2f grad_norm=%.4f max_feature_frames=%s "
                            "train_avg_blank_prob=%.4f train_argmax_blank_frac=%.4f "
                            "train_avg_top_nonblank_prob=%.4f train_target_tokens_per_frame=%.4f "
                            "train_ctc_impossible_frac=%.4f train_ctc_tight_frac=%.4f "
                            "train_avg_blank_logit=%.4f train_avg_top_logit=%.4f "
                            "train_avg_top2_margin=%.4f train_avg_blank_nonblank_margin=%.4f "
                            "train_avg_entropy=%.4f "
                            "train_blank_bias=%.4f train_blank_weight_norm=%.4f "
                            "train_nonblank_weight_norm_mean=%.4f "
                            "train_blank_weight_grad_norm=%.4f "
                            "train_nonblank_weight_grad_norm_mean=%.4f "
                            "train_blank_bias_grad=%.4f train_nonblank_bias_grad_mean=%.4f "
                            "train_encoder_mean=%.4f train_encoder_std=%.4f "
                            "train_encoder_token_l2_norm=%.4f "
                            "%s %s %s %s %s"
                        ),
                        epoch,
                        batch_index,
                        train_batches,
                        global_step,
                        train_loss_step,
                        train_main_ctc_loss_step,
                        train_intermediate_ctc_loss_step,
                        train_blank_logit_regularization_loss_step,
                        train_aed_loss_step,
                        train_liberta_distill_loss_step,
                        train_audio_teacher_loss_step,
                        batch_audio_minutes,
                        grad_norm,
                        max_feature_frames,
                        ctc_diagnostics["avg_blank_probability"],
                        ctc_diagnostics["argmax_blank_fraction"],
                        ctc_diagnostics["avg_top_nonblank_probability"],
                        ctc_diagnostics["target_tokens_per_frame"],
                        ctc_diagnostics["impossible_sample_fraction"],
                        ctc_diagnostics["tight_sample_fraction"],
                        ctc_logit_stats["avg_blank_logit"],
                        ctc_logit_stats["avg_top_logit"],
                        ctc_logit_stats["avg_top2_margin"],
                        ctc_logit_stats["avg_blank_nonblank_margin"],
                        ctc_logit_stats["avg_entropy"],
                        head_row_stats["blank_bias"],
                        head_row_stats["blank_weight_norm"],
                        head_row_stats["nonblank_weight_norm_mean"],
                        head_row_stats["blank_weight_grad_norm"],
                        head_row_stats["nonblank_weight_grad_norm_mean"],
                        head_row_stats["blank_bias_grad"],
                        head_row_stats["nonblank_bias_grad_mean"],
                        encoder_stats["avg_mean"],
                        encoder_stats["avg_std"],
                        encoder_stats["avg_token_l2_norm"],
                        intermediate_loss_detail,
                        intermediate_length_detail,
                        intermediate_blank_detail,
                        memory_snapshot,
                        " ".join(f"{name}={value:.6g}" for name, value in learning_rates.items()),
                    )
                    if log_probs is not None:
                        top_token_histogram_text = ", ".join(
                            f"{token_text}:{fraction:.3f}"
                            for _token_id, fraction, token_text in top_token_histogram
                        )
                        if (
                            not blank_starvation_warning_logged
                            and _should_warn_on_blank_starvation(
                                global_step=global_step,
                                avg_blank_probability=ctc_diagnostics["avg_blank_probability"],
                                argmax_blank_fraction=ctc_diagnostics["argmax_blank_fraction"],
                                avg_top_nonblank_probability=ctc_diagnostics[
                                    "avg_top_nonblank_probability"
                                ],
                            )
                        ):
                            blank_starvation_warning_logged = True
                            logger.warning(
                                "ctc blank starvation detected at global_step=%s: "
                                "avg_blank_prob=%.4f argmax_blank_frac=%.4f "
                                "avg_top_nonblank_prob=%.4f. Blank never wins a frame, so "
                                "training will usually collapse into repeated nonblank pieces. "
                                "This is commonly caused by overly aggressive blank suppression; "
                                "if you set --blank-logit-offset > 0, retry with "
                                "--blank-logit-offset 0.0 first.",
                                global_step,
                                ctc_diagnostics["avg_blank_probability"],
                                ctc_diagnostics["argmax_blank_fraction"],
                                ctc_diagnostics["avg_top_nonblank_probability"],
                            )
                        preview_greedy_hypothesis, preview_beam_hypothesis = (
                            _decode_train_preview_hypotheses(
                                log_probs=log_probs,
                                output_lengths=output_lengths,
                                tokenizer=tokenizer,
                                beam_size=args.beam_size,
                                lm_scorer=lm_scorer,
                                lm_weight=args.lm_weight,
                                beam_length_bonus=args.beam_length_bonus,
                            )
                        )
                        logger.info(
                            "train preview ref=%r hyp_greedy=%r hyp_beam=%r avg_output_frames=%.1f avg_target_tokens=%.1f top_tokens=%s",
                            _truncate_for_log(batch["transcripts"][0]),
                            _truncate_for_log(preview_greedy_hypothesis),
                            _truncate_for_log(preview_beam_hypothesis),
                            ctc_diagnostics["avg_output_frames"],
                            ctc_diagnostics["avg_target_tokens"],
                            top_token_histogram_text,
                        )
                    trackio.log(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss_step": train_loss_step,
                            "train_main_ctc_loss_step": train_main_ctc_loss_step,
                            "train_intermediate_ctc_loss_step": train_intermediate_ctc_loss_step,
                            "train_blank_logit_regularization_loss_step": (
                                train_blank_logit_regularization_loss_step
                            ),
                            "train_aed_loss_step": train_aed_loss_step,
                            "train_liberta_distill_loss_step": train_liberta_distill_loss_step,
                            "train_audio_teacher_loss_step": train_audio_teacher_loss_step,
                            "grad_norm": grad_norm,
                            "train_max_feature_frames_step": max_feature_frames,
                            "train_avg_blank_probability_step": ctc_diagnostics[
                                "avg_blank_probability"
                            ],
                            "train_argmax_blank_fraction_step": ctc_diagnostics[
                                "argmax_blank_fraction"
                            ],
                            "train_avg_top_nonblank_probability_step": ctc_diagnostics[
                                "avg_top_nonblank_probability"
                            ],
                            "train_target_tokens_per_frame_step": ctc_diagnostics[
                                "target_tokens_per_frame"
                            ],
                            "train_ctc_impossible_frac_step": ctc_diagnostics[
                                "impossible_sample_fraction"
                            ],
                            "train_ctc_tight_frac_step": ctc_diagnostics[
                                "tight_sample_fraction"
                            ],
                            "train_avg_blank_logit_step": ctc_logit_stats["avg_blank_logit"],
                            "train_avg_top_logit_step": ctc_logit_stats["avg_top_logit"],
                            "train_avg_top2_margin_step": ctc_logit_stats["avg_top2_margin"],
                            "train_avg_blank_nonblank_margin_step": ctc_logit_stats[
                                "avg_blank_nonblank_margin"
                            ],
                            "train_avg_entropy_step": ctc_logit_stats["avg_entropy"],
                            "train_encoder_mean_step": encoder_stats["avg_mean"],
                            "train_encoder_std_step": encoder_stats["avg_std"],
                            "train_encoder_token_l2_norm_step": encoder_stats[
                                "avg_token_l2_norm"
                            ],
                            "train_blank_bias_step": head_row_stats["blank_bias"],
                            "train_blank_weight_norm_step": head_row_stats[
                                "blank_weight_norm"
                            ],
                            "train_nonblank_weight_norm_mean_step": head_row_stats[
                                "nonblank_weight_norm_mean"
                            ],
                            "train_blank_weight_grad_norm_step": head_row_stats[
                                "blank_weight_grad_norm"
                            ],
                            "train_nonblank_weight_grad_norm_mean_step": head_row_stats[
                                "nonblank_weight_grad_norm_mean"
                            ],
                            "train_blank_bias_grad_step": head_row_stats[
                                "blank_bias_grad"
                            ],
                            "train_nonblank_bias_grad_mean_step": head_row_stats[
                                "nonblank_bias_grad_mean"
                            ],
                            "ema_decay": ema.current_decay() if ema is not None else 0.0,
                            "cpu_rss_bytes": _read_proc_status_memory_bytes("VmRSS:") or 0,
                            "cpu_peak_rss_bytes": (
                                _read_proc_status_memory_bytes("VmHWM:")
                                or _peak_process_memory_bytes()
                                or 0
                            ),
                            "cuda_allocated_bytes": (
                                torch.cuda.memory_allocated(device)
                                if device.type == "cuda" and torch.cuda.is_available()
                                else 0
                            ),
                            "cuda_reserved_bytes": (
                                torch.cuda.memory_reserved(device)
                                if device.type == "cuda" and torch.cuda.is_available()
                                else 0
                            ),
                            "cuda_peak_allocated_bytes": (
                                torch.cuda.max_memory_allocated(device)
                                if device.type == "cuda" and torch.cuda.is_available()
                                else 0
                            ),
                            "cuda_peak_reserved_bytes": (
                                torch.cuda.max_memory_reserved(device)
                                if device.type == "cuda" and torch.cuda.is_available()
                                else 0
                            ),
                            **learning_rates,
                            **_build_trackio_grouped_metrics(
                                groups={
                                    "train": {
                                        "loss_step": train_loss_step,
                                        "main_ctc_loss_step": train_main_ctc_loss_step,
                                        "intermediate_ctc_loss_step": train_intermediate_ctc_loss_step,
                                        "blank_logit_regularization_loss_step": (
                                            train_blank_logit_regularization_loss_step
                                        ),
                                        "aed_loss_step": train_aed_loss_step,
                                        "liberta_distill_loss_step": (
                                            train_liberta_distill_loss_step
                                        ),
                                        "audio_teacher_loss_step": train_audio_teacher_loss_step,
                                        "max_feature_frames_step": max_feature_frames,
                                        "avg_blank_probability_step": ctc_diagnostics[
                                            "avg_blank_probability"
                                        ],
                                        "argmax_blank_fraction_step": ctc_diagnostics[
                                            "argmax_blank_fraction"
                                        ],
                                        "avg_top_nonblank_probability_step": (
                                            ctc_diagnostics["avg_top_nonblank_probability"]
                                        ),
                                        "target_tokens_per_frame_step": ctc_diagnostics[
                                            "target_tokens_per_frame"
                                        ],
                                        "ctc_impossible_frac_step": ctc_diagnostics[
                                            "impossible_sample_fraction"
                                        ],
                                        "ctc_tight_frac_step": ctc_diagnostics[
                                            "tight_sample_fraction"
                                        ],
                                        "avg_blank_logit_step": ctc_logit_stats[
                                            "avg_blank_logit"
                                        ],
                                        "avg_top_logit_step": ctc_logit_stats["avg_top_logit"],
                                        "avg_top2_margin_step": ctc_logit_stats[
                                            "avg_top2_margin"
                                        ],
                                        "avg_blank_nonblank_margin_step": ctc_logit_stats[
                                            "avg_blank_nonblank_margin"
                                        ],
                                        "avg_entropy_step": ctc_logit_stats["avg_entropy"],
                                        "encoder_mean_step": encoder_stats["avg_mean"],
                                        "encoder_std_step": encoder_stats["avg_std"],
                                        "encoder_token_l2_norm_step": encoder_stats[
                                            "avg_token_l2_norm"
                                        ],
                                        "blank_bias_step": head_row_stats["blank_bias"],
                                        "blank_weight_norm_step": head_row_stats[
                                            "blank_weight_norm"
                                        ],
                                        "nonblank_weight_norm_mean_step": head_row_stats[
                                            "nonblank_weight_norm_mean"
                                        ],
                                        "blank_weight_grad_norm_step": head_row_stats[
                                            "blank_weight_grad_norm"
                                        ],
                                        "nonblank_weight_grad_norm_mean_step": head_row_stats[
                                            "nonblank_weight_grad_norm_mean"
                                        ],
                                        "blank_bias_grad_step": head_row_stats[
                                            "blank_bias_grad"
                                        ],
                                        "nonblank_bias_grad_mean_step": head_row_stats[
                                            "nonblank_bias_grad_mean"
                                        ],
                                        "avg_output_frames_step": ctc_diagnostics[
                                            "avg_output_frames"
                                        ],
                                        "avg_target_tokens_step": ctc_diagnostics[
                                            "avg_target_tokens"
                                        ],
                                        "grad_norm": grad_norm,
                                        "ema_decay": (
                                            ema.current_decay() if ema is not None else 0.0
                                        ),
                                        "batch_audio_minutes": batch_audio_minutes,
                                    },
                                    "system": {
                                        "cpu_rss_bytes": _read_proc_status_memory_bytes("VmRSS:") or 0,
                                        "cpu_peak_rss_bytes": (
                                            _read_proc_status_memory_bytes("VmHWM:")
                                            or _peak_process_memory_bytes()
                                            or 0
                                        ),
                                        "cuda_allocated_bytes": (
                                            torch.cuda.memory_allocated(device)
                                            if device.type == "cuda" and torch.cuda.is_available()
                                            else 0
                                        ),
                                        "cuda_reserved_bytes": (
                                            torch.cuda.memory_reserved(device)
                                            if device.type == "cuda" and torch.cuda.is_available()
                                            else 0
                                        ),
                                        "cuda_peak_allocated_bytes": (
                                            torch.cuda.max_memory_allocated(device)
                                            if device.type == "cuda" and torch.cuda.is_available()
                                            else 0
                                        ),
                                        "cuda_peak_reserved_bytes": (
                                            torch.cuda.max_memory_reserved(device)
                                            if device.type == "cuda" and torch.cuda.is_available()
                                            else 0
                                        ),
                                    },
                                    "lr": {
                                        name.removeprefix("learning_rate_"): value
                                        for name, value in learning_rates.items()
                                    },
                                },
                            ),
                        }
                    )

                if args.validate_every_steps > 0 and global_step % args.validate_every_steps == 0:
                    train_metrics = {
                        "train_loss": _distributed_weighted_mean(
                            running_loss / max(1.0, running_sample_count),
                            weight=running_sample_count,
                            device=device,
                            distributed=distributed,
                        ),
                        "train_main_ctc_loss": _distributed_weighted_mean(
                            running_main_ctc_loss / max(1.0, running_sample_count),
                            weight=running_sample_count,
                            device=device,
                            distributed=distributed,
                        ),
                        "train_intermediate_ctc_loss": _distributed_weighted_mean(
                            running_intermediate_ctc_loss / max(1.0, running_sample_count),
                            weight=running_sample_count,
                            device=device,
                            distributed=distributed,
                        ),
                        "train_blank_logit_regularization_loss": _distributed_weighted_mean(
                            running_blank_logit_regularization_loss / max(1.0, running_sample_count),
                            weight=running_sample_count,
                            device=device,
                            distributed=distributed,
                        ),
                        "train_aed_loss": _distributed_weighted_mean(
                            running_aed_loss / max(1.0, running_sample_count),
                            weight=running_sample_count,
                            device=device,
                            distributed=distributed,
                        ),
                        "train_liberta_distill_loss": _distributed_weighted_mean(
                            running_liberta_distill_loss / max(1.0, running_sample_count),
                            weight=running_sample_count,
                            device=device,
                            distributed=distributed,
                        ),
                        "train_audio_teacher_loss": _distributed_weighted_mean(
                            running_audio_teacher_loss / max(1.0, running_sample_count),
                            weight=running_sample_count,
                            device=device,
                            distributed=distributed,
                        ),
                    }
                    if distributed:
                        _distributed_barrier()
                    if is_main_process:
                        logger.info(
                            (
                                "epoch %s step=%s/%s global_step=%s reached validation interval; "
                                "starting validation"
                            ),
                            epoch,
                            batch_index,
                            train_batches,
                            global_step,
                        )
                    best_val_wer = _evaluate_and_checkpoint(
                        model=model,
                        val_loader=val_loader,
                        criterion=criterion,
                        tokenizer=tokenizer,
                        device=device,
                        dtype=args.dtype,
                        fp8_recipe=fp8_recipe,
                        decode_strategy=args.decode_strategy,
                        beam_size=args.beam_size,
                        lm_scorer=lm_scorer,
                        lm_weight=args.lm_weight,
                        beam_length_bonus=args.beam_length_bonus,
                        example_limit=args.example_limit,
                        intermediate_ctc_weight=intermediate_ctc_weight,
                        aed_loss_weight=args.aed_loss_weight,
                        liberta_teacher=liberta_teacher,
                        liberta_distill_weight=args.liberta_distill_weight,
                        audio_teacher=audio_teacher,
                        audio_teacher_weight=args.audio_teacher_weight,
                        audio_teacher_objective=args.audio_teacher_objective,
                        ema=ema,
                        validation_model_source=args.validation_model_source,
                        train_metrics=train_metrics,
                        epoch=epoch,
                        global_step=global_step,
                        output_dir=output_dir,
                        encoder_config=encoder_config,
                        featurizer=featurizer,
                        optimizers=optimizers,
                        optimizer_names=optimizer_names,
                        schedulers=schedulers,
                        scaler=scaler,
                        args=args,
                        best_val_wer=best_val_wer,
                        split_audit=split_audit,
                        logger=logger,
                        save_last_checkpoint=False,
                        report_stem=f"epoch_{epoch:04d}_step_{global_step:08d}",
                        distributed=distributed,
                        is_main_process=is_main_process,
                    )
                    best_val_wer = _synchronize_best_val_wer(
                        best_val_wer,
                        device=device,
                        distributed=distributed,
                    )
                    if distributed:
                        _distributed_barrier()

        local_train_mean = running_loss / max(1.0, running_sample_count)
        local_train_main_ctc_mean = running_main_ctc_loss / max(1.0, running_sample_count)
        local_train_intermediate_ctc_mean = running_intermediate_ctc_loss / max(
            1.0, running_sample_count
        )
        local_train_blank_reg_mean = running_blank_logit_regularization_loss / max(
            1.0, running_sample_count
        )
        local_train_aed_mean = running_aed_loss / max(1.0, running_sample_count)
        local_train_liberta_mean = running_liberta_distill_loss / max(1.0, running_sample_count)
        local_train_audio_teacher_mean = running_audio_teacher_loss / max(
            1.0, running_sample_count
        )
        train_loss = _distributed_weighted_mean(
            local_train_mean,
            weight=running_sample_count,
            device=device,
            distributed=distributed,
        )
        train_main_ctc_loss = _distributed_weighted_mean(
            local_train_main_ctc_mean,
            weight=running_sample_count,
            device=device,
            distributed=distributed,
        )
        train_intermediate_ctc_loss = _distributed_weighted_mean(
            local_train_intermediate_ctc_mean,
            weight=running_sample_count,
            device=device,
            distributed=distributed,
        )
        train_blank_logit_regularization_loss = _distributed_weighted_mean(
            local_train_blank_reg_mean,
            weight=running_sample_count,
            device=device,
            distributed=distributed,
        )
        train_aed_loss = _distributed_weighted_mean(
            local_train_aed_mean,
            weight=running_sample_count,
            device=device,
            distributed=distributed,
        )
        train_liberta_distill_loss = _distributed_weighted_mean(
            local_train_liberta_mean,
            weight=running_sample_count,
            device=device,
            distributed=distributed,
        )
        train_audio_teacher_loss = _distributed_weighted_mean(
            local_train_audio_teacher_mean,
            weight=running_sample_count,
            device=device,
            distributed=distributed,
        )
        if distributed:
            _distributed_barrier()
        if is_main_process:
            logger.info(
                "epoch %s training complete train_loss=%.4f train_main_ctc_loss=%.4f train_intermediate_ctc_loss=%.4f train_blank_logit_regularization_loss=%.4f train_aed_loss=%.4f train_liberta_distill_loss=%.4f train_audio_teacher_loss=%.4f elapsed=%s, starting validation",
                epoch,
                train_loss,
                train_main_ctc_loss,
                train_intermediate_ctc_loss,
                train_blank_logit_regularization_loss,
                train_aed_loss,
                train_liberta_distill_loss,
                train_audio_teacher_loss,
                _format_elapsed_seconds(time.perf_counter() - epoch_start_time),
            )
        best_val_wer = _evaluate_and_checkpoint(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            tokenizer=tokenizer,
            device=device,
            dtype=args.dtype,
            fp8_recipe=fp8_recipe,
            decode_strategy=args.decode_strategy,
            beam_size=args.beam_size,
            lm_scorer=lm_scorer,
            lm_weight=args.lm_weight,
            beam_length_bonus=args.beam_length_bonus,
            example_limit=args.example_limit,
            intermediate_ctc_weight=intermediate_ctc_weight,
            aed_loss_weight=args.aed_loss_weight,
            liberta_teacher=liberta_teacher,
            liberta_distill_weight=args.liberta_distill_weight,
            audio_teacher=audio_teacher,
            audio_teacher_weight=args.audio_teacher_weight,
            audio_teacher_objective=args.audio_teacher_objective,
            ema=ema,
            validation_model_source=args.validation_model_source,
            train_metrics={
                "train_loss": train_loss,
                "train_main_ctc_loss": train_main_ctc_loss,
                "train_intermediate_ctc_loss": train_intermediate_ctc_loss,
                "train_blank_logit_regularization_loss": train_blank_logit_regularization_loss,
                "train_aed_loss": train_aed_loss,
                "train_liberta_distill_loss": train_liberta_distill_loss,
                "train_audio_teacher_loss": train_audio_teacher_loss,
            },
            epoch=epoch,
            global_step=global_step,
            output_dir=output_dir,
            encoder_config=encoder_config,
            featurizer=featurizer,
            optimizers=optimizers,
            optimizer_names=optimizer_names,
            schedulers=schedulers,
            scaler=scaler,
            args=args,
            best_val_wer=best_val_wer,
            split_audit=split_audit,
            logger=logger,
            save_last_checkpoint=True,
            report_stem=f"epoch_{epoch:04d}",
            distributed=distributed,
            is_main_process=is_main_process,
        )
        best_val_wer = _synchronize_best_val_wer(
            best_val_wer,
            device=device,
            distributed=distributed,
        )
        if distributed:
            _distributed_barrier()

    if is_main_process:
        (output_dir / "train_summary.json").write_text(
            json.dumps(
                {
                    "best_val_wer": best_val_wer,
                    "variant": args.variant,
                    "keep_top_k": args.keep_top_k,
                    "decode_strategy": args.decode_strategy,
                    "distributed": distributed,
                    "world_size": world_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        trackio.finish()
        logger.info(
            "training finished epochs=%s best_val_wer=%.4f summary=%s",
            args.epochs,
            best_val_wer,
            output_dir / "train_summary.json",
        )
    if hasattr(train_records, "close"):
        train_records.close()
    if hasattr(val_records, "close"):
        val_records.close()
    if distributed and dist.is_initialized():
        _distributed_barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
