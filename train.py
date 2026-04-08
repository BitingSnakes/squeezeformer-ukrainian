from __future__ import annotations

import json
import os
import time
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
from squeezeformer_pytorch.model import SqueezeformerConfig, squeezeformer_variant
from squeezeformer_pytorch.runtime_types import DTypeChoice
from squeezeformer_pytorch.training.cli import (
    _resolve_block_pattern,
    _resolve_float_tuple,
    _resolve_scheduler_kwargs,
    _validate_device_argument,
    _validate_startup_args,
    parse_args,
)
from squeezeformer_pytorch.training.data_loading import (
    DiskBackedRecordStore,
    _build_disk_backed_record_store,
    _build_split_audit,
    _ensure_opus_decode_support,
    _load_records_from_dataset_roots,
    _load_train_val_records,
    _record_store_duration_hours,
    _resolve_dataset_roots,
    _resolve_dataset_sources,
    _resolve_validation_dataset_sources,
    _shard_records_for_rank,
)
from squeezeformer_pytorch.training.evaluation import (
    _aed_cross_entropy_loss,
    _build_aed_targets,
    _evaluate_and_checkpoint,
    decode_batch,
    evaluate,
    speaker_level_metrics,
)
from squeezeformer_pytorch.training.runtime import (
    ExponentialMovingAverage,
    FrozenLibertaTeacher,
    _autocast_context,
    _average_topk_checkpoints,
    _build_fp8_recipe,
    _compute_grad_norm,
    _configure_console_logger,
    _configure_trackio_storage,
    _format_elapsed_seconds,
    _format_memory_snapshot,
    _frames_to_minutes,
    _log_batch_autotune_snapshot,
    _peak_process_memory_bytes,
    _read_proc_status_memory_bytes,
    _resolve_aed_settings,
    _resolve_autocast_dtype,
    _resolve_blank_pruning_settings,
    _resolve_fp8_format,
    _resolve_intermediate_ctc_settings,
    _resolve_liberta_settings,
    _resolve_model_load_dtype,
    _resolve_resume_checkpoint_path,
    _safetensors_path,
    _update_top_checkpoints,
    _validate_device_ready,
    _validate_fp8_runtime,
    _validate_resume_checkpoint_payload,
    _variant_defaults,
    build_optimizer,
    build_paper_scheduler,
    resolve_device,
)


def main() -> None:
    args = parse_args()
    process_start_time = time.perf_counter()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _validate_startup_args(args, world_size=world_size)
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested_device = resolve_device(args.device)
    is_main_process = rank == 0
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
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
        validation_dataset_sources,
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
        logger.info(
            "resume checkpoint loaded path=%s elapsed=%s",
            resume_path,
            _format_elapsed_seconds(time.perf_counter() - stage_start_time),
        )
    stage_start_time = time.perf_counter()
    logger.info(
        "preparing tokenizer mode=%s resume=%s tokenizer_path=%s",
        args.tokenizer,
        checkpoint is not None,
        args.tokenizer_path or "auto",
    )
    if checkpoint is not None:
        tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
    elif args.tokenizer_path is not None:
        tokenizer = load_tokenizer(args.tokenizer_path)
    elif args.tokenizer == "sentencepiece":
        tokenizer = SentencePieceTokenizer.train(
            (record.transcript for record in train_records),
            model_prefix=output_dir / "tokenizer",
            vocab_size=args.spm_vocab_size,
            model_type=args.spm_model_type,
        )
        tokenizer.save(output_dir / "tokenizer.model")
    else:
        tokenizer = CharacterTokenizer.build(record.transcript for record in train_records)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    logger.info(
        "tokenizer ready vocab_size=%s elapsed=%s",
        tokenizer.vocab_size,
        _format_elapsed_seconds(time.perf_counter() - stage_start_time),
    )
    if args.fit_shallow_fusion_lm:
        stage_start_time = time.perf_counter()
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
        shallow_fusion_lm_path = output_dir / "shallow_fusion_lm.json"
        shallow_fusion_lm.save(shallow_fusion_lm_path)
        if lm_scorer is None:
            lm_scorer = shallow_fusion_lm.score_extension
        logger.info(
            "shallow-fusion LM ready path=%s elapsed=%s",
            shallow_fusion_lm_path,
            _format_elapsed_seconds(time.perf_counter() - stage_start_time),
        )
    if hasattr(train_records, "close"):
        train_records.close()
    if hasattr(val_records, "close"):
        val_records.close()

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
    local_train_records = _shard_records_for_rank(
        train_records,
        rank=rank,
        world_size=world_size,
    )
    train_dataset = ASRDataset(
        local_train_records,
        tokenizer=tokenizer,
        featurizer=featurizer,
        specaugment=specaugment,
        waveform_augment=waveform_augment,
        feature_cache_dir=train_feature_cache_dir,
    )
    val_dataset = ASRDataset(
        val_records,
        tokenizer=tokenizer,
        featurizer=featurizer,
        feature_cache_dir=val_feature_cache_dir,
    )
    stage_start_time = time.perf_counter()
    logger.info(
        "building dataloaders train_shard_samples=%s val_samples=%s train_hours=%.2f val_hours=%.2f num_workers=%s metadata_workers=%s persistent_workers=%s prefetch_factor=%s",
        len(local_train_records),
        len(val_records),
        _record_store_duration_hours(local_train_records, hop_length=args.hop_length),
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
        max_batch_frames=args.max_batch_frames,
        adaptive_batch_unit=args.adaptive_batch_unit,
        adaptive_batch_budget=args.adaptive_batch_budget,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        metadata_workers=args.metadata_workers,
        longest_batches_first=args.longest_batches_first,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        bucket_by_length=args.bucket_by_length,
        max_batch_frames=args.max_batch_frames,
        adaptive_batch_unit=args.adaptive_batch_unit,
        adaptive_batch_budget=args.adaptive_batch_budget,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        metadata_workers=args.metadata_workers,
        longest_batches_first=False,
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
        encoder_config = replace(
            deepcopy(encoder_config),
            block_pattern=_resolve_block_pattern(args.block_pattern),
            activation_checkpointing=args.activation_checkpointing,
            attention_backend=args.attention_backend,
        )
    intermediate_ctc_layers, intermediate_ctc_weight = _resolve_intermediate_ctc_settings(
        args,
        encoder_config,
        checkpoint,
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
    args.blank_prune = blank_prune_layer is not None and blank_prune_threshold > 0.0
    args.blank_prune_layer = blank_prune_layer
    args.blank_prune_threshold = blank_prune_threshold
    args.blank_prune_min_keep_frames = blank_prune_min_keep_frames
    fp8_recipe = _build_fp8_recipe(args)
    if distributed and requested_device.type == "cuda":
        if requested_device.index not in {None, local_rank}:
            raise ValueError(
                f"--device {args.device} conflicts with LOCAL_RANK={local_rank}. "
                "Use --device cuda or the matching cuda:<local_rank>."
            )
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = requested_device
    if args.dtype == DTypeChoice.FP8:
        _validate_fp8_runtime(device, encoder_config)
    stage_start_time = time.perf_counter()
    logger.info(
        "building model variant=%s dtype=%s compile=%s intermediate_ctc_layers=%s aed=%s liberta=%s blank_prune_layer=%s",
        args.variant,
        args.dtype,
        args.compile,
        list(intermediate_ctc_layers),
        aed_decoder_enabled,
        liberta_distill_enabled,
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
        use_transformer_engine=args.dtype == DTypeChoice.FP8,
    )
    model.to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if distributed:
        forward_model: nn.Module = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    else:
        forward_model = torch.compile(model) if args.compile else model
    aux_model = model
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
        (len(train_loader) + args.gradient_accumulation_steps - 1)
        // args.gradient_accumulation_steps,
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
    liberta_teacher = (
        FrozenLibertaTeacher(
            str(Path(liberta_model_path).expanduser().resolve())
            if liberta_model_path is not None
            else liberta_model_name,
            device=device,
            dtype=_resolve_model_load_dtype(args.dtype),
            max_length=liberta_max_length,
        )
        if liberta_distill_enabled
        else None
    )
    logger.info(
        "model and auxiliaries ready params=%s elapsed=%s",
        sum(parameter.numel() for parameter in model.parameters()),
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
        trackio.init(
            project=args.trackio_project,
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
        logger.info(
            "trackio initialized project=%s trackio_dir=%s elapsed_since_start=%s",
            args.trackio_project,
            trackio_dir,
            _format_elapsed_seconds(time.perf_counter() - process_start_time),
        )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
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
        running_aed_loss = 0.0
        running_liberta_distill_loss = 0.0
        tune_effective_frames = 0
        tune_padded_frames = 0
        tune_target_tokens = 0
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            if batch is None:
                logger.warning("skipping empty training batch after dataset filtering")
                continue
            if (batch_index - 1) % args.gradient_accumulation_steps == 0:
                tune_effective_frames = 0
                tune_padded_frames = 0
                tune_target_tokens = 0
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

            with _autocast_context(device, args.dtype, fp8_recipe=fp8_recipe):
                encoded, output_lengths, intermediate_encoded, intermediate_output_lengths = (
                    aux_model.encode_with_intermediates(features, feature_lengths)
                )
                log_probs, intermediate_log_probs = aux_model.ctc_log_probs_from_encoded(
                    encoded,
                    intermediate_encoded,
                )
                main_ctc_loss = criterion(
                    log_probs.float().transpose(0, 1),
                    targets,
                    output_lengths,
                    target_lengths,
                )
                if intermediate_log_probs and intermediate_output_lengths:
                    intermediate_ctc_losses = [
                        criterion(
                            intermediate_log_probs[layer_index].float().transpose(0, 1),
                            targets,
                            intermediate_output_lengths[layer_index],
                            target_lengths,
                        )
                        for layer_index in intermediate_ctc_layers
                    ]
                    intermediate_ctc_loss = torch.stack(intermediate_ctc_losses).mean()
                    loss = (
                        1.0 - intermediate_ctc_weight
                    ) * main_ctc_loss + intermediate_ctc_weight * intermediate_ctc_loss
                else:
                    intermediate_ctc_loss = None
                    loss = main_ctc_loss
                if (
                    aed_decoder_enabled
                    and decoder_inputs is not None
                    and decoder_targets is not None
                ):
                    aed_logits, aed_hidden = aux_model.aed_from_encoded(
                        encoded,
                        output_lengths,
                        decoder_inputs,
                    )
                    aed_loss = _aed_cross_entropy_loss(
                        aed_logits,
                        decoder_targets,
                        pad_id=model.aed_decoder.pad_id,
                    )
                    loss = (1.0 - args.aed_loss_weight) * loss + args.aed_loss_weight * aed_loss
                else:
                    aed_loss = None
                    aed_hidden = None
            if (
                liberta_teacher is not None
                and aed_hidden is not None
                and decoder_target_lengths is not None
            ):
                teacher_embeddings = liberta_teacher.encode(batch["transcripts"])
                student_embeddings = aux_model.project_aed_hidden_for_liberta(
                    aed_hidden,
                    decoder_target_lengths,
                )
                liberta_distill_loss = F.mse_loss(
                    F.normalize(student_embeddings.float(), dim=-1),
                    F.normalize(teacher_embeddings.float(), dim=-1),
                )
                loss = loss + (args.liberta_distill_weight * liberta_distill_loss)
            else:
                liberta_distill_loss = None
            running_loss += float(loss.item())
            running_main_ctc_loss += float(main_ctc_loss.item())
            running_intermediate_ctc_loss += float(
                intermediate_ctc_loss.item() if intermediate_ctc_loss is not None else 0.0
            )
            running_aed_loss += float(aed_loss.item() if aed_loss is not None else 0.0)
            running_liberta_distill_loss += float(
                liberta_distill_loss.item() if liberta_distill_loss is not None else 0.0
            )
            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            should_step = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(
                train_loader
            )
            if should_step:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                grad_norm = float(_compute_grad_norm(model.parameters()).item())
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
                        total_batches=len(train_loader),
                        optimizer_step_index=global_step,
                        effective_frames=tune_effective_frames,
                        padded_frames=tune_padded_frames,
                        target_tokens=tune_target_tokens,
                        device=device,
                    )

                if is_main_process and global_step % args.log_every == 0:
                    learning_rates = {
                        f"learning_rate_{name}": optimizer.param_groups[0]["lr"]
                        for name, optimizer in zip(optimizer_names, optimizers, strict=True)
                    }
                    memory_snapshot = _format_memory_snapshot(device)
                    batch_audio_minutes = _frames_to_minutes(
                        tune_effective_frames,
                        hop_length=args.hop_length,
                    )
                    logger.info(
                        (
                            "epoch=%s step=%s/%s global_step=%s train_loss=%.4f "
                            "train_main_ctc_loss=%.4f train_intermediate_ctc_loss=%.4f "
                            "train_aed_loss=%.4f train_liberta_distill_loss=%.4f "
                            "batch_audio_minutes=%.2f grad_norm=%.4f %s %s"
                        ),
                        epoch,
                        batch_index,
                        len(train_loader),
                        global_step,
                        float(loss.item()),
                        float(main_ctc_loss.item()),
                        float(
                            intermediate_ctc_loss.item()
                            if intermediate_ctc_loss is not None
                            else 0.0
                        ),
                        float(aed_loss.item() if aed_loss is not None else 0.0),
                        float(
                            liberta_distill_loss.item() if liberta_distill_loss is not None else 0.0
                        ),
                        batch_audio_minutes,
                        grad_norm,
                        memory_snapshot,
                        " ".join(f"{name}={value:.6g}" for name, value in learning_rates.items()),
                    )
                    trackio.log(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss_step": float(loss.item()),
                            "train_main_ctc_loss_step": float(main_ctc_loss.item()),
                            "train_intermediate_ctc_loss_step": float(
                                intermediate_ctc_loss.item()
                                if intermediate_ctc_loss is not None
                                else 0.0
                            ),
                            "train_aed_loss_step": float(
                                aed_loss.item() if aed_loss is not None else 0.0
                            ),
                            "train_liberta_distill_loss_step": float(
                                liberta_distill_loss.item()
                                if liberta_distill_loss is not None
                                else 0.0
                            ),
                            "grad_norm": grad_norm,
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
                        }
                    )

                if (
                    is_main_process
                    and args.validate_every_steps > 0
                    and global_step % args.validate_every_steps == 0
                ):
                    train_metrics = {
                        "train_loss": running_loss / max(1, batch_index),
                        "train_main_ctc_loss": running_main_ctc_loss / max(1, batch_index),
                        "train_intermediate_ctc_loss": (
                            running_intermediate_ctc_loss / max(1, batch_index)
                        ),
                        "train_aed_loss": running_aed_loss / max(1, batch_index),
                        "train_liberta_distill_loss": (
                            running_liberta_distill_loss / max(1, batch_index)
                        ),
                    }
                    logger.info(
                        (
                            "epoch %s step=%s/%s global_step=%s reached validation interval; "
                            "starting validation"
                        ),
                        epoch,
                        batch_index,
                        len(train_loader),
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
                        example_limit=args.example_limit,
                        intermediate_ctc_weight=intermediate_ctc_weight,
                        aed_loss_weight=args.aed_loss_weight,
                        liberta_teacher=liberta_teacher,
                        liberta_distill_weight=args.liberta_distill_weight,
                        ema=ema,
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
                    )

        train_loss = running_loss / max(1, len(train_loader))
        train_main_ctc_loss = running_main_ctc_loss / max(1, len(train_loader))
        train_intermediate_ctc_loss = running_intermediate_ctc_loss / max(1, len(train_loader))
        train_aed_loss = running_aed_loss / max(1, len(train_loader))
        train_liberta_distill_loss = running_liberta_distill_loss / max(1, len(train_loader))
        if is_main_process:
            logger.info(
                "epoch %s training complete train_loss=%.4f train_main_ctc_loss=%.4f train_intermediate_ctc_loss=%.4f train_aed_loss=%.4f train_liberta_distill_loss=%.4f elapsed=%s, starting validation",
                epoch,
                train_loss,
                train_main_ctc_loss,
                train_intermediate_ctc_loss,
                train_aed_loss,
                train_liberta_distill_loss,
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
                example_limit=args.example_limit,
                intermediate_ctc_weight=intermediate_ctc_weight,
                aed_loss_weight=args.aed_loss_weight,
                liberta_teacher=liberta_teacher,
                liberta_distill_weight=args.liberta_distill_weight,
                ema=ema,
                train_metrics={
                    "train_loss": train_loss,
                    "train_main_ctc_loss": train_main_ctc_loss,
                    "train_intermediate_ctc_loss": train_intermediate_ctc_loss,
                    "train_aed_loss": train_aed_loss,
                    "train_liberta_distill_loss": train_liberta_distill_loss,
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
            )
        if distributed:
            dist.barrier()

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
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
