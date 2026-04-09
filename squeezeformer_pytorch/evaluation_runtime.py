from __future__ import annotations

from typing import Any


def resolve_evaluation_checkpoint_settings(checkpoint: dict[str, Any]) -> dict[str, Any]:
    training_args = checkpoint.get("training_args", {})
    intermediate_ctc_weight = float(training_args.get("intermediate_ctc_weight", 0.0))
    intermediate_ctc_layers = training_args.get("intermediate_ctc_layers")
    intermediate_ctc_layer = training_args.get("intermediate_ctc_layer")

    if intermediate_ctc_weight > 0.0:
        if intermediate_ctc_layers is not None:
            resolved_intermediate_ctc_layers = tuple(
                int(layer) for layer in intermediate_ctc_layers
            )
        elif intermediate_ctc_layer is not None:
            resolved_intermediate_ctc_layers = (int(intermediate_ctc_layer),)
        else:
            resolved_intermediate_ctc_layers = ()
    else:
        resolved_intermediate_ctc_layers = ()

    blank_prune_threshold = float(training_args.get("blank_prune_threshold", 0.0))
    blank_prune_layer = training_args.get("blank_prune_layer")

    return {
        "intermediate_ctc_weight": intermediate_ctc_weight,
        "resolved_intermediate_ctc_layers": resolved_intermediate_ctc_layers,
        "blank_prune_threshold": blank_prune_threshold,
        "blank_prune_layer": (
            int(blank_prune_layer)
            if blank_prune_threshold > 0.0 and blank_prune_layer is not None
            else None
        ),
        "blank_prune_min_keep_frames": int(training_args.get("blank_prune_min_keep_frames", 1)),
        "initial_ctc_blank_bias": float(training_args.get("initial_ctc_blank_bias", 0.0)),
        "identical_initial_ctc_heads": bool(
            training_args.get("identical_initial_ctc_heads", False)
        ),
        "aed_decoder_enabled": bool(training_args.get("aed_decoder", False)),
        "aed_decoder_layers": int(training_args.get("aed_decoder_layers", 1)),
        "aed_decoder_heads": int(training_args.get("aed_decoder_heads", 4)),
        "aed_decoder_dropout": float(training_args.get("aed_decoder_dropout", 0.1)),
        "aed_loss_weight": float(training_args.get("aed_loss_weight", 0.3)),
        "liberta_distill_enabled": bool(training_args.get("liberta_distill", False)),
        "liberta_model_name": str(training_args.get("liberta_model_name", "Yehor/Liberta")),
        "liberta_distill_weight": float(training_args.get("liberta_distill_weight", 0.05)),
        "liberta_max_length": int(training_args.get("liberta_max_length", 256)),
        "audio_teacher_enabled": bool(training_args.get("audio_teacher", False)),
        "audio_teacher_model_name": str(
            training_args.get("audio_teacher_model_name", "facebook/wav2vec2-bert-2.0")
        ),
        "audio_teacher_weight": float(training_args.get("audio_teacher_weight", 0.05)),
        "audio_teacher_objective": str(training_args.get("audio_teacher_objective", "hidden_mse")),
        "audio_teacher_target": str(training_args.get("audio_teacher_target", "encoder")),
        "audio_teacher_layer": int(training_args.get("audio_teacher_layer", -1)),
        "audio_teacher_sample_rate": int(training_args.get("audio_teacher_sample_rate", 16000)),
        "audio_teacher_max_seconds": float(training_args.get("audio_teacher_max_seconds", 30.0)),
    }


def resolve_lowercase_transcripts(
    override: bool | None,
    *,
    checkpoint_tokenizer_type: str,
) -> bool:
    if override is not None:
        return override
    return checkpoint_tokenizer_type != "sentencepiece"


def build_evaluation_payload(
    *,
    metrics: dict[str, object],
    result: dict[str, object],
    split: str,
    samples: int,
    decode_strategy: object,
) -> dict[str, object]:
    enriched_metrics = metrics | {
        "split": split,
        "samples": samples,
        "decode_strategy": decode_strategy,
    }
    return {
        "metrics": enriched_metrics,
        "hardest_examples": result["hardest_examples"],
        "random_examples": result["random_examples"],
        "speaker_metrics": result["speaker_metrics"],
    }
