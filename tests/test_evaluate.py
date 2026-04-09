from __future__ import annotations

from dataclasses import asdict

import evaluate
from squeezeformer_pytorch import squeezeformer_variant
from squeezeformer_pytorch.evaluation_runtime import resolve_evaluation_checkpoint_settings
from squeezeformer_pytorch.runtime_types import DTypeChoice


def _build_checkpoint(tokenizer_type: str = "character") -> dict[str, object]:
    return {
        "tokenizer": {"type": tokenizer_type, "symbols": ["а"]},
        "encoder_config": asdict(squeezeformer_variant("xs")),
        "training_args": {},
        "model_state_dict": {},
        "featurizer_config": {},
    }


def test_evaluate_defaults_to_checkpoint_tokenizer_casing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 32
        blank_id = 0

    class DummyModel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def load_state_dict(self, _state_dict):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

    class DummyFeaturizer:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class DummyDataset:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    def fake_load_records(**kwargs):
        captured["lowercase_transcripts"] = kwargs["lowercase_transcripts"]
        return []

    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "checkpoint": "checkpoint.safetensors",
                "dataset_repo": "repo",
                "dataset_source": None,
                "validation_dataset_source": None,
                "hf_token": None,
                "cache_dir": None,
                "split": "test",
                "batch_size": 1,
                "num_workers": 0,
                "seed": 13,
                "val_fraction": 0.1,
                "test_fraction": 0.1,
                "max_samples": None,
                "feature_cache_dir": None,
                "lowercase_transcripts": None,
                "bucket_by_length": True,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "prevalidate_audio": False,
                "prevalidate_workers": 1,
                "device": "cpu",
                "dtype": DTypeChoice.FLOAT32,
                "decode_strategy": "greedy",
                "beam_size": 8,
                "lm_scorer": None,
                "lm_weight": 0.0,
                "example_limit": 1,
                "report_path": None,
                "trackio_project": "test",
                "trackio_space_id": None,
            },
        )(),
    )
    monkeypatch.setattr(
        evaluate,
        "load_checkpoint",
        lambda *_args, **_kwargs: _build_checkpoint(tokenizer_type="sentencepiece"),
    )
    monkeypatch.setattr(evaluate, "tokenizer_from_dict", lambda *_args, **_kwargs: DummyTokenizer())
    monkeypatch.setattr(evaluate, "SqueezeformerCTC", DummyModel)
    monkeypatch.setattr(evaluate, "resolve_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(evaluate, "_validate_device_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluate, "_load_records_from_dataset_roots", lambda *_args, **kwargs: fake_load_records(**kwargs))
    monkeypatch.setattr(evaluate, "AudioFeaturizer", DummyFeaturizer)
    monkeypatch.setattr(evaluate, "ASRDataset", DummyDataset)
    monkeypatch.setattr(evaluate, "create_dataloader", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(evaluate, "load_lm_scorer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evaluate,
        "evaluate",
        lambda **_kwargs: {
            "metrics": {},
            "hardest_examples": [],
            "random_examples": [],
            "speaker_metrics": {},
        },
    )
    monkeypatch.setattr(evaluate.trackio, "init", lambda **_kwargs: None)
    monkeypatch.setattr(evaluate.trackio, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluate.trackio, "finish", lambda: None)

    evaluate.main()

    assert captured["lowercase_transcripts"] is False


def test_evaluate_respects_lowercase_transcripts_override(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 32
        blank_id = 0

    class DummyModel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def load_state_dict(self, _state_dict):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

    class DummyFeaturizer:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class DummyDataset:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    def fake_load_records(**kwargs):
        captured["lowercase_transcripts"] = kwargs["lowercase_transcripts"]
        return []

    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "checkpoint": "checkpoint.safetensors",
                "dataset_repo": "repo",
                "dataset_source": None,
                "validation_dataset_source": None,
                "hf_token": None,
                "cache_dir": None,
                "split": "test",
                "batch_size": 1,
                "num_workers": 0,
                "seed": 13,
                "val_fraction": 0.1,
                "test_fraction": 0.1,
                "max_samples": None,
                "feature_cache_dir": None,
                "lowercase_transcripts": False,
                "bucket_by_length": True,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "prevalidate_audio": False,
                "prevalidate_workers": 1,
                "device": "cpu",
                "dtype": DTypeChoice.FLOAT32,
                "decode_strategy": "greedy",
                "beam_size": 8,
                "lm_scorer": None,
                "lm_weight": 0.0,
                "example_limit": 1,
                "report_path": None,
                "trackio_project": "test",
                "trackio_space_id": None,
            },
        )(),
    )
    monkeypatch.setattr(
        evaluate,
        "load_checkpoint",
        lambda *_args, **_kwargs: _build_checkpoint(tokenizer_type="character"),
    )
    monkeypatch.setattr(evaluate, "tokenizer_from_dict", lambda *_args, **_kwargs: DummyTokenizer())
    monkeypatch.setattr(evaluate, "SqueezeformerCTC", DummyModel)
    monkeypatch.setattr(evaluate, "resolve_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(evaluate, "_validate_device_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluate, "_load_records_from_dataset_roots", lambda *_args, **kwargs: fake_load_records(**kwargs))
    monkeypatch.setattr(evaluate, "AudioFeaturizer", DummyFeaturizer)
    monkeypatch.setattr(evaluate, "ASRDataset", DummyDataset)
    monkeypatch.setattr(evaluate, "create_dataloader", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(evaluate, "load_lm_scorer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evaluate,
        "evaluate",
        lambda **_kwargs: {
            "metrics": {},
            "hardest_examples": [],
            "random_examples": [],
            "speaker_metrics": {},
        },
    )
    monkeypatch.setattr(evaluate.trackio, "init", lambda **_kwargs: None)
    monkeypatch.setattr(evaluate.trackio, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluate.trackio, "finish", lambda: None)

    evaluate.main()

    assert captured["lowercase_transcripts"] is False


def test_evaluate_prefers_validation_dataset_sources_when_dataset_source_is_unset(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyTokenizer:
        vocab_size = 32
        blank_id = 0

    class DummyModel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def load_state_dict(self, _state_dict):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

    class DummyFeaturizer:
        def __init__(self, **_kwargs: object) -> None:
            pass

    class DummyDataset:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    def fake_load_records_from_dataset_roots(
        dataset_sources,
        **kwargs,
    ):
        captured["dataset_sources"] = dataset_sources
        captured["split"] = kwargs["split"]
        return []

    monkeypatch.setattr(
        evaluate,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "checkpoint": "checkpoint.safetensors",
                "dataset_repo": "speech-uk/cv22",
                "dataset_source": None,
                "validation_dataset_source": ["/tmp/local-eval-set"],
                "hf_token": None,
                "cache_dir": None,
                "split": "test",
                "batch_size": 1,
                "num_workers": 0,
                "seed": 13,
                "val_fraction": 0.1,
                "test_fraction": 0.1,
                "max_samples": None,
                "feature_cache_dir": None,
                "lowercase_transcripts": None,
                "bucket_by_length": True,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "prevalidate_audio": False,
                "prevalidate_workers": 1,
                "device": "cpu",
                "dtype": DTypeChoice.FLOAT32,
                "decode_strategy": "greedy",
                "beam_size": 8,
                "lm_scorer": None,
                "lm_weight": 0.0,
                "example_limit": 1,
                "report_path": None,
                "trackio_project": "test",
                "trackio_space_id": None,
            },
        )(),
    )
    monkeypatch.setattr(
        evaluate,
        "load_checkpoint",
        lambda *_args, **_kwargs: _build_checkpoint(tokenizer_type="character"),
    )
    monkeypatch.setattr(evaluate, "tokenizer_from_dict", lambda *_args, **_kwargs: DummyTokenizer())
    monkeypatch.setattr(evaluate, "SqueezeformerCTC", DummyModel)
    monkeypatch.setattr(evaluate, "resolve_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(evaluate, "_validate_device_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evaluate,
        "_resolve_sources",
        lambda sources, fallback=None: list(sources or ([fallback] if fallback is not None else [])),
    )
    monkeypatch.setattr(
        evaluate,
        "_load_records_from_dataset_roots",
        fake_load_records_from_dataset_roots,
    )
    monkeypatch.setattr(evaluate, "AudioFeaturizer", DummyFeaturizer)
    monkeypatch.setattr(evaluate, "ASRDataset", DummyDataset)
    monkeypatch.setattr(evaluate, "create_dataloader", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(evaluate, "load_lm_scorer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evaluate,
        "evaluate",
        lambda **_kwargs: {
            "metrics": {},
            "hardest_examples": [],
            "random_examples": [],
            "speaker_metrics": {},
        },
    )
    monkeypatch.setattr(evaluate.trackio, "init", lambda **_kwargs: None)
    monkeypatch.setattr(evaluate.trackio, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluate.trackio, "finish", lambda: None)

    evaluate.main()

    assert captured["dataset_sources"] == ["/tmp/local-eval-set"]
    assert captured["split"] == "test"


def test_evaluation_runtime_resolves_audio_teacher_metadata() -> None:
    settings = resolve_evaluation_checkpoint_settings(
        {
            "training_args": {
                "audio_teacher": True,
                "audio_teacher_model_name": "facebook/wav2vec2-bert-2.0",
                "audio_teacher_weight": 0.2,
                "audio_teacher_objective": "hidden_cosine",
                "audio_teacher_target": "encoder",
                "audio_teacher_layer": 6,
                "audio_teacher_sample_rate": 16_000,
                "audio_teacher_max_seconds": 12.5,
            }
        }
    )

    assert settings["audio_teacher_enabled"] is True
    assert settings["audio_teacher_model_name"] == "facebook/wav2vec2-bert-2.0"
    assert settings["audio_teacher_weight"] == 0.2
    assert settings["audio_teacher_objective"] == "hidden_cosine"
    assert settings["audio_teacher_target"] == "encoder"
    assert settings["audio_teacher_layer"] == 6
    assert settings["audio_teacher_sample_rate"] == 16_000
    assert settings["audio_teacher_max_seconds"] == 12.5
