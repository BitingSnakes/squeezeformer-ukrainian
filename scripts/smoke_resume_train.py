from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import torch


def _write_wave(path: Path, frequency: float, duration_seconds: float = 0.3) -> None:
    sample_rate = 16_000
    steps = int(sample_rate * duration_seconds)
    timeline = torch.linspace(0, duration_seconds, steps=steps)
    waveform = 0.2 * torch.sin(2 * math.pi * frequency * timeline)
    pcm = (waveform.clamp(-1.0, 1.0) * 32767).to(torch.int16).tolist()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(torch.tensor(pcm, dtype=torch.int16).numpy().tobytes())


def _build_local_dataset(root: Path) -> Path:
    dataset_dir = root / "dataset"
    clips_dir = dataset_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rows = ["path\tsentence\tclient_id\tduration"]
    transcripts = [
        "це тест",
        "це другий тест",
        "мовна модель",
        "глибоке навчання",
        "розпізнавання мовлення",
        "короткий приклад",
    ]

    utterance_index = 0
    for speaker_index in range(12):
        for variant_index in range(2):
            clip_name = f"utt_{utterance_index:03d}.wav"
            transcript = transcripts[(speaker_index + variant_index) % len(transcripts)]
            clip_path = clips_dir / clip_name
            _write_wave(clip_path, frequency=220 + 15 * utterance_index)
            rows.append(
                f"clips/{clip_name}\t{transcript}\tspeaker_{speaker_index:02d}\t0.3"
            )
            utterance_index += 1

    (dataset_dir / "train.tsv").write_text("\n".join(rows), encoding="utf-8")
    return dataset_dir


def _run_train(command: list[str], workdir: Path) -> None:
    subprocess.run(command, cwd=workdir, check=True)


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        dataset_dir = _build_local_dataset(temp_dir)
        output_dir = temp_dir / "artifacts"

        base_command = [
            sys.executable,
            str(workspace / "train.py"),
            "--dataset-repo",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--variant",
            "xs",
            "--tokenizer",
            "character",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--val-fraction",
            "0.2",
            "--test-fraction",
            "0.2",
            "--max-train-samples",
            "12",
            "--max-val-samples",
            "6",
            "--trackio-project",
            "squeezeformer-smoke",
            "--gradient-accumulation-steps",
            "2",
            "--fit-shallow-fusion-lm",
            "--decode-strategy",
            "beam",
            "--beam-size",
            "4",
            "--lm-weight",
            "0.1",
            "--keep-top-k",
            "2",
        ]

        first_command = base_command + ["--epochs", "1"]
        _run_train(first_command, workdir=workspace)
        first_checkpoint = torch.load(
            output_dir / "checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )
        if first_checkpoint["epoch"] != 1:
            raise RuntimeError("first smoke run did not finish epoch 1")
        first_global_step = int(first_checkpoint.get("global_step", 0))
        if not (output_dir / "shallow_fusion_lm.json").exists():
            raise FileNotFoundError("shallow fusion LM artifact was not written")

        second_command = base_command + [
            "--epochs",
            "2",
            "--resume",
            str(output_dir / "checkpoint_last.pt"),
        ]
        _run_train(second_command, workdir=workspace)
        second_checkpoint = torch.load(
            output_dir / "checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )
        if second_checkpoint["epoch"] != 2:
            raise RuntimeError("resume smoke run did not finish epoch 2")
        if int(second_checkpoint.get("global_step", 0)) <= first_global_step:
            raise RuntimeError("resume smoke run did not advance the global step")

        summary = {
            "output_dir": str(output_dir),
            "epoch_after_resume": int(second_checkpoint["epoch"]),
            "global_step_after_resume": int(second_checkpoint["global_step"]),
            "best_val_wer": float(second_checkpoint["best_val_wer"]),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
