from __future__ import annotations

import argparse
import tempfile
from contextlib import ExitStack, nullcontext
from pathlib import Path

import gradio as gr
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from squeezeformer_pytorch.asr import SqueezeformerCTC, tokenizer_from_dict
from squeezeformer_pytorch.checkpoints import load_checkpoint
from squeezeformer_pytorch.frontend import AudioFeaturizer
from squeezeformer_pytorch.model import (
    FP8_SHAPE_ALIGNMENT,
    SqueezeformerConfig,
    transformer_engine_available,
)
from squeezeformer_pytorch.runtime_types import DTypeChoice

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
except ImportError:
    te = None
    DelayedScaling = None
    Format = None

DEFAULT_CHECKPOINT = (
    "https://huggingface.co/speech-uk/squeezeformer-sm/resolve/main/checkpoint_best.pt"
)


class ASRInferenceSession:
    def __init__(
        self,
        checkpoint: str,
        device: torch.device,
        dtype: DTypeChoice,
        *,
        fp8_recipe=None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.fp8_recipe = fp8_recipe
        self.checkpoint_path = resolve_checkpoint_path(checkpoint)

        checkpoint_data = load_checkpoint(self.checkpoint_path, map_location="cpu")
        self.tokenizer = tokenizer_from_dict(checkpoint_data["tokenizer"])
        encoder_config = SqueezeformerConfig(**checkpoint_data["encoder_config"])
        training_args = checkpoint_data.get("training_args", {})
        checkpoint_dtype = str(training_args.get("dtype", ""))
        intermediate_ctc_weight = float(training_args.get("intermediate_ctc_weight", 0.0))
        intermediate_ctc_layer = training_args.get("intermediate_ctc_layer")
        use_transformer_engine = checkpoint_dtype == "fp8" or dtype == DTypeChoice.FP8
        if dtype == DTypeChoice.FP8:
            validate_fp8_inference_runtime(device, encoder_config)

        self.model = SqueezeformerCTC(
            encoder_config=encoder_config,
            vocab_size=self.tokenizer.vocab_size,
            intermediate_ctc_layer=(
                int(intermediate_ctc_layer)
                if intermediate_ctc_weight > 0.0 and intermediate_ctc_layer is not None
                else None
            ),
            use_transformer_engine=use_transformer_engine,
        )
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self.featurizer = AudioFeaturizer(**checkpoint_data.get("featurizer_config", {}))

    def transcribe_file(self, audio_path: str | Path) -> str:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        return self.transcribe_waveform(waveform, sample_rate)

    def transcribe_waveform(self, waveform: torch.Tensor, sample_rate: int) -> str:
        features = self.featurizer(waveform, sample_rate).unsqueeze(0).to(self.device)
        feature_lengths = torch.tensor([features.size(1)], device=self.device)

        with (
            torch.inference_mode(),
            inference_autocast_context(
                self.device,
                self.dtype,
                fp8_recipe=self.fp8_recipe,
            ),
        ):
            log_probs, _ = self.model.log_probs(features, feature_lengths)

        token_ids = log_probs.argmax(dim=-1)[0].cpu().tolist()
        return self.tokenizer.decode_ctc(token_ids)


def resolve_checkpoint_path(checkpoint: str) -> str:
    if checkpoint == DEFAULT_CHECKPOINT:
        return hf_hub_download(repo_id="speech-uk/squeezeformer-sm", filename="checkpoint_best.pt")
    if checkpoint.startswith("https://huggingface.co/"):
        parts = checkpoint.removeprefix("https://huggingface.co/").split("/")
        if len(parts) >= 5 and parts[2] == "resolve":
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[4:])
            return hf_hub_download(repo_id=repo_id, filename=filename)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASR inference or launch a small Gradio UI.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path or Hugging Face URL.",
    )
    parser.add_argument("--audio", help="Path to an audio file to transcribe.")
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch the Gradio app instead of transcribing one file and exiting.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device, for example 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--dtype",
        type=DTypeChoice,
        choices=list(DTypeChoice),
        default=DTypeChoice.BFLOAT16 if torch.cuda.is_available() else DTypeChoice.FLOAT32,
        help="Inference autocast dtype.",
    )
    parser.add_argument(
        "--fp8-format",
        default="hybrid",
        choices=["hybrid", "e4m3"],
        help="Transformer Engine FP8 format when --dtype fp8 is used.",
    )
    parser.add_argument(
        "--fp8-amax-history-len",
        type=int,
        default=16,
        help="Transformer Engine FP8 amax history length.",
    )
    parser.add_argument(
        "--fp8-amax-compute-algo",
        default="max",
        choices=["max", "most_recent"],
        help="Transformer Engine FP8 amax reduction algorithm.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def resolve_fp8_format(name: str):
    if Format is None:
        raise RuntimeError("transformer-engine is required for FP8 inference.")
    normalized = name.strip().lower()
    if normalized == "hybrid":
        return Format.HYBRID
    if normalized == "e4m3":
        return Format.E4M3
    raise ValueError(f"Unsupported FP8 format: {name}")


def build_fp8_recipe(args: argparse.Namespace):
    if args.dtype != DTypeChoice.FP8:
        return None
    if DelayedScaling is None:
        raise RuntimeError("transformer-engine is required for FP8 inference.")
    return DelayedScaling(
        fp8_format=resolve_fp8_format(args.fp8_format),
        amax_history_len=args.fp8_amax_history_len,
        amax_compute_algo=args.fp8_amax_compute_algo,
    )


def validate_fp8_inference_runtime(
    device: torch.device, encoder_config: SqueezeformerConfig
) -> None:
    if device.type != "cuda":
        raise ValueError("FP8 inference requires a CUDA device.")
    if not transformer_engine_available() or te is None:
        raise RuntimeError(
            "FP8 inference requires transformer-engine. Install the package and CUDA extension."
        )
    if encoder_config.d_model % FP8_SHAPE_ALIGNMENT != 0:
        raise ValueError(
            "FP8 inference requires d_model to be divisible by "
            f"{FP8_SHAPE_ALIGNMENT}; choose variant xs, sm, ml, or l."
        )
    if hasattr(te, "is_fp8_available"):
        availability = te.is_fp8_available()
        if isinstance(availability, tuple):
            is_available, reason = availability
        else:
            is_available, reason = bool(availability), None
        if not is_available:
            suffix = f" {reason}" if reason else ""
            raise RuntimeError(
                f"Transformer Engine reports FP8 is unavailable on this runtime.{suffix}"
            )


def inference_autocast_context(device: torch.device, dtype: DTypeChoice, *, fp8_recipe=None):
    if dtype == DTypeChoice.FLOAT32:
        return nullcontext()
    if dtype == DTypeChoice.FLOAT16:
        if device.type == "cpu":
            raise ValueError("float16 inference is not supported on CPU. Use bfloat16 or float32.")
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if dtype == DTypeChoice.BFLOAT16:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if dtype == DTypeChoice.FP8:
        if device.type != "cuda":
            raise ValueError("FP8 inference requires a CUDA device.")
        if te is None:
            raise RuntimeError("transformer-engine is required for FP8 inference.")
        stack = ExitStack()
        stack.enter_context(torch.autocast(device_type=device.type, dtype=torch.bfloat16))
        stack.enter_context(te.autocast(enabled=True, recipe=fp8_recipe))
        return stack
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_app(session: ASRInferenceSession) -> gr.Blocks:
    current_session = {"value": session}

    def load_session_for_checkpoint(checkpoint: str) -> str:
        checkpoint = checkpoint.strip() or DEFAULT_CHECKPOINT
        current_session["value"] = ASRInferenceSession(
            checkpoint,
            session.device,
            session.dtype,
            fp8_recipe=session.fp8_recipe,
        )
        return f"Loaded checkpoint: {current_session['value'].checkpoint_path}"

    def transcribe_for_gradio(audio: str | tuple[int, list[float]] | None) -> str:
        if audio is None:
            return "Upload or record audio first."

        if isinstance(audio, str):
            return current_session["value"].transcribe_file(audio)

        sample_rate, samples = audio
        waveform = torch.tensor(samples, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.transpose(0, 1)

        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            torchaudio.save(temp_audio.name, waveform, sample_rate)
            return current_session["value"].transcribe_file(temp_audio.name)

    with gr.Blocks(title="Ukrainian Squeezeformer ASR") as app:
        gr.Markdown("# Ukrainian Squeezeformer ASR")
        checkpoint_input = gr.Textbox(
            value=session.checkpoint_path,
            label="Checkpoint",
            lines=1,
        )
        checkpoint_status = gr.Textbox(
            value=f"Loaded checkpoint: {session.checkpoint_path}",
            label="Checkpoint Status",
            interactive=False,
        )
        load_checkpoint_button = gr.Button("Load Checkpoint")
        audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
        output = gr.Textbox(label="Transcript", lines=4)
        submit = gr.Button("Transcribe")
        load_checkpoint_button.click(
            fn=load_session_for_checkpoint,
            inputs=checkpoint_input,
            outputs=checkpoint_status,
        )
        submit.click(fn=transcribe_for_gradio, inputs=audio_input, outputs=output)
        audio_input.change(fn=transcribe_for_gradio, inputs=audio_input, outputs=output)
    return app


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    fp8_recipe = build_fp8_recipe(args)
    session = ASRInferenceSession(args.checkpoint, device, args.dtype, fp8_recipe=fp8_recipe)

    if args.gradio:
        app = build_app(session)
        app.launch(server_name=args.host, server_port=args.port, share=args.share)
        return

    if not args.audio:
        raise ValueError("--audio is required unless --gradio is used.")

    print(session.transcribe_file(args.audio))


if __name__ == "__main__":
    main()
