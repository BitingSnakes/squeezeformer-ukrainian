from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from squeezeformer_pytorch.asr import SqueezeformerCTC, tokenizer_from_dict
from squeezeformer_pytorch.checkpoints import load_checkpoint
from squeezeformer_pytorch.data import AudioFeaturizer
from squeezeformer_pytorch.model import SqueezeformerConfig

DEFAULT_CHECKPOINT = "https://huggingface.co/speech-uk/squeezeformer-sm/resolve/main/checkpoint_best.pt"


class ASRInferenceSession:
    def __init__(self, checkpoint: str, device: torch.device) -> None:
        self.device = device
        self.checkpoint_path = resolve_checkpoint_path(checkpoint)

        checkpoint_data = load_checkpoint(self.checkpoint_path, map_location="cpu")
        self.tokenizer = tokenizer_from_dict(checkpoint_data["tokenizer"])
        encoder_config = SqueezeformerConfig(**checkpoint_data["encoder_config"])
        training_args = checkpoint_data.get("training_args", {})
        checkpoint_dtype = str(training_args.get("dtype", ""))

        self.model = SqueezeformerCTC(
            encoder_config=encoder_config,
            vocab_size=self.tokenizer.vocab_size,
            use_transformer_engine=checkpoint_dtype == "fp8",
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

        with torch.inference_mode():
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
    parser.add_argument("--host", default="127.0.0.1", help="Host for the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device, for example 'cpu' or 'cuda:0'.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def build_app(session: ASRInferenceSession) -> gr.Blocks:
    def transcribe_for_gradio(audio: str | tuple[int, list[float]] | None) -> str:
        if audio is None:
            return "Upload or record audio first."

        if isinstance(audio, str):
            return session.transcribe_file(audio)

        sample_rate, samples = audio
        waveform = torch.tensor(samples, dtype=torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.transpose(0, 1)

        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            torchaudio.save(temp_audio.name, waveform, sample_rate)
            return session.transcribe_file(temp_audio.name)

    with gr.Blocks(title="Ukrainian Squeezeformer ASR") as app:
        gr.Markdown("# Ukrainian Squeezeformer ASR")
        gr.Markdown(f"Checkpoint: `{session.checkpoint_path}`")
        audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
        output = gr.Textbox(label="Transcript", lines=4)
        submit = gr.Button("Transcribe")
        submit.click(fn=transcribe_for_gradio, inputs=audio_input, outputs=output)
        audio_input.change(fn=transcribe_for_gradio, inputs=audio_input, outputs=output)
    return app


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    session = ASRInferenceSession(args.checkpoint, device)

    if args.gradio:
        app = build_app(session)
        app.launch(server_name=args.host, server_port=args.port)
        return

    if not args.audio:
        raise ValueError("--audio is required unless --gradio is used.")

    print(session.transcribe_file(args.audio))


if __name__ == "__main__":
    main()
