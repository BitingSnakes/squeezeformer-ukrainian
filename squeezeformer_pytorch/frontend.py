from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F

try:
    import audioflux as af
    from audioflux.type import SpectralDataType, SpectralFilterBankScaleType
except ImportError:
    af = None
    SpectralDataType = None
    SpectralFilterBankScaleType = None


ZIPFORMER_PAPER_FEATURIZER_CONFIG: dict[str, object] = {
    "sample_rate": 16_000,
    "n_fft": 400,
    "win_length": 400,
    "hop_length": 160,
    "n_mels": 80,
    "backend": "torchaudio",
    "preemphasis": 0.0,
    "normalize_signal": False,
    "normalize_feature": False,
    "normalize_per_frame": False,
}


def estimate_num_feature_frames(
    num_samples: int,
    *,
    sample_rate: int = 16_000,
    target_sample_rate: int = 16_000,
    n_fft: int = 400,
    win_length: int | None = None,
    hop_length: int = 160,
    backend: str = "torchaudio",
) -> int:
    if num_samples <= 0 or sample_rate <= 0 or target_sample_rate <= 0:
        return 0
    if hop_length <= 0:
        raise ValueError(f"hop_length must be > 0, got {hop_length}.")
    effective_samples = int(num_samples)
    if sample_rate != target_sample_rate:
        effective_samples = max(
            1,
            int(math.ceil((effective_samples * target_sample_rate) / sample_rate)),
        )
    analysis_window = max(int(n_fft), int(win_length if win_length is not None else n_fft))
    effective_samples = max(effective_samples, analysis_window)
    if backend == "torchaudio":
        # torchaudio's MelSpectrogram uses centered STFT by default, adding n_fft // 2
        # samples of padding on both sides. With no additional explicit padding this gives
        # floor(num_samples / hop_length) + 1 frames after our minimum-length padding.
        return max(1, (effective_samples // hop_length) + 1)
    return max(1, int(effective_samples / hop_length))


def zipformer_paper_featurizer_config(
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    config = dict(ZIPFORMER_PAPER_FEATURIZER_CONFIG)
    if overrides is not None:
        config.update(overrides)
    return config


def resolve_checkpoint_featurizer_config(
    featurizer_config: Mapping[str, object] | None,
    *,
    use_zipformer: bool,
    use_w2v_bert: bool = False,
) -> dict[str, object]:
    if featurizer_config:
        return dict(featurizer_config)
    if use_w2v_bert:
        return {"type": "w2v_bert", "model_source": "facebook/w2v-bert-2.0"}
    if use_zipformer:
        return zipformer_paper_featurizer_config()
    return {}


def build_featurizer_from_config(
    featurizer_config: Mapping[str, object] | None,
    *,
    use_zipformer: bool = False,
    use_w2v_bert: bool = False,
):
    config = resolve_checkpoint_featurizer_config(
        featurizer_config,
        use_zipformer=use_zipformer,
        use_w2v_bert=use_w2v_bert,
    )
    if str(config.get("type", "")) == "w2v_bert":
        return RustW2VBertFeatureExtractor.from_config(config)
    frontend_type = "zipformer" if use_zipformer else "squeezeformer"
    return RustAudioFeaturizer(frontend_type=frontend_type, **config)


def _waveform_to_numpy(waveform: Tensor) -> np.ndarray:
    return waveform.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def _numpy_features_to_tensor(features: object) -> Tensor:
    return torch.from_numpy(np.asarray(features, dtype=np.float32))


class RustAudioFeaturizer(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int = 160,
        n_mels: int = 80,
        backend: str = "torchaudio",
        preemphasis: float = 0.97,
        normalize_signal: bool = True,
        normalize_feature: bool = True,
        normalize_per_frame: bool = False,
        *,
        frontend_type: str = "squeezeformer",
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = self.n_fft if win_length is None else int(win_length)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.backend = str(backend)
        self.preemphasis = float(preemphasis)
        self.normalize_signal = bool(normalize_signal)
        self.normalize_feature = bool(normalize_feature)
        self.normalize_per_frame = bool(normalize_per_frame)
        self.frontend_type = str(frontend_type)
        if self.backend != "torchaudio":
            raise ValueError(
                "RustAudioFeaturizer supports only backend='torchaudio'. "
                "Use AudioFeaturizer directly for the legacy audioflux Python path."
            )
        if self.frontend_type not in {"squeezeformer", "zipformer"}:
            raise ValueError(f"Unsupported Rust frontend type: {self.frontend_type}")
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be > 0, got {self.n_fft}.")
        if self.win_length <= 0:
            raise ValueError(f"win_length must be > 0, got {self.win_length}.")
        if self.win_length > self.n_fft:
            raise ValueError(
                f"win_length must be <= n_fft, got win_length={self.win_length}, n_fft={self.n_fft}."
            )
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be > 0, got {self.hop_length}.")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be > 0, got {self.n_mels}.")

    def forward(self, waveform: Tensor, sample_rate: int) -> Tensor:
        from asr_features import extract_squeezeformer, extract_zipformer

        extractor = extract_zipformer if self.frontend_type == "zipformer" else extract_squeezeformer
        features = extractor(
            _waveform_to_numpy(waveform),
            int(sample_rate),
            target_sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            preemphasis=self.preemphasis,
            normalize_signal=self.normalize_signal,
            normalize_feature=self.normalize_feature,
            normalize_per_frame=self.normalize_per_frame,
        )
        return _numpy_features_to_tensor(features)

    def estimate_num_frames(self, num_samples: int, sample_rate: int) -> int:
        return estimate_num_feature_frames(
            num_samples,
            sample_rate=sample_rate,
            target_sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            backend=self.backend,
        )

    def config_dict(self) -> dict[str, object]:
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "n_mels": self.n_mels,
            "backend": self.backend,
            "preemphasis": self.preemphasis,
            "normalize_signal": self.normalize_signal,
            "normalize_feature": self.normalize_feature,
            "normalize_per_frame": self.normalize_per_frame,
            "hop_length": self.hop_length,
        }


class RustW2VBertFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        model_source: str = "facebook/w2v-bert-2.0",
        *,
        sample_rate: int = 16_000,
        feature_size: int = 80,
        stride: int = 2,
        feature_dim: int | None = None,
        padding_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_source = str(model_source)
        self.sample_rate = int(sample_rate)
        self.feature_size = int(feature_size)
        self.stride = max(1, int(stride))
        resolved_feature_dim = self.feature_size * self.stride
        self.n_mels = int(feature_dim) if feature_dim is not None else resolved_feature_dim
        self.hop_length = 160 * self.stride
        self.padding_value = float(padding_value)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}.")
        if self.feature_size <= 0:
            raise ValueError(f"feature_size must be > 0, got {self.feature_size}.")
        if self.n_mels != resolved_feature_dim:
            raise ValueError(
                "RustW2VBertFeatureExtractor requires feature_dim to equal "
                f"feature_size * stride ({resolved_feature_dim}), got {self.n_mels}."
            )

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "RustW2VBertFeatureExtractor":
        sample_rate = config.get("sample_rate")
        feature_size = config.get("feature_size")
        stride = config.get("stride")
        feature_dim = config.get("feature_dim")
        padding_value = config.get("padding_value")
        return cls(
            model_source=str(config.get("model_source", "facebook/w2v-bert-2.0")),
            sample_rate=int(sample_rate) if sample_rate is not None else 16_000,
            feature_size=int(feature_size) if feature_size is not None else 80,
            stride=int(stride) if stride is not None else 2,
            feature_dim=int(feature_dim) if feature_dim is not None else None,
            padding_value=float(padding_value) if padding_value is not None else 1.0,
        )

    def forward(self, waveform: Tensor, sample_rate: int) -> Tensor:
        from asr_features import extract_w2v_bert

        features = extract_w2v_bert(
            _waveform_to_numpy(waveform),
            int(sample_rate),
            target_sample_rate=self.sample_rate,
            feature_size=self.feature_size,
            stride=self.stride,
            padding_value=self.padding_value,
        )
        return _numpy_features_to_tensor(features)

    def estimate_num_frames(self, num_samples: int, sample_rate: int) -> int:
        if num_samples <= 0 or sample_rate <= 0:
            return 0
        effective_samples = int(num_samples)
        if int(sample_rate) != self.sample_rate:
            effective_samples = max(
                1,
                int(math.ceil((effective_samples * self.sample_rate) / int(sample_rate))),
            )
        return max(1, int(math.ceil(effective_samples / max(1, self.hop_length))))

    def config_dict(self) -> dict[str, object]:
        return {
            "type": "w2v_bert",
            "model_source": self.model_source,
            "sample_rate": self.sample_rate,
            "feature_size": self.feature_size,
            "stride": self.stride,
            "feature_dim": self.n_mels,
            "padding_value": self.padding_value,
        }


class AudioFeaturizer(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int = 160,
        n_mels: int = 80,
        backend: str = "torchaudio",
        preemphasis: float = 0.97,
        normalize_signal: bool = True,
        normalize_feature: bool = True,
        normalize_per_frame: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = n_fft if win_length is None else int(win_length)
        self.n_mels = n_mels
        self.backend = backend
        self.preemphasis = preemphasis
        self.normalize_signal = normalize_signal
        self.normalize_feature = normalize_feature
        self.normalize_per_frame = normalize_per_frame
        self.hop_length = hop_length
        if self.win_length <= 0:
            raise ValueError(f"win_length must be > 0, got {self.win_length}.")
        if self.win_length > self.n_fft:
            raise ValueError(
                f"win_length must be <= n_fft, got win_length={self.win_length}, n_fft={self.n_fft}."
            )
        if backend == "torchaudio":
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=self.win_length,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            self._audioflux_bft = None
        elif backend == "audioflux":
            if af is None or SpectralFilterBankScaleType is None or SpectralDataType is None:
                raise ImportError("audioflux backend requested, but audioflux is not installed.")
            if n_fft <= 0 or n_fft & (n_fft - 1) != 0:
                raise ValueError(
                    "audioflux backend requires n_fft to be a power of two because "
                    "its mel frontend uses fft_length=2**radix2_exp."
                )
            if self.win_length != n_fft:
                raise ValueError(
                    "audioflux backend requires win_length to equal n_fft because the "
                    "configured BFT frontend does not expose a separate analysis window."
                )
            self.mel = None
            self._audioflux_bft = af.BFT(
                num=n_mels,
                radix2_exp=int(math.log2(n_fft)),
                samplate=sample_rate,
                slide_length=hop_length,
                scale_type=SpectralFilterBankScaleType.MEL,
                data_type=SpectralDataType.POWER,
            )
        else:
            raise ValueError(f"Unsupported frontend backend: {backend}")

    def forward(self, waveform: Tensor, sample_rate: int) -> Tensor:
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                sample_rate,
                self.sample_rate,
            )[0]
        if self.normalize_signal:
            waveform = waveform - waveform.mean()
            waveform = waveform / waveform.abs().amax().clamp_min(1e-6)
        if self.preemphasis > 0:
            waveform = torch.cat(
                [waveform[:1], waveform[1:] - self.preemphasis * waveform[:-1]],
                dim=0,
            )
        required_samples = max(self.n_fft, self.win_length)
        if waveform.numel() < required_samples:
            waveform = F.pad(waveform, (0, required_samples - waveform.numel()))
        if self.backend == "torchaudio":
            features = self.mel(waveform)
            features = torch.log(features.clamp_min(1e-5)).transpose(0, 1)
        else:
            spec = self._audioflux_bft.bft(
                waveform.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            spec = np.abs(spec)
            features = torch.from_numpy(spec).to(dtype=waveform.dtype).transpose(0, 1)
            features = torch.log(features.clamp_min(1e-5))
        if self.normalize_feature:
            if self.normalize_per_frame:
                mean = features.mean(dim=-1, keepdim=True)
                std = features.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
            else:
                mean = features.mean(dim=0, keepdim=True)
                std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
            features = (features - mean) / std
        return features

    def estimate_num_frames(self, num_samples: int, sample_rate: int) -> int:
        return estimate_num_feature_frames(
            num_samples,
            sample_rate=sample_rate,
            target_sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            backend=self.backend,
        )

    def config_dict(self) -> dict[str, object]:
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "n_mels": self.n_mels,
            "backend": self.backend,
            "preemphasis": self.preemphasis,
            "normalize_signal": self.normalize_signal,
            "normalize_feature": self.normalize_feature,
            "normalize_per_frame": self.normalize_per_frame,
            "hop_length": self.hop_length,
        }


class SpecAugment(torch.nn.Module):
    def __init__(
        self,
        num_freq_masks: int = 2,
        freq_mask_param: int = 27,
        num_time_masks: int = 5,
        time_mask_max_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_freq_masks = num_freq_masks
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.time_mask_max_ratio = time_mask_max_ratio

    def forward(self, features: Tensor) -> Tensor:
        augmented = features.clone()
        time_steps, feature_bins = augmented.shape

        for _ in range(self.num_freq_masks):
            width = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            if width == 0 or width >= feature_bins:
                continue
            start = int(torch.randint(0, feature_bins - width + 1, (1,)).item())
            augmented[:, start : start + width] = 0

        max_time_width = max(1, int(time_steps * self.time_mask_max_ratio))
        for _ in range(self.num_time_masks):
            width = int(torch.randint(0, max_time_width + 1, (1,)).item())
            if width == 0 or width >= time_steps:
                continue
            start = int(torch.randint(0, time_steps - width + 1, (1,)).item())
            augmented[start : start + width, :] = 0

        return augmented


class WaveformAugment(torch.nn.Module):
    def __init__(
        self,
        speed_perturb_prob: float = 0.0,
        speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1),
        noise_prob: float = 0.0,
        noise_snr_db_range: tuple[float, float] = (10.0, 30.0),
        reverb_prob: float = 0.0,
        reverb_decay_range: tuple[float, float] = (0.15, 0.5),
        reverb_delay_ms_range: tuple[float, float] = (8.0, 35.0),
    ) -> None:
        super().__init__()
        self.speed_perturb_prob = speed_perturb_prob
        self.speed_factors = speed_factors
        self.noise_prob = noise_prob
        self.noise_snr_db_range = noise_snr_db_range
        self.reverb_prob = reverb_prob
        self.reverb_decay_range = reverb_decay_range
        self.reverb_delay_ms_range = reverb_delay_ms_range

    def is_enabled(self) -> bool:
        return any(
            probability > 0.0
            for probability in (
                self.speed_perturb_prob,
                self.noise_prob,
                self.reverb_prob,
            )
        )

    def forward(self, waveform: Tensor, sample_rate: int) -> tuple[Tensor, int]:
        augmented = waveform
        current_sample_rate = sample_rate
        if self.speed_perturb_prob > 0 and torch.rand(1).item() < self.speed_perturb_prob:
            factor = self.speed_factors[int(torch.randint(0, len(self.speed_factors), (1,)).item())]
            if factor != 1.0:
                target_rate = max(1, int(round(current_sample_rate * factor)))
                augmented = torchaudio.functional.resample(
                    augmented,
                    current_sample_rate,
                    target_rate,
                )
                augmented = torchaudio.functional.resample(
                    augmented,
                    target_rate,
                    current_sample_rate,
                )
        if self.noise_prob > 0 and torch.rand(1).item() < self.noise_prob:
            augmented = self._add_noise(augmented)
        if self.reverb_prob > 0 and torch.rand(1).item() < self.reverb_prob:
            augmented = self._add_reverb(augmented, current_sample_rate)
        return augmented, current_sample_rate

    def _add_noise(self, waveform: Tensor) -> Tensor:
        low, high = self.noise_snr_db_range
        snr_db = float(torch.empty(1).uniform_(low, high).item())
        signal_power = waveform.pow(2).mean().clamp_min(1e-8)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return (waveform + noise).clamp(-1.0, 1.0)

    def _add_reverb(self, waveform: Tensor, sample_rate: int) -> Tensor:
        low_decay, high_decay = self.reverb_decay_range
        low_delay_ms, high_delay_ms = self.reverb_delay_ms_range
        decay = float(torch.empty(1).uniform_(low_decay, high_decay).item())
        delay_ms = float(torch.empty(1).uniform_(low_delay_ms, high_delay_ms).item())
        delay_samples = max(1, int(sample_rate * delay_ms / 1000.0))
        impulse_length = min(waveform.size(-1), max(delay_samples * 4, delay_samples + 1))
        impulse = waveform.new_zeros(1, 1, impulse_length)
        impulse[0, 0, 0] = 1.0
        for tap in range(1, 4):
            index = min(impulse_length - 1, tap * delay_samples)
            impulse[0, 0, index] += decay**tap
        impulse = impulse / impulse.abs().sum().clamp_min(1e-6)
        reverberated = F.conv1d(
            waveform.unsqueeze(0),
            impulse.expand(waveform.size(0), -1, -1),
            padding=impulse_length - 1,
            groups=waveform.size(0),
        )[0]
        return reverberated[..., : waveform.size(-1)].clamp(-1.0, 1.0)
