from __future__ import annotations

from enum import StrEnum


class DTypeChoice(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FP8 = "fp8"


class OptimizerChoice(StrEnum):
    MUON = "muon"
    SCALEDADAM = "scaledadam"
    ADAMW = "adamw"


class DecodeStrategy(StrEnum):
    GREEDY = "greedy"
    BEAM = "beam"


class ValidationModelSource(StrEnum):
    RAW = "raw"
    EMA = "ema"


class AdaptiveBatchUnit(StrEnum):
    FRAMES = "frames"
    TOKENS = "tokens"


class FeatureCacheFormat(StrEnum):
    FILE = "file"
    PARQUET = "parquet"
