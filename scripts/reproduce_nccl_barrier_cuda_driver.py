#!/usr/bin/env python3
"""Reproduce the NCCL barrier CUDA driver/runtime mismatch failure.

Run this on the same machine and Python environment used for training:

    NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=2 \
        scripts/reproduce_nccl_barrier_cuda_driver.py

The important operation mirrors squeezeformer_pytorch.training.data_loading:

    dist.barrier(device_ids=[torch.cuda.current_device()])

On a host with an NVIDIA driver that is too old for the installed PyTorch CUDA
runtime, this should fail with a DistBackendError containing:

    Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}.") from exc


def _rank_prefix() -> str:
    rank = os.environ.get("RANK", "?")
    return f"[rank{rank}]"


def _rank_print(message: str) -> None:
    print(f"{_rank_prefix()} {message}", flush=True)


def _torch_cuda_driver_version() -> str:
    get_driver_version = getattr(torch._C, "_cuda_getDriverVersion", None)
    if get_driver_version is None:
        return "unavailable"
    try:
        version = int(get_driver_version())
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"error: {type(exc).__name__}: {exc}"
    if version <= 0:
        return str(version)
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor} ({version})"


def _torch_cuda_runtime_version() -> str:
    get_compiled_version = getattr(torch._C, "_cuda_getCompiledVersion", None)
    if get_compiled_version is None:
        return torch.version.cuda or "unavailable"
    try:
        version = int(get_compiled_version())
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"error: {type(exc).__name__}: {exc}"
    if version <= 0:
        return str(version)
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor} ({version})"


def _nccl_version() -> str:
    try:
        version = torch.cuda.nccl.version()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"error: {type(exc).__name__}: {exc}"
    if isinstance(version, tuple):
        return ".".join(str(part) for part in version)
    return str(version)


def _print_diagnostics() -> None:
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    _rank_print(
        "host=%s local_rank=%s world_size=%s cuda_visible_devices=%s"
        % (socket.gethostname(), local_rank, world_size, os.environ.get("CUDA_VISIBLE_DEVICES"))
    )
    _rank_print(
        "torch=%s torch_cuda_runtime=%s compiled_cuda=%s cuda_driver=%s nccl=%s"
        % (
            torch.__version__,
            torch.version.cuda,
            _torch_cuda_runtime_version(),
            _torch_cuda_driver_version(),
            _nccl_version(),
        )
    )
    try:
        _rank_print(
            "cuda_is_available=%s cuda_device_count=%s current_device=%s"
            % (
                torch.cuda.is_available(),
                torch.cuda.device_count(),
                torch.cuda.current_device() if torch.cuda.is_available() else "unavailable",
            )
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        _rank_print(f"CUDA diagnostics raised {type(exc).__name__}: {exc}")


def _require_torchrun_env() -> None:
    missing = [
        name
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
        if name not in os.environ
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"missing torchrun environment variables: {missing_text}. "
            "Run with: NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=2 "
            "scripts/reproduce_nccl_barrier_cuda_driver.py"
        )


def _set_local_cuda_device() -> None:
    local_rank = _env_int("LOCAL_RANK", 0) or 0
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        _rank_print("torch.cuda.device_count() returned 0; continuing to expose CUDA/NCCL failure")
        return
    device_index = local_rank % device_count
    _rank_print(f"setting CUDA device to local_rank % device_count = {device_index}")
    torch.cuda.set_device(device_index)


def _distributed_barrier_like_training() -> None:
    if not dist.is_initialized():
        raise RuntimeError("process group was not initialized")
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        device_id = torch.cuda.current_device()
        _rank_print(f"calling dist.barrier(device_ids=[{device_id}])")
        dist.barrier(device_ids=[device_id])
        return
    _rank_print("calling dist.barrier()")
    dist.barrier()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal torch.distributed NCCL barrier repro for CUDA driver/runtime mismatch."
        )
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Process-group timeout in seconds.",
    )
    parser.add_argument(
        "--skip-set-device",
        action="store_true",
        help="Do not call torch.cuda.set_device(LOCAL_RANK) before the barrier.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip CUDA/NCCL diagnostic prints before initializing the process group.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    _require_torchrun_env()
    if not args.skip_diagnostics:
        _print_diagnostics()
    if not args.skip_set_device:
        _set_local_cuda_device()

    _rank_print("initializing process group with backend='nccl'")
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    try:
        _distributed_barrier_like_training()
        _rank_print("barrier completed successfully")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
