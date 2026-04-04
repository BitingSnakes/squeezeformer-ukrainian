#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRETS_MODULE_PATH = REPO_ROOT / "squeezeformer_pytorch" / "secrets.py"
_SECRETS_SPEC = importlib.util.spec_from_file_location(
    "squeezeformer_pytorch_secrets", SECRETS_MODULE_PATH
)
if _SECRETS_SPEC is None or _SECRETS_SPEC.loader is None:
    raise RuntimeError(f"Unable to load secrets helper from {SECRETS_MODULE_PATH}.")
_SECRETS_MODULE = importlib.util.module_from_spec(_SECRETS_SPEC)
_SECRETS_SPEC.loader.exec_module(_SECRETS_MODULE)

redact_text_secrets = _SECRETS_MODULE.redact_text_secrets
sanitize_json_text = _SECRETS_MODULE.sanitize_json_text

TEXT_FILE_EXTENSIONS = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redact Hugging Face tokens and secret-bearing JSON fields in place."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("checkpoint_best.json"),
            Path("checkpoint_last.json"),
            Path("checkpoint_topk_avg.json"),
        ],
        help="Files or directories to scrub. Defaults to common checkpoint sidecars.",
    )
    return parser.parse_args()


def iter_target_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in TEXT_FILE_EXTENSIONS
            )
        elif path.is_file():
            files.append(path)
    return files


def scrub_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        return False

    if path.suffix.lower() == ".json":
        try:
            updated = sanitize_json_text(original)
        except Exception:
            updated = redact_text_secrets(original)
    else:
        updated = redact_text_secrets(original)

    if updated == original:
        return False

    path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    changed_files = [path for path in iter_target_files(args.paths) if scrub_file(path)]
    print(f"Redacted secrets in {len(changed_files)} file(s).")
    for path in changed_files:
        print(path)


if __name__ == "__main__":
    main()
