#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_data(config: dict) -> None:

    data_dir = Path(config.get("data", "data/processed/mlx"))
    train_file = data_dir / "train.jsonl"
    if not train_file.exists() or train_file.stat().st_size == 0:
        raise SystemExit(
            "Training data is missing. Run:\n"
            "  python scripts/download_kaggle.py\n"
            "  python scripts/prepare_mlx_data.py --max-examples 5000\n"
            f"\nExpected non-empty file: {train_file.resolve()}"
        )


def prepare_adapter_dir(config: dict) -> Path:
    adapter_path = Path(config.get("adapter_path", "adapters")).expanduser()
    adapter_path.mkdir(parents=True, exist_ok=True)

    probe = adapter_path / ".write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(f"Adapter directory is not writable: {adapter_path.resolve()}\n{exc}") from exc

    return adapter_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLX LoRA training.")
    parser.add_argument("--config", default="configs/lora_8gb.yaml")
    args, extra = parser.parse_known_args()

    config = load_config(Path(args.config))
    validate_data(config)
    adapter_path = prepare_adapter_dir(config)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--config",
        args.config,
        "--adapter-path",
        str(adapter_path),
        *extra,
    ]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
