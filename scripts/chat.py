#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with the fine-tuned adapter.")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="adapters/qwen25_05b_lavita")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    medical_prompt = (
        "You are a medical education assistant. Do not diagnose. "
        "Encourage professional care for personal medical concerns.\n\n"
        f"User question: {args.prompt}"
    )
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "generate",
        "--model",
        args.model,
        "--adapter-path",
        args.adapter_path,
        "--prompt",
        medical_prompt,
        "--max-tokens",
        str(args.max_tokens),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
