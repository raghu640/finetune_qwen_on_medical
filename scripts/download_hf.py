#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a LaViTA dataset split from Hugging Face.")
    parser.add_argument("--name", default="lavita/ChatDoctor-HealthCareMagic-100k")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_part = f"__{args.config}" if args.config else ""
    target = out_dir / f"{args.name.replace('/', '__')}{config_part}__{args.split}.jsonl"

    dataset = load_dataset(args.name, args.config, split=args.split)
    dataset.to_json(str(target), orient="records", lines=True)
    print(f"Wrote {len(dataset):,} rows to {target.resolve()}")


if __name__ == "__main__":
    main()
