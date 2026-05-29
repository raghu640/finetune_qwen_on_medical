#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm


SUPPORTED = {".csv", ".json", ".jsonl", ".parquet"}
SYSTEM_PROMPT = (
    "You are a careful medical education assistant. Provide helpful, concise, "
    "evidence-aware information and tell users to consult a qualified clinician "
    "for diagnosis, treatment, or urgent symptoms."
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return " ".join(str(value).replace("\r", " ").split())


def first_present(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".csv":
        for chunk in pd.read_csv(path, chunksize=1000):
            for row in chunk.to_dict(orient="records"):
                yield row
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        data = pd.read_json(path)
        for row in data.to_dict(orient="records"):
            yield row
    elif path.suffix == ".parquet":
        data = pd.read_parquet(path)
        for row in data.to_dict(orient="records"):
            yield row


def to_chat_row(row: dict[str, Any], max_chars: int) -> dict[str, Any] | None:
    instruction = first_present(row, ["instruction", "system", "task"])
    user_input = first_present(row, ["input", "question", "prompt", "query", "text"])
    output = first_present(row, ["output", "answer", "completion", "response", "assistant"])

    if not user_input and instruction:
        user_input = instruction
        instruction = ""
    if not user_input or not output:
        return None

    user_parts = []
    if instruction:
        user_parts.append(instruction)
    user_parts.append(user_input)
    user_content = "\n\n".join(user_parts)

    if len(user_content) + len(output) > max_chars:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LaViTA-style rows to MLX chat JSONL.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed/mlx")
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--valid-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    files = [p for p in raw_dir.rglob("*") if p.suffix.lower() in SUPPORTED]
    if not files:
        raise SystemExit(f"No supported files found under {raw_dir.resolve()}")

    rows: list[dict[str, Any]] = []
    for path in files:
        for row in tqdm(iter_rows(path), desc=f"Reading {path.name}"):
            chat = to_chat_row(row, args.max_chars)
            if chat is None:
                continue
            rows.append(chat)
            if len(rows) >= args.max_examples:
                break
        if len(rows) >= args.max_examples:
            break

    if not rows:
        raise SystemExit("No usable instruction/input/output rows were found.")

    random.Random(args.seed).shuffle(rows)
    test_size = min(args.test_size, max(0, len(rows) // 10))
    valid_size = min(args.valid_size, max(0, (len(rows) - test_size) // 10))

    test_rows = rows[:test_size]
    valid_rows = rows[test_size : test_size + valid_size]
    train_rows = rows[test_size + valid_size :]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    print(f"train: {len(train_rows):,} -> {(out_dir / 'train.jsonl').resolve()}")
    print(f"valid: {len(valid_rows):,} -> {(out_dir / 'valid.jsonl').resolve()}")
    print(f"test:  {len(test_rows):,} -> {(out_dir / 'test.jsonl').resolve()}")


if __name__ == "__main__":
    main()
