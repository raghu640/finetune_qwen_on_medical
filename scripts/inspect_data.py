#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


SUPPORTED = {".csv", ".json", ".jsonl", ".parquet"}


def read_sample(path: Path, n: int) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=n)
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for _, line in zip(range(n), f):
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if path.suffix == ".json":
        return pd.read_json(path).head(n)
    if path.suffix == ".parquet":
        return pd.read_parquet(path).head(n)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw dataset files.")
    parser.add_argument("path", nargs="?", default="data/raw")
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.path)
    console = Console()
    files = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED]

    if not files:
        console.print(f"No supported data files found under {root.resolve()}")
        return

    for path in files:
        table = Table(title=str(path))
        try:
            df = read_sample(path, args.rows)
        except Exception as exc:
            console.print(f"[red]Failed to read {path}: {exc}[/red]")
            continue

        table.add_column("column")
        table.add_column("dtype")
        table.add_column("sample")
        for col in df.columns:
            sample = ""
            if not df.empty:
                sample = str(df[col].iloc[0]).replace("\n", " ")[:120]
            table.add_row(str(col), str(df[col].dtype), sample)
        console.print(table)


if __name__ == "__main__":
    main()
