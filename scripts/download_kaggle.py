#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def looks_like_html(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as f:
        sample = f.read(512).lstrip().lower()
    return sample.startswith(b"<!doctype html") or sample.startswith(b"<html")


def validate_downloaded_files(path: Path) -> None:
    html_files = [p for p in path.rglob("*") if looks_like_html(p)]
    data_files = [
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl", ".parquet"}
    ]
    if html_files and not data_files:
        raise SystemExit(
            "Kaggle downloaded an HTML page instead of dataset records.\n\n"
            "This Kaggle package points at the Hugging Face dataset page. Fetch the actual data with:\n"
            "  python scripts/download_hf.py\n"
            "  python scripts/prepare_mlx_data.py --max-examples 5000\n"
        )


def download_with_kagglehub(dataset: str) -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(dataset))


def download_with_kaggle_api(dataset: str, out_dir: Path, config_dir: str | None) -> Path:
    if config_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = str(Path(config_dir).expanduser())

    from kaggle.api.kaggle_api_extended import KaggleApi

    cache_dir = out_dir / ".kaggle_download"
    cache_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(cache_dir), unzip=False)

    zip_files = sorted(cache_dir.glob("*.zip"))
    if not zip_files:
        raise RuntimeError(f"Kaggle API did not create a zip file in {cache_dir}")

    extract_dir = cache_dir / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_files[-1]) as archive:
        archive.extractall(extract_dir)
    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle LaViTA dataset.")
    parser.add_argument("--dataset", default="arungarimella/lavita")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--kaggle-config-dir",
        default=None,
        help="Directory containing kaggle.json. Defaults to Kaggle's standard location.",
    )
    args = parser.parse_args()

    load_env(Path(args.env_file))

    out_dir = Path(args.out_dir)
    print(f"Downloading Kaggle dataset: {args.dataset}")
    try:
        downloaded = download_with_kagglehub(args.dataset)
        print(f"KaggleHub cache path: {downloaded}")
    except Exception as exc:
        print(f"KaggleHub failed: {exc}")
        print("Falling back to the official Kaggle API.")
        try:
            downloaded = download_with_kaggle_api(
                args.dataset, out_dir, args.kaggle_config_dir
            )
        except Exception as api_exc:
            raise SystemExit(
                "Kaggle download failed.\n\n"
                "Make sure Kaggle credentials are configured in one of these ways:\n"
                "  1. Put kaggle.json in ~/.kaggle/kaggle.json\n"
                "  2. Put kaggle.json in .kaggle/kaggle.json and run:\n"
                "     python scripts/download_kaggle.py --kaggle-config-dir .kaggle\n"
                "  3. Put KAGGLE_USERNAME and KAGGLE_KEY in .env\n"
                "  4. Export KAGGLE_USERNAME and KAGGLE_KEY\n\n"
                f"Underlying error: {api_exc}"
            ) from api_exc
        print(f"Kaggle API extracted path: {downloaded}")

    validate_downloaded_files(downloaded)
    copy_tree(downloaded, out_dir)
    print(f"Copied files into: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
