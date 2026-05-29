# LaViTA Medical LoRA on Apple Silicon

Local-first fine-tuning project for the LaViTA medical QA dataset on an 8 GB M2 MacBook Air using `mlx-lm`.

## Staff-Engineer Read

Use `mlx-community/Qwen2.5-0.5B-Instruct-4bit` first. It is small enough to fine-tune with LoRA on 8 GB unified memory and still has a modern chat template. `mlx-community/Qwen2.5-1.5B-Instruct-4bit` is the quality upgrade, but expect tighter memory, smaller context, and slower iteration. I would only move to 1.5B after the 0.5B data path, loss curve, and eval prompts look sane.

This is not a clinical decision system. It is a medical QA style-adaptation experiment. Keep outputs framed as educational and require clinician review for anything real.

## Why MLX

`mlx-lm` is built for Apple Silicon and supports LoRA/QLoRA fine-tuning with local JSONL datasets. Its LoRA loader expects `train.jsonl`, optional `valid.jsonl`, and optional `test.jsonl`; chat rows can be formatted as `{"messages": [...]}`.

Sources checked:

- MLX LM repository: https://github.com/ml-explore/mlx-lm
- MLX LM LoRA docs: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- LaViTA medical QA dataset mirror/schema: https://huggingface.co/datasets/lavita/medical-qa-datasets
- Kaggle dataset path requested: https://www.kaggle.com/datasets/arungarimella/lavita

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This workspace already has `.venv` created and dependencies installed.

For Kaggle downloads, configure credentials with `.env`, `~/.kaggle/kaggle.json`, or deployment secrets.

Recommended local `.env`:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

If you prefer keeping credentials inside this project, put the token at `.kaggle/kaggle.json`; that directory is git-ignored.

## Download Data

Preferred, using your Kaggle dataset:

```bash
python scripts/download_kaggle.py
```

Project-local Kaggle token:

```bash
python scripts/download_kaggle.py --kaggle-config-dir .kaggle
```

Fallback, using the actual Hugging Face dataset referenced by the Kaggle package:

```bash
python scripts/download_hf.py
```

For a first Mac run, do not start with all 239k+ rows. Start with 2k to 10k examples and validate the pipeline.

## Inspect Data

```bash
python scripts/inspect_data.py data/raw
```

## Prepare MLX JSONL

```bash
python scripts/prepare_mlx_data.py \
  --raw-dir data/raw \
  --out-dir data/processed/mlx \
  --max-examples 5000 \
  --valid-size 200 \
  --test-size 200 \
  --max-chars 6000
```

The converter looks for `instruction`, `input`, and `output`. It also handles common alternates like `question`, `answer`, `prompt`, `completion`, and `response`.

## Train

```bash
python scripts/train_lora.py --config configs/lora_8gb.yaml
```

Equivalent direct command:

```bash
mlx_lm.lora --config configs/lora_8gb.yaml
```

On an 8 GB Air, close memory-heavy apps first. If the system starts swapping heavily, reduce `--max-chars` during data prep, reduce `num_layers`, or use fewer iterations.

If you see `No Metal device available`, run the command from a normal macOS Terminal session rather than a headless or sandboxed environment. MLX needs access to the Apple GPU through Metal.

## Generate

```bash
python scripts/chat.py \
  --adapter-path adapters/qwen25_05b_lavita \
  --prompt "What are common causes of ankle swelling?"
```

## Upgrade Path

After the 0.5B run works:

1. Increase prepared rows from 5k to 20k.
2. Increase `iters` from 200 to 800.
3. Try `mlx-community/Qwen2.5-1.5B-Instruct-4bit` with the same rank and `batch_size: 1`.
4. Add a held-out evaluation file with clinician-reviewed questions.

## Notes

The dataset can contain noisy, synthetic, or scraped medical answers. Filter obvious unsafe examples before any serious evaluation. A small model fine-tuned on medical text can become more confident without becoming more correct.
# finetune_qwen_on_medical
