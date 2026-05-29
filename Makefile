.PHONY: setup download-kaggle download-hf inspect prepare train chat

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

download-kaggle:
	. .venv/bin/activate && python scripts/download_kaggle.py

download-hf:
	. .venv/bin/activate && python scripts/download_hf.py

inspect:
	. .venv/bin/activate && python scripts/inspect_data.py data/raw

prepare:
	. .venv/bin/activate && python scripts/prepare_mlx_data.py --raw-dir data/raw --out-dir data/processed/mlx --max-examples 5000 --valid-size 200 --test-size 200 --max-chars 6000

train:
	. .venv/bin/activate && python scripts/train_lora.py --config configs/lora_8gb.yaml

chat:
	. .venv/bin/activate && python scripts/chat.py --prompt "What are common causes of ankle swelling?"
