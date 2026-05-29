# Model Choice

## Default

`mlx-community/Qwen2.5-0.5B-Instruct-4bit`

This is the right first model for an 8 GB M2 Air because it keeps the base model small, uses an MLX-ready 4-bit checkpoint, and leaves enough memory headroom for LoRA activations. It is also fast enough that dataset and prompt-format mistakes show up quickly.

## Upgrade Candidate

`mlx-community/Qwen2.5-1.5B-Instruct-4bit`

Use this after the full pipeline works. It should produce better answers than the 0.5B model, but it is a tighter fit on 8 GB. Keep `batch_size: 1`, `max_seq_length: 1024`, `num_layers: 4-8`, and gradient checkpointing enabled.

## Why Not Bigger

7B-class models are not a practical fine-tuning target on an 8 GB Air. Even when quantized, LoRA training memory is driven by activations, sequence length, trainable layers, optimizer state, and OS pressure. You can run some larger models for inference, but training them locally on this machine will be unpleasant or fail.

## Medical Risk

Fine-tuning on medical QA data mostly teaches answer style and domain vocabulary. It does not verify facts, improve clinical reasoning reliably, or make the model safe for diagnosis. Treat the result as a local research prototype.
