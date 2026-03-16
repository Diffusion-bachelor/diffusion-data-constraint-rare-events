# Train On Mac (No CUDA)

## Project Overview

This repository is an implementation of the paper **"Diffusion Beats Autoregressive in Data-Constrained Settings"**.

Main idea:

- In low-data regimes (many passes over limited data), **Masked Diffusion Models (MDM)** can outperform standard **autoregressive (AR)** language models.
- The primary training stack in this repo is based on Megatron + DeepSpeed and is designed for NVIDIA CUDA environments.

Key entry points:

- `pretrain_gpt.py`: AR GPT training
- `pretrain_diff_gpt.py`: diffusion GPT training

## Why a Separate Mac Script Is Needed

The standard Megatron/DeepSpeed scripts in this repository depend on CUDA-centric components (DeepSpeed ZeRO GPU workflows, fused kernels, Flash Attention, Apex). That path is not practical on macOS without CUDA.

To make small-scale local experimentation possible, use `train_mac.py`, which:

- Uses plain PyTorch + HuggingFace Transformers
- Runs on Apple Silicon (`mps`) or CPU
- Trains a small GPT-style language model on `data/c4_sample.jsonl`
- Reports validation loss and perplexity

## Quick Start

Run with defaults:

```bash
python train_mac.py
```

This will:

- Load `data/c4_sample.jsonl`
- Tokenize with GPT-2 tokenizer
- Split into train/validation
- Train a small causal LM
- Save best checkpoint to `checkpoints/tiny_gpt2_best.pt`

## Fast Sanity Run (Very Small)

```bash
python train_mac.py \
	--max-samples 100 \
	--n-embd 64 \
	--n-layer 2 \
	--n-head 4 \
	--batch-size 16 \
	--block-size 32 \
	--epochs 5
```

Good for checking the full pipeline quickly.

## Closer To Paper's Small Model Scale

The repo includes a ~7M parameter reference config in `utils/model_params.sh` (`d_model=128, n_heads=4, n_layers=3`).

Approximate this with:

```bash
python train_mac.py \
	--n-embd 128 \
	--n-head 4 \
	--n-layer 3 \
	--epochs 20
```

## How To Evaluate Performance

During training, monitor:

- `train_loss`
- `val_loss`
- `val_ppl` (validation perplexity)

Lower perplexity means better language modeling quality on the validation split.

Example output format:

```text
 Epoch  train_loss   val_loss   val_ppl
--------------------------------------------
		 1      9.9914     9.0915    8879.1
		 2      8.3577     8.0939    3274.5
		 3      7.6291     7.7565    2336.7
```

If `val_ppl` decreases across epochs, the model is learning.

## Suggested Mac Settings

- Start with `--block-size 64` or `128`
- Use smaller model dimensions first (`--n-embd 64` to `192`)
- Increase `--epochs` once training is stable
- If runs are slow, reduce `--max-samples` for quick iteration

## Notes

- This Mac workflow is for practical local experimentation, not full reproduction of the GPU scaling experiments in the paper.
- For exact paper-scale results, use the CUDA-based scripts in `examples_scaling/` on supported hardware.
