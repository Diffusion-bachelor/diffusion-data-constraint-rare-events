#!/usr/bin/env python3
"""
Small-scale GPT-2 language model training — runs on Mac (MPS) or CPU.
No CUDA / NVIDIA GPU required.

Uses the same C4 data and GPT-2 tokenizer as the main Megatron project,
but replaces the heavy CUDA-only infrastructure (DeepSpeed, Apex,
Flash Attention) with plain PyTorch + HuggingFace Transformers.

Usage
-----
# Quick test (~3 M param model, 5 epochs):
    python train_mac.py

# Slightly larger model, more epochs:
    python train_mac.py --n-embd 384 --n-layer 6 --n-head 6 --epochs 20

# Change data file:
    python train_mac.py --data path/to/data.jsonl
"""

import argparse
import json
import logging
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# Silence the "sequence length > 1024" warning from the GPT-2 tokenizer;
# we control sequence length via block_size inside C4BlockDataset.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# ──────────────────────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """MPS (Apple Silicon) → CUDA → CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class C4BlockDataset(Dataset):
    """
    Reads a JSONL file where each line has a ``"text"`` field,
    tokenises all text into a single flat token stream, then
    chunks it into non-overlapping blocks of length ``block_size``.

    Each item returned is (input_ids, target_ids) where target_ids
    is input_ids shifted one position to the right — the standard
    causal language-modelling setup.
    """

    def __init__(
        self,
        path: str,
        tokenizer: GPT2Tokenizer,
        block_size: int = 128,
        max_samples: int | None = None,
    ):
        all_ids: list[int] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                text = json.loads(line)["text"]
                all_ids.extend(tokenizer.encode(text))
                all_ids.append(tokenizer.eos_token_id)

        # Cut into complete blocks of (block_size + 1) so we have targets
        stride = block_size
        self.blocks = [
            torch.tensor(all_ids[i : i + block_size + 1], dtype=torch.long)
            for i in range(0, len(all_ids) - block_size, stride)
        ]
        print(f"  Tokenised {len(all_ids):,} tokens → {len(self.blocks)} blocks")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int):
        block = self.blocks[idx]
        return block[:-1], block[1:]   # input, target (causal shift)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: GPT2LMHeadModel, loader: DataLoader, device: torch.device):
    """Return (avg cross-entropy loss, perplexity) on the given dataloader."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs).logits                         # [B, T, V]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device : {device}")

    # Tokeniser — same one used by the main Megatron project
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading data: {args.data}")
    full_dataset = C4BlockDataset(
        args.data, tokenizer,
        block_size=args.block_size,
        max_samples=args.max_samples,
    )

    if len(full_dataset) < 10:
        raise RuntimeError(
            f"Only {len(full_dataset)} blocks found — use a larger dataset or "
            "reduce --block-size."
        )

    n_val   = max(1, int(0.1 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train blocks: {n_train}  |  Val blocks: {n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    # Default: ~3 M-parameter model — comfortably fits in Mac RAM and
    # trains in seconds per epoch on CPU; even faster on M-series MPS.
    #
    # To get closer to the paper's smallest config (7 M params from
    # model_params.sh: d_model=128, n_heads=4, n_layers=3) pass:
    #   --n-embd 128 --n-head 4 --n-layer 3
    config = GPT2Config(
        vocab_size=50_257,          # GPT-2 vocabulary
        n_positions=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_embd * 4,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : {n_params:,} parameters  ({n_params / 1e6:.2f} M)")

    # ── Optimiser + LR schedule ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr / 10
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\n{'Epoch':>6}  {'train_loss':>10}  {'val_loss':>9}  {'val_ppl':>8}")
    print("-" * 44)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits = model(inputs).logits                       # [B, T, V]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss   += loss.item() * targets.numel()
            total_tokens += targets.numel()

        train_loss          = total_loss / total_tokens
        val_loss, val_ppl   = evaluate(model, val_loader, device)

        marker = " *" if val_loss < best_val_loss else ""
        print(
            f"{epoch:6d}  {train_loss:10.4f}  {val_loss:9.4f}  {val_ppl:8.1f}{marker}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "config": config},
                "checkpoints/tiny_gpt2_best.pt",
            )

    print("-" * 44)
    print(f"Best validation perplexity : {math.exp(best_val_loss):.2f}")
    print(f"Checkpoint saved to        : checkpoints/tiny_gpt2_best.pt")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiny GPT-2 training on C4 data — CPU/MPS, no CUDA needed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", default="data/c4_sample.jsonl",
        help="Path to a JSONL file with {'text': '...'} entries.",
    )
    parser.add_argument("--block-size", type=int, default=128,
                        help="Token sequence length per training example.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size.")
    parser.add_argument("--epochs",     type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Peak learning rate (cosine-decayed).")

    # ── Model size ────────────────────────────────────────────────────────────
    # Defaults give ~3 M params (fast even on CPU).
    # Closest to the paper's 7 M config: --n-embd 128 --n-layer 3 --n-head 4
    parser.add_argument("--n-embd",  type=int, default=192,
                        help="Embedding / hidden dimension.")
    parser.add_argument("--n-layer", type=int, default=4,
                        help="Number of transformer layers.")
    parser.add_argument("--n-head",  type=int, default=3,
                        help="Number of attention heads (must divide n-embd).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit the number of JSONL lines loaded (useful for quick tests).")

    args = parser.parse_args()

    if args.n_embd % args.n_head != 0:
        parser.error(f"--n-embd ({args.n_embd}) must be divisible by --n-head ({args.n_head})")

    main(args)
