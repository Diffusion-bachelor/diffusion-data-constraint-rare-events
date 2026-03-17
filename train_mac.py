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
import time

import torch
import torch.nn as nn
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
        model_type: str = "ar",
    ):
        if model_type not in {"ar", "diffusion"}:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.model_type = model_type
        all_ids: list[int] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                text = json.loads(line)["text"]
                all_ids.extend(tokenizer.encode(text))
                all_ids.append(tokenizer.eos_token_id)

        stride = block_size
        if self.model_type == "ar":
            # AR uses (block_size + 1) to create shifted input/target pairs.
            self.blocks = [
                torch.tensor(all_ids[i : i + block_size + 1], dtype=torch.long)
                for i in range(0, len(all_ids) - block_size, stride)
            ]
        else:
            # Diffusion predicts clean tokens from noised versions of block_size tokens.
            self.blocks = [
                torch.tensor(all_ids[i : i + block_size], dtype=torch.long)
                for i in range(0, len(all_ids) - block_size + 1, stride)
            ]
        print(f"  Tokenised {len(all_ids):,} tokens → {len(self.blocks)} blocks")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int):
        block = self.blocks[idx]
        if self.model_type == "ar":
            return block[:-1], block[1:]   # input, target (causal shift)
        return block


class DiffusionGPT2(nn.Module):
    """GPT-2 denoiser with simple timestep conditioning via additive embeddings."""

    def __init__(self, config: GPT2Config, diffusion_steps: int):
        super().__init__()
        self.backbone = GPT2LMHeadModel(config)
        self.time_embedding = nn.Embedding(diffusion_steps + 1, config.n_embd)

    def forward(self, noisy_ids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        token_embeds = self.backbone.transformer.wte(noisy_ids)
        time_embeds = self.time_embedding(timesteps).unsqueeze(1)
        logits = self.backbone(inputs_embeds=token_embeds + time_embeds).logits
        return logits


def build_alpha_bar(diffusion_steps: int, device: torch.device) -> torch.Tensor:
    """Linear beta schedule with cumulative keep-probability alpha_bar[t]."""
    betas = torch.linspace(1e-4, 0.02, diffusion_steps, device=device)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def q_sample_discrete(
    clean_tokens: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bar: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Categorical noising: keep x0 with prob alpha_bar[t], else random token."""
    keep_prob = alpha_bar[timesteps - 1].unsqueeze(1)
    keep_mask = torch.rand(clean_tokens.shape, device=clean_tokens.device) < keep_prob
    random_tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=clean_tokens.shape,
        device=clean_tokens.device,
    )
    return torch.where(keep_mask, clean_tokens, random_tokens)


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


@torch.no_grad()
def evaluate_diffusion(
    model: DiffusionGPT2,
    loader: DataLoader,
    device: torch.device,
    diffusion_steps: int,
    alpha_bar: torch.Tensor,
    vocab_size: int,
):
    """Return (avg token CE, exp(CE)) for diffusion denoising objective."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for clean_tokens in loader:
        clean_tokens = clean_tokens.to(device)
        timesteps = torch.randint(
            1, diffusion_steps + 1, (clean_tokens.size(0),), device=device
        )
        noisy_tokens = q_sample_discrete(clean_tokens, timesteps, alpha_bar, vocab_size)
        logits = model(noisy_tokens, timesteps)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            clean_tokens.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += clean_tokens.numel()
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
        model_type=args.model_type,
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
    if args.model_type == "ar":
        model = GPT2LMHeadModel(config).to(device)
        alpha_bar = None
    else:
        model = DiffusionGPT2(config, args.diffusion_steps).to(device)
        alpha_bar = build_alpha_bar(args.diffusion_steps, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : {n_params:,} parameters  ({n_params / 1e6:.2f} M)")
    print(f"Type   : {args.model_type}")

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
    ckpt_dir = os.path.join("checkpoints", args.model_type)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "tiny_gpt2_best.pt")
    table_path = os.path.join(ckpt_dir, "training_table.txt")
    epoch_durations: list[float] = []
    table_header = f"{'Epoch':>6}  {'train_loss':>10}  {'val_loss':>9}  {'val_ppl':>8}  {'sec':>7}"
    table_sep = "-" * 55
    table_lines = [table_header, table_sep]
    print(f"\n{table_header}")
    print(table_sep)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        total_loss, total_tokens = 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad()

            if args.model_type == "ar":
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs).logits                   # [B, T, V]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                token_count = targets.numel()
            else:
                clean_tokens = batch.to(device)
                timesteps = torch.randint(
                    1, args.diffusion_steps + 1, (clean_tokens.size(0),), device=device
                )
                noisy_tokens = q_sample_discrete(
                    clean_tokens,
                    timesteps,
                    alpha_bar,
                    config.vocab_size,
                )
                logits = model(noisy_tokens, timesteps)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    clean_tokens.reshape(-1),
                )
                token_count = clean_tokens.numel()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * token_count
            total_tokens += token_count

        train_loss          = total_loss / total_tokens
        if args.model_type == "ar":
            val_loss, val_ppl = evaluate(model, val_loader, device)
        else:
            val_loss, val_ppl = evaluate_diffusion(
                model,
                val_loader,
                device,
                args.diffusion_steps,
                alpha_bar,
                config.vocab_size,
            )
        epoch_seconds = time.perf_counter() - epoch_start_time
        epoch_durations.append(epoch_seconds)

        marker = " *" if val_loss < best_val_loss else ""
        row_line = (
            f"{epoch:6d}  {train_loss:10.4f}  {val_loss:9.4f}  {val_ppl:8.1f}  {epoch_seconds:7.2f}{marker}"
        )
        table_lines.append(row_line)
        print(row_line)

        if args.save_each_epoch:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "model_type": args.model_type,
                    "diffusion_steps": args.diffusion_steps,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                },
                os.path.join(ckpt_dir, f"tiny_gpt2_epoch_{epoch:03d}.pt"),
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "model_type": args.model_type,
                    "diffusion_steps": args.diffusion_steps,
                },
                best_ckpt_path,
            )

    avg_epoch_seconds = sum(epoch_durations) / len(epoch_durations)
    table_lines.append(table_sep)

    with open(table_path, "w") as f:
        f.write("\n".join(table_lines) + "\n")

    print(table_sep)
    print(f"Best validation perplexity : {math.exp(best_val_loss):.2f}")
    print(f"Average epoch time (sec)  : {avg_epoch_seconds:.2f}")
    print(f"Best checkpoint saved to   : {best_ckpt_path}")
    print(f"Training table saved to    : {table_path}")
    if args.save_each_epoch:
        print(f"Per-epoch checkpoints      : {ckpt_dir}/tiny_gpt2_epoch_###.pt")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiny GPT-2 training on C4 data — CPU/MPS, no CUDA needed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", default="data/c4_sample_1k.jsonl",
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
    parser.add_argument(
        "--model-type",
        choices=["ar", "diffusion"],
        default="ar",
        help="Training objective: autoregressive next-token or discrete diffusion denoising.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=100,
        help="Number of diffusion timesteps when --model-type diffusion.",
    )
    parser.add_argument(
        "--save-each-epoch",
        action="store_true",
        help="Save an additional checkpoint file after every epoch.",
    )

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
    if args.diffusion_steps < 2:
        parser.error("--diffusion-steps must be >= 2")

    main(args)
