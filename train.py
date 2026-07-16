import argparse
import math
import os
import time
import torch
from torch.optim import AdamW

from src.tokenizer import HMGTokenizer
from src.dataset import create_dataloaders
from src.model import LLMConfig, LLMTransformer
from src.utils import get_device, set_seed, save_checkpoint


def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, min_lr: float, max_lr: float) -> float:
    """Cosine learning rate scheduler with linear warmup."""
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch LLM with Differential Attention & HMG Tokenizer")
    parser.add_argument("--data_path", type=str, default="data/shakespeare.txt", help="Path to text corpus")
    parser.add_argument("--vocab_size", type=int, default=500, help="Tokenizer max vocabulary size")
    parser.add_argument("--mode", type=str, default="diff_llama", choices=["diff_llama", "llama", "gpt2"], help="Model mode")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--block_size", type=int, default=128, help="Context window sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Max learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Min learning rate for cosine schedule")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pt", help="Checkpoint save path")
    parser.add_argument("--vocab_path", type=str, default="checkpoints/vocab.json", help="Vocab save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Train Tokenizer
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path {args.data_path} not found.")

    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Training HMG Tokenizer on '{args.data_path}' ({len(raw_text)} characters)...")
    tokenizer = HMGTokenizer(max_vocab_size=args.vocab_size)
    tokenizer.train_from_text(raw_text)
    tokenizer.save_vocab(args.vocab_path)
    print(f"Tokenizer trained! Actual vocab size: {tokenizer.vocab_size}")

    # 2. Data Loaders
    train_loader, val_loader = create_dataloaders(
        data_path=args.data_path,
        tokenizer=tokenizer,
        seq_len=args.block_size,
        batch_size=args.batch_size,
    )
    print(f"Dataset loaded: {len(train_loader)} train batches, {len(val_loader)} val batches.")

    # 3. Model Setup
    config = LLMConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        mode=args.mode,
    )
    model = LLMTransformer(config).to(device)
    print(f"Model Mode: '{config.mode}' | Trainable Parameters: {model.get_num_params():,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # AMP setup (CUDA or MPS if supported)
    use_amp = device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    best_val_loss = float("inf")
    global_step = 0

    print("\n--- Starting Training ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Learning rate decay schedule
            lr = get_lr(global_step, warmup_steps, total_steps, args.min_lr, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits, loss, _ = model(x, targets=y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss, _ = model(x, targets=y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()
            global_step += 1

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                _, loss_v, _ = model(x_val, targets=y_val)
                val_loss += loss_v.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else avg_train_loss
        val_perplexity = math.exp(min(avg_val_loss, 20.0))

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Perplexity: {val_perplexity:.2f} | "
            f"LR: {lr:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_val_loss, args.save_path)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
