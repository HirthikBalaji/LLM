import argparse
import math
import os
import torch

from src.tokenizer import HMGTokenizer
from src.dataset import create_dataloaders
from src.model import LLMTransformer
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Perplexity & Metrics")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--vocab_path", type=str, default="checkpoints/vocab.json", help="Path to tokenizer vocab")
    parser.add_argument("--data_path", type=str, default="data/shakespeare.txt", help="Path to evaluation dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load Tokenizer
    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocab file {args.vocab_path} not found.")
    tokenizer = HMGTokenizer.load_vocab(args.vocab_path)

    # Load Checkpoint
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {args.checkpoint_path} not found.")

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = LLMTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # DataLoader
    _, val_loader = create_dataloaders(
        data_path=args.data_path,
        tokenizer=tokenizer,
        seq_len=config.block_size,
        batch_size=args.batch_size,
    )

    total_loss = 0.0
    total_batches = len(val_loader)

    print("\n--- Running Evaluation ---")
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            _, loss, _ = model(x_val, targets=y_val)
            total_loss += loss.item()

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    perplexity = math.exp(min(avg_loss, 20.0))
    bits_per_token = avg_loss / math.log(2.0)

    print("-" * 50)
    print(f"Model Mode:              {config.mode}")
    print(f"Parameters:              {model.get_num_params():,}")
    print(f"Validation Loss:         {avg_loss:.4f}")
    print(f"Perplexity (PPL):        {perplexity:.2f}")
    print(f"Bits Per Token (BPT):    {bits_per_token:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
