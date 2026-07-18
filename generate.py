import argparse
import os
import torch

from src.tokenizer import HMGTokenizer
from src.model import LLMTransformer
from src.utils import get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="CLI Text Generation with trained LLM")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--vocab_path", type=str, default="checkpoints/vocab.json", help="Path to tokenizer vocab")
    parser.add_argument("--prompt", type=str, default="First Citizen:\nBefore we proceed", help="Text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling threshold")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling threshold")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Enable KV Caching for speed")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load Tokenizer
    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocab file {args.vocab_path} not found. Please run train.py first.")
    tokenizer = HMGTokenizer.load_vocab(args.vocab_path)

    # Load Checkpoint
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {args.checkpoint_path} not found. Please run train.py first.")

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = LLMTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Model Architecture: {config.mode} | Parameters: {model.get_num_params():,}")
    print(f"\n--- Generating Output for Prompt: ---\n'{args.prompt}'\n" + "-" * 40)

    # Encode prompt
    input_ids = tokenizer.encode(args.prompt)
    if not input_ids:
        input_ids = [tokenizer.bos_id]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    out_ids = model.generate(
        input_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=args.use_cache,
    )

    generated_text = tokenizer.decode(out_ids[0].tolist())
    print("\n--- Generated Result: ---")
    print(generated_text)
    print("-" * 40)


if __name__ == "__main__":
    main()
