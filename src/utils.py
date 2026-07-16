import os
import random
import numpy as np
import torch


def get_device() -> torch.device:
    """Auto-detect available acceleration hardware (MPS / CUDA / CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, filepath: str):
    """Saves model checkpoint to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "loss": loss,
        "config": getattr(model, "config", None),
    }
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """Loads checkpoint from disk into model and optimizer."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")
    
    state = torch.load(filepath, map_location=get_device(), weights_only=False)
    model.load_state_dict(state["model_state"])
    if optimizer and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    print(f"Loaded checkpoint from {filepath} (Epoch {state.get('epoch', 0)}, Loss: {state.get('loss', 'N/A')})")
    return state
