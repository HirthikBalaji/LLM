import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from src.tokenizer import HMGTokenizer


class TextDataset(Dataset):
    """
    PyTorch Dataset for causal language modeling.
    Extracts fixed-length chunks (x) and 1-step offset targets (y).
    """

    def __init__(self, token_ids: list, seq_len: int):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.token_ids[idx : idx + self.seq_len]
        y = self.token_ids[idx + 1 : idx + self.seq_len + 1]
        return x, y


def create_dataloaders(
    data_path: str,
    tokenizer: HMGTokenizer,
    seq_len: int,
    batch_size: int,
    train_split: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    """Loads raw text from file, tokenizes it, and splits into train & val DataLoaders."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer.encode(text)

    n = len(token_ids)
    train_data = token_ids[: int(n * train_split)]
    val_data = token_ids[int(n * train_split) :]

    train_dataset = TextDataset(train_data, seq_len)
    val_dataset = TextDataset(val_data, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader
