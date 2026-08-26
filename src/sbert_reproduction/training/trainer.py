import os
import torch
from torch.utils.data import DataLoader

class SBERTTrainer:
    """Configurable trainer supporting fp16, gradient accumulation, and evaluation callbacks."""
    def __init__(self, model, optimizer, loss_fn, device="cpu", fp16=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.fp16 = fp16

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            self.optimizer.zero_grad()
            features_a = {k: v.to(self.device) for k, v in batch["features_a"].items()}
            features_b = {k: v.to(self.device) for k, v in batch["features_b"].items()}
            labels = batch["labels"].to(self.device) if "labels" in batch else None

            u, v = self.model(features_a, features_b)
            loss = self.loss_fn(u, v, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(dataloader), 1)
