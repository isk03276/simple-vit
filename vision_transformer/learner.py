from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from vision_transformer.models import ViT


class ViTLearner:
    def __init__(self, model: ViT, lr: float = 3e-4):
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)

    def estimate_loss(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = self.model(images)
        loss = self.loss_func(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        return loss, acc

    def step(
        self, images: torch.Tensor, labels: torch.Tensor, is_train: bool = True
    ) -> Tuple[float, float]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        loss, acc = self.estimate_loss(images, labels)
        if is_train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item(), acc.item()
