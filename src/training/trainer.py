from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


from src.inference.mcd import mc_dropout_predict, _enable_dropout_only

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        grad_clip: Optional[float] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0

        for X, y, _ in loader:
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred, y)
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def evaluate_loss(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X, y, _ in loader:
                X = X.to(self.device)
                y = y.to(self.device)

                pred = self.model(X)
                loss = self.criterion(pred, y)
                total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def predict_mcd(self, loader, num_samples: int):
        return mc_dropout_predict(self.model, loader, self.device, num_samples)

    def evaluate_loss_mcd(self, loader, num_samples: int):
        self.model.eval()
        _enable_dropout_only(self.model)
        total_loss = 0.0
        for X, y, _ in loader:
            X = X.to(self.device)
            y = y.to(self.device)
            preds = []
            for _ in range(num_samples):
                pred = self.model(X)
                preds.append(pred.detach())
            pred_mean = torch.stack(preds).mean(dim=0)
            loss = self.criterion(pred_mean, y)
            total_loss += loss.item()
        return total_loss / max(1, len(loader))
