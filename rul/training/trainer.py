from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


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

    def predict(self, loader):
        self.model.eval()

        preds: List[np.ndarray] = []
        trues: List[np.ndarray] = []
        metas: List[Dict] = []

        with torch.no_grad():
            for X, y, meta in loader:
                X = X.to(self.device)
                pred = self.model(X).cpu().numpy()
                true = y.cpu().numpy()

                preds.append(pred)
                trues.append(true)

                batch_size = len(true)
                for i in range(batch_size):
                    row = {}
                    for k, v in meta.items():
                        if isinstance(v, list):
                            row[k] = v[i]
                        else:
                            row[k] = v[i]
                    metas.append(row)

        y_pred = np.concatenate(preds) if preds else np.array([], dtype=float)
        y_true = np.concatenate(trues) if trues else np.array([], dtype=float)
        return y_true, y_pred, metas
