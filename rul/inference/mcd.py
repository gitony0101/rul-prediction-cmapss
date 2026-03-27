from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def _predict_one_pass(model, loader, device):
    preds = []
    trues = []
    metas = []

    for X, y, meta in loader:
        X = X.to(device)
        out = model(X).detach().cpu().numpy()
        preds.append(out)
        trues.append(y.cpu().numpy())

        batch_size = len(y)
        for i in range(batch_size):
            row = {}
            for k, v in meta.items():
                row[k] = v[i]
            metas.append(row)

    y_pred = np.concatenate(preds) if preds else np.array([], dtype=float)
    y_true = np.concatenate(trues) if trues else np.array([], dtype=float)
    return y_true, y_pred, metas


def _enable_dropout_only(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout):
            submodule.train()


def mc_dropout_predict(model, loader, device, n_samples: int):
    model.eval()
    _enable_dropout_only(model)

    preds_all = []
    y_true_ref = None
    metas_ref = None

    for sample_idx in range(n_samples):
        y_true, y_pred, metas = _predict_one_pass(model, loader, device)
        preds_all.append(y_pred)

        if sample_idx == 0:
            y_true_ref = y_true
            metas_ref = metas

    preds_all = np.stack(preds_all, axis=0)
    mean = preds_all.mean(axis=0)
    std = preds_all.std(axis=0)

    return y_true_ref, mean, std, metas_ref
