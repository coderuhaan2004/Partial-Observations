"""The independent local encoder pretraining loop."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

from .model import LocalEncoderWithHead


class WeightedChannelBCE(nn.Module):
    """Binary cross-entropy with one positive weight per channel."""

    def __init__(self, pos_weight: torch.Tensor, n_ch: int) -> None:
        super().__init__()
        if pos_weight.shape != (n_ch,):
            raise ValueError(
                f"pos_weight must have shape ({n_ch},), got {tuple(pos_weight.shape)}"
            )
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


def pretrain_encoder(dataset, cfg: dict[str, Any], save_dir: str):
    """Train a local encoder on mobile single-SU observations."""
    os.makedirs(save_dir, exist_ok=True)
    device = cfg["device"]
    n_total = len(dataset)
    n_test = int(n_total * cfg["test_split"])
    n_val = int(n_total * cfg["val_split"])
    n_train = n_total - n_val - n_test
    if min(n_train, n_val, n_test) == 0:
        raise ValueError("Pretraining dataset is too small for train/val/test splits")

    train_ds, val_ds, _ = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    loader_args = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg.get("num_workers", 0),
        "pin_memory": device == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    labels = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
    pos_weight = (
        (1 - labels).sum(dim=0).clamp(min=1)
        / labels.sum(dim=0).clamp(min=1)
    ).to(device)
    model = LocalEncoderWithHead(
        cfg["input_dim"], cfg["latent_dim"], cfg["n_ch"]
    ).to(device)
    criterion = WeightedChannelBCE(pos_weight, cfg["n_ch"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_f1 = -1.0
    checkpoint = os.path.join(save_dir, "best_encoder.pt")
    for epoch in range(1, cfg.get("pretrain_epochs", 30) + 1):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_losses, logits, targets = [], [], []
        with torch.no_grad():
            for features, batch_targets in val_loader:
                features = features.to(device)
                batch_targets = batch_targets.to(device)
                batch_logits = model(features)
                val_losses.append(criterion(batch_logits, batch_targets).item())
                logits.append(batch_logits.cpu().numpy())
                targets.append(batch_targets.cpu().numpy())
        val_loss = float(np.mean(val_losses))
        val_f1 = f1_score(
            np.concatenate(targets).ravel(),
            (1 / (1 + np.exp(-np.concatenate(logits))) >= 0.5).ravel(),
            zero_division=0,
        )
        scheduler.step(val_loss)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint)
        if epoch == 1 or epoch % cfg.get("log_every", 5) == 0:
            print(
                f"Pretrain epoch {epoch} | train_loss: "
                f"{train_loss / len(train_loader):.4f} | val_loss: {val_loss:.4f} "
                f"| val_F1: {val_f1:.4f}"
            )

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model
