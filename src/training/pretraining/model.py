"""Models and weight transfer used by the local encoder pretraining phase."""

from __future__ import annotations

import torch.nn as nn

from training.GNN.graph import LocalEncoder, STGNN


class LocalEncoderWithHead(nn.Module):
    """Pretraining wrapper; the prediction head is discarded after training."""

    def __init__(self, input_dim: int = 32, latent_dim: int = 64, n_ch: int = 10):
        super().__init__()
        self.encoder = LocalEncoder(input_dim, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_ch),
        )

    def forward(self, features):
        return self.head(self.encoder(features))


def transfer_encoder_weights(
    pretrained_model: LocalEncoderWithHead, stgnn: STGNN
) -> None:
    """Copy only the pretrained local encoder into an STGNN."""
    stgnn.encoder.load_state_dict(pretrained_model.encoder.state_dict())
    count = sum(parameter.numel() for parameter in stgnn.encoder.parameters())
    print(f"Transferred {count:,} encoder parameters to STGNN")
