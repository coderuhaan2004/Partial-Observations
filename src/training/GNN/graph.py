import torch
import torch.nn as nn
import numpy as np

def compute_edge_weights(locations):
    """
    Inverse distance edge weights between all SU pairs.
    Returns adjacency matrix (3, 3).
    """
    n = len(locations)
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(locations[i] - locations[j])
                W[i, j] = 1.0 / (dist + 1e-3)
    # Row-normalise
    W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
    return W

class LocalEncoder(nn.Module):
    """
    Maps each SU's raw feature vector to a shared latent space.
    Same weights for all SUs enforces the shared representation
    described in the document.
    """
    def __init__(self, input_dim=32, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)  # (n_su, latent_dim)


class GraphAttentionLayer(nn.Module):
    """
    One round of attention-weighted message passing.
    Implements distance-weighted attention from the document.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        # Attention: given two node embeddings, compute scalar weight
        self.attention = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        self.update = nn.GRUCell(latent_dim, latent_dim)

    def forward(self, h, edge_weights):
        """
        h:            (batch, n_su, latent_dim)
        edge_weights: (n_su, n_su) — physical distance weights

        Returns updated h: (batch, n_su, latent_dim)
        """
        batch, n_su, dim = h.shape
        h_new = torch.zeros_like(h)

        for i in range(n_su):
            messages = []
            attn_weights = []

            for j in range(n_su):
                if i == j:
                    continue
                # Concatenate sender and receiver embeddings
                pair = torch.cat([h[:, i, :], h[:, j, :]], dim=-1)
                # Learned attention score
                attn = self.attention(pair)             # (batch, 1)
                # Multiply by physical distance weight
                dist_w = edge_weights[j, i]            # scalar
                attn = attn * dist_w
                messages.append(h[:, j, :])            # (batch, dim)
                attn_weights.append(attn)

            # Softmax over neighbors
            attn_weights = torch.cat(attn_weights, dim=-1)  # (batch, n_neighbors)
            attn_weights = torch.softmax(attn_weights, dim=-1).unsqueeze(-1)
            messages = torch.stack(messages, dim=1)     # (batch, n_neighbors, dim)
            aggregated = (attn_weights * messages).sum(dim=1)  # (batch, dim)

            # GRU update: combine own state with aggregated neighbor info
            h_new[:, i, :] = self.update(
                aggregated,
                h[:, i, :]
            )

        return h_new


class GlobalClassifier(nn.Module):
    """
    Takes the updated embeddings of all SUs and predicts
    the global 10-channel occupancy.
    """
    def __init__(self, latent_dim=64, n_ch=10, n_su=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * n_su, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_ch)   # raw logits, apply sigmoid for probability
        )

    def forward(self, h):
        # h: (batch, n_su, latent_dim)
        batch = h.shape[0]
        h_flat = h.view(batch, -1)   # (batch, n_su * latent_dim)
        return self.classifier(h_flat)  # (batch, n_ch)


class STGNN(nn.Module):
    def __init__(self, input_dim=32, latent_dim=64, n_su=3, n_ch=10,
                 n_rounds=2):
        super().__init__()
        self.encoder   = LocalEncoder(input_dim, latent_dim)
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(latent_dim) for _ in range(n_rounds)
        ])
        self.classifier = GlobalClassifier(latent_dim, n_ch, n_su)

    def forward(self, node_features, edge_weights):
        """
        node_features: (batch, n_su, input_dim)
        edge_weights:  (n_su, n_su)
        """
        batch, n_su, _ = node_features.shape

        # Local encoding same encoder applied to each SU
        h = torch.stack([
            self.encoder(node_features[:, j, :])
            for j in range(n_su)
        ], dim=1)                           # (batch, n_su, latent_dim)

        # Message passing rounds
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_weights) # (batch, n_su, latent_dim)

        # Global prediction
        return self.classifier(h)          # (batch, n_ch) logits