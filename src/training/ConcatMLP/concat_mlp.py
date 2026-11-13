import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.GNN.dataloader import JsonlFrameIndex, SpectrumDataset
from training.GNN.train import CFG, SU_LOCATIONS, train


class ConcatMLP(nn.Module):
    """Concatenate all SU features and predict global channel occupancy."""

    def __init__(self, input_dim=32, n_su=3, n_ch=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * n_su, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_ch),
        )

    def forward(self, node_feats, edge_weights):
        del edge_weights
        batch_size = node_feats.shape[0]
        return self.classifier(node_feats.reshape(batch_size, -1))


def _load_jsonl(path):
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def _print_results(test_metrics, model):
    print("\nModel              F1      PD      PFA   Params")
    print(
        f"ConcatMLP         {test_metrics['f1']:.4f}  "
        f"{test_metrics['mean_pd']:.4f}  "
        f"{test_metrics['mean_pfa']:.4f}  "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )


def main(data_dir):
    split_generator = torch.Generator().manual_seed(42)
    torch.set_rng_state(split_generator.get_state())
    np.random.seed(42)

    data_dir = Path(data_dir)
    pu_logs_list = [
        _load_jsonl(data_dir / f"pu-{index:03d}.jsonl")
        for index in range(1, 4)
    ]
    su_frames_list = [
        JsonlFrameIndex(data_dir / f"su-{index:03d}.jsonl")
        for index in range(1, 4)
    ]

    dataset = SpectrumDataset(su_frames_list, pu_logs_list, SU_LOCATIONS)
    model = ConcatMLP(
        input_dim=CFG["input_dim"],
        n_su=CFG["n_su"],
        n_ch=CFG["n_ch"],
    )

    cfg = {**CFG, "save_dir": "./checkpoints/concat_mlp"}
    os.makedirs(cfg["save_dir"], exist_ok=True)
    _, _, test_metrics = train(model, dataset, cfg)
    _print_results(test_metrics, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="src/data/secondary/Real/Experiment_1",
        help="Directory containing su-001..003.jsonl and pu-001..003.jsonl",
    )
    args = parser.parse_args()
    main(args.data_dir)
