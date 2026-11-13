"""Scenario-aware entry point for pretraining followed by STGNN training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from training.GNN.dataloader import SpectrumDataset
from training.GNN.graph import STGNN
from training.GNN.train import CFG, train

from .data import MobileSUDataset, load_manifest
from .model import transfer_encoder_weights
from .train import pretrain_encoder


def run(manifest_path: str, save_dir: str = "./checkpoints") -> None:
    data = load_manifest(manifest_path)
    cfg = dict(CFG)
    cfg["save_dir"] = str(Path(save_dir) / "stgnn")
    mobile_dataset = MobileSUDataset(
        data["mobile_periods"], data["mobile_pu_logs"]
    )
    stationary_dataset = SpectrumDataset(
        data["stationary_su_frames"],
        data["stationary_pu_logs"],
        {
            key: np.asarray(value, dtype=np.float32)
            for key, value in data["stationary_locations"].items()
        },
    )

    pretrained = pretrain_encoder(
        mobile_dataset,
        cfg,
        str(Path(save_dir) / "pretrain"),
    )
    stgnn = STGNN(
        cfg["input_dim"],
        cfg["latent_dim"],
        cfg["n_su"],
        cfg["n_ch"],
        cfg["n_rounds"],
    )
    transfer_encoder_weights(pretrained, stgnn)
    train(stgnn, stationary_dataset, cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Scenario manifest JSON path")
    parser.add_argument("--save-dir", default="./checkpoints")
    args = parser.parse_args()
    run(args.manifest, args.save_dir)


if __name__ == "__main__":
    main()
