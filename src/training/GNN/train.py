import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score
from training.GNN.graph import STGNN
from training.GNN.dataloader import SpectrumDataset, label_shift_sweep
import json
import os
import time
from datetime import datetime

# CONFIGURATIONS

SU_LOCATIONS = {
    'su1': np.array([0.0, 0.0]),
    'su2': np.array([10.0, 0.0]),
    'su3': np.array([5, -2.87]),
}

CFG = {
    'input_dim':   32,      # 10 psd + 10 snr + 10 binary + 2 loc
    'latent_dim':  64,
    'n_su':        3,
    'n_ch':        10,
    'n_rounds':    2,       # GNN message passing rounds
    'batch_size':  128,
    'lr':          1e-3,
    'weight_decay':1e-4,
    'n_epochs':    50,
    'val_split':   0.15,
    'test_split':  0.15,
    'device':      'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir':    './checkpoints',
    'log_every':   5,       # print metrics every N epochs
    'num_workers': 2,
    'progress_every_batches': 100,
}

class WeightedChannelBCE(nn.Module):
    """
    Binary cross-entropy with per-channel class weighting.

    pos_weight: scalar or (n_ch,) tensor.
    Compute from your dataset:
        n_neg / n_pos per channel.
    """
    def __init__(self, pos_weight=None, n_ch=10):
        super().__init__()
        if pos_weight is None:
            pos_weight = torch.ones(n_ch) * 4.0
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, targets):
        # logits:  (batch, n_ch)
        # targets: (batch, n_ch) float 0/1
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight
        )
        return loss

def compute_pos_weight(dataset, n_ch=10, device='cpu'):
    """
    Compute per-channel positive class weight from the full dataset.
    """
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, 'labels'):
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        all_labels = dataset.dataset.labels[indices]
    elif hasattr(dataset, 'labels'):
        all_labels = dataset.labels
    else:
        raise TypeError(
            "Dataset must expose a labels tensor to compute class weights "
            "without decoding every IQ frame"
        )
    # all_labels: (N, n_ch)
    n_pos = all_labels.sum(dim=0).clamp(min=1)        # (n_ch,)
    n_neg = (1 - all_labels).sum(dim=0).clamp(min=1)  # (n_ch,)
    pos_weight = (n_neg / n_pos).to(device)
    print(f"Per-channel pos_weight: {pos_weight.cpu().numpy().round(2)}")
    return pos_weight

def compute_metrics(all_logits, all_labels, threshold=0.5):
    """
    all_logits: (N, n_ch) numpy array — raw model outputs
    all_labels: (N, n_ch) numpy array — ground truth binary

    Returns dict of metrics, both per-channel and macro-averaged.
    """
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs >= threshold).astype(int)
    labels = all_labels.astype(int)

    # Flatten across channels for macro metrics
    metrics = {
        'precision': precision_score(
            labels.flatten(), preds.flatten(),
            zero_division=0
        ),
        'recall': recall_score(
            labels.flatten(), preds.flatten(),
            zero_division=0
        ),
        'f1': f1_score(
            labels.flatten(), preds.flatten(),
            zero_division=0
        ),
        'accuracy': (preds == labels).mean(),
    }

    # Per-channel detection probability (PD) and false alarm (PFA)
    per_ch_pd  = []
    per_ch_pfa = []
    for ch in range(all_labels.shape[1]):
        true_ch = labels[:, ch]
        pred_ch = preds[:, ch]
        tp = ((pred_ch == 1) & (true_ch == 1)).sum()
        fn = ((pred_ch == 0) & (true_ch == 1)).sum()
        fp = ((pred_ch == 1) & (true_ch == 0)).sum()
        tn = ((pred_ch == 0) & (true_ch == 0)).sum()
        pd  = tp / (tp + fn + 1e-8)
        pfa = fp / (fp + tn + 1e-8)
        per_ch_pd.append(round(float(pd), 3))
        per_ch_pfa.append(round(float(pfa), 3))

    metrics['per_channel_pd']  = per_ch_pd
    metrics['per_channel_pfa'] = per_ch_pfa
    metrics['mean_pd']  = float(np.mean(per_ch_pd))
    metrics['mean_pfa'] = float(np.mean(per_ch_pfa))

    return metrics


def compute_label_shift_sweep_metrics(
    all_logits,
    dataset,
    sample_indices,
    offsets_s,
):
    """Evaluate fixed predictions against labels at candidate time offsets."""
    shifted_labels = label_shift_sweep(dataset, offsets_s)
    indices = torch.as_tensor(sample_indices, dtype=torch.long)
    return {
        offset: compute_metrics(all_logits, labels[indices].numpy())
        for offset, labels in shifted_labels.items()
    }

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    started = time.monotonic()
    total_batches = len(loader)

    for node_feats, labels, edge_weights in loader:
        node_feats   = node_feats.to(device)    # (batch, n_su, input_dim)
        labels       = labels.to(device)        # (batch, n_ch)
        edge_weights = edge_weights[0].to(device)  # (n_su, n_su) — same for all

        optimizer.zero_grad()
        logits = model(node_feats, edge_weights) # (batch, n_ch)
        loss   = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
        if n_batches % CFG['progress_every_batches'] == 0:
            elapsed = time.monotonic() - started
            print(
                f"[train] batch {n_batches:,}/{total_batches:,} "
                f"({n_batches / total_batches:.1%}), "
                f"loss={loss.item():.4f}, {elapsed:.1f}s elapsed",
                flush=True,
            )

    return total_loss / n_batches

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    n_batches   = 0
    all_logits  = []
    all_labels  = []

    for node_feats, labels, edge_weights in loader:
        node_feats   = node_feats.to(device)
        labels       = labels.to(device)
        edge_weights = edge_weights[0].to(device)

        logits = model(node_feats, edge_weights)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        n_batches  += 1

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss   = total_loss / n_batches
    metrics    = compute_metrics(all_logits, all_labels)

    return avg_loss, metrics, all_logits, all_labels


def train(model, dataset, cfg):
    device = cfg['device']
    model  = model.to(device)

    # ── Split dataset ──────────────────────────────────────────────
    n_total = len(dataset)
    n_test  = int(n_total * cfg['test_split'])
    n_val   = int(n_total * cfg['val_split'])
    n_train = n_total - n_val - n_test

    # Keep time-ordered blocks separate. Randomly mixing adjacent frames
    # leaks highly correlated observations into validation and test.
    train_ds = Subset(dataset, range(0, n_train))
    val_ds = Subset(dataset, range(n_train, n_train + n_val))
    test_ds = Subset(dataset, range(n_train + n_val, n_total))
    print(f"Dataset split — train: {n_train}, val: {n_val}, test: {n_test}")

    loader_kwargs = {
        'batch_size': cfg['batch_size'],
        'num_workers': cfg['num_workers'],
        'pin_memory': (device == 'cuda'),
        'persistent_workers': cfg['num_workers'] > 0,
    }
    if cfg['num_workers'] > 0:
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_kwargs,
    )

    # ── Loss with class-imbalance correction ───────────────────────
    pos_weight = compute_pos_weight(train_ds, cfg['n_ch'], device)
    criterion  = WeightedChannelBCE(pos_weight, cfg['n_ch']).to(device)

    # ── Optimizer and scheduler ────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    # Reduce LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # ── Training ───────────────────────────────────────────────────
    best_val_f1   = 0.0
    best_epoch    = 0
    history       = []

    for epoch in range(1, cfg['n_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, _, _ = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history.append({
            'epoch':      epoch,
            'train_loss': round(train_loss, 5),
            'val_loss':   round(val_loss, 5),
            'val_f1':     round(val_metrics['f1'], 4),
            'val_pd':     round(val_metrics['mean_pd'], 4),
            'val_pfa':    round(val_metrics['mean_pfa'], 4),
        })

        # Save best model by val F1
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch  = epoch
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'val_f1':      best_val_f1,
                'cfg':         cfg,
            }, os.path.join(cfg['save_dir'], 'best_model.pt'))

        if epoch % cfg['log_every'] == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{cfg['n_epochs']} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_F1: {val_metrics['f1']:.4f} | "
                f"val_PD: {val_metrics['mean_pd']:.4f} | "
                f"val_PFA: {val_metrics['mean_pfa']:.4f}"
            )

    print(f"\nBest val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Final test evaluation ──────────────────────────────────────
    print("\n── Test Set Evaluation (best checkpoint) ──")
    ckpt = torch.load(os.path.join(cfg['save_dir'], 'best_model.pt'))
    model.load_state_dict(ckpt['model_state'])

    _, test_metrics, test_logits, test_labels = eval_epoch(
        model, test_loader, criterion, device
    )

    print(f"  Precision:   {test_metrics['precision']:.4f}")
    print(f"  Recall:      {test_metrics['recall']:.4f}")
    print(f"  F1:          {test_metrics['f1']:.4f}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Mean PD:     {test_metrics['mean_pd']:.4f}")
    print(f"  Mean PFA:    {test_metrics['mean_pfa']:.4f}")
    print(f"\n  Per-channel PD:  {test_metrics['per_channel_pd']}")
    print(f"  Per-channel PFA: {test_metrics['per_channel_pfa']}")

    sweep_offsets = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]
    shift_metrics = compute_label_shift_sweep_metrics(
        test_logits,
        dataset,
        range(n_train + n_val, n_total),
        sweep_offsets,
    )
    print("\nLabel timestamp-offset sweep (diagnostic; not used for selection):")
    for offset, metrics in shift_metrics.items():
        print(f"  offset {offset:+.2f}s: F1={metrics['f1']:.4f}")

    # ── Save training history ──────────────────────────────────────
    history_path = os.path.join(cfg['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    return model, history, test_metrics

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from training.GNN.dataloader import JsonlFrameIndex

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="src/data/secondary/Experiment_1",
        help="Experiment directory containing su-001..003.jsonl and pu-001..003.jsonl",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    su_frames_list = [
        JsonlFrameIndex(data_dir / f"su-{index:03d}.jsonl")
        for index in range(1, 4)
    ]
    pu_logs_list = []
    for index in range(1, 4):
        with (data_dir / f"pu-{index:03d}.jsonl").open() as file:
            pu_logs_list.append([json.loads(line) for line in file if line.strip()])

    # ── Build dataset ──────────────────────────────────────────────
    dataset = SpectrumDataset(su_frames_list, pu_logs_list, SU_LOCATIONS)
    print(f"Total samples: {len(dataset)}")
    print("Local-detector audit:")
    for report in dataset.audit_local_detection_agreement():
        print(
            f"  SU {report['su_index'] + 1}: "
            f"available={report['available_frames']:,}, "
            f"missing={report['missing_frames']:,}, "
            f"agreement={report['channel_agreement']}"
        )
    print("Local-detector audit:")
    for report in dataset.audit_local_detection_agreement():
        print(
            f"  SU {report['su_index'] + 1}: "
            f"available={report['available_frames']:,}, "
            f"missing={report['missing_frames']:,}, "
            f"agreement={report['channel_agreement']}"
        )

    # ── Build model ────────────────────────────────────────────────
    model = STGNN(
        input_dim=CFG['input_dim'],
        latent_dim=CFG['latent_dim'],
        n_su=CFG['n_su'],
        n_ch=CFG['n_ch'],
        n_rounds=CFG['n_rounds']
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Train ──────────────────────────────────────────────────────
    trained_model, history, test_metrics = train(model, dataset, CFG)