import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
from training.GNN.graph import STGNN
from training.GNN.dataloader import SpectrumDataset
import json
import os
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
    'batch_size':  64,
    'lr':          1e-3,
    'weight_decay':1e-4,
    'n_epochs':    50,
    'val_split':   0.15,
    'test_split':  0.15,
    'device':      'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir':    './checkpoints',
    'log_every':   5,       # print metrics every N epochs
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
    all_labels = torch.stack([dataset[i][1] for i in range(len(dataset))])
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

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0

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

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset split — train: {n_train}, val: {n_val}, test: {n_test}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=(device == 'cuda')
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=(device == 'cuda')
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

    # ── Save training history ──────────────────────────────────────
    history_path = os.path.join(cfg['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    return model, history, test_metrics

if __name__ == '__main__':

    def load_json(path):
        with open(path) as f:
            return json.load(f)

    su_frames_list = [
        load_json('./dataset/fusion/JSON/su-001.json'),
        load_json('./dataset/fusion/JSON/su-002.json'),
        load_json('./dataset/fusion/JSON/su-003.json'),
    ]
    pu_logs_list = [
        load_json('./dataset/fusion/JSON/pu-001.json'),
        load_json('./dataset/fusion/JSON/pu-002.json'),
        load_json('./dataset/fusion/JSON/pu-003.json'),
    ]

    # ── Build dataset ──────────────────────────────────────────────
    dataset = SpectrumDataset(su_frames_list, pu_logs_list, SU_LOCATIONS)
    print(f"Total samples: {len(dataset)}")

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