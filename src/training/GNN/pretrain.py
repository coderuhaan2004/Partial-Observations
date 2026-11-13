# training/GNN/pretrain.py

import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score

from training.GNN.graph import LocalEncoder, STGNN
from training.GNN.dataloader import extract_node_features, build_global_label
from train import (
    WeightedChannelBCE,
    compute_pos_weight,
    compute_metrics,
    CFG,
    SU_LOCATIONS,
)


# ─────────────────────────────────────────────────────────────────
# 1. MOBILE SU DATASET
#    Each sample is ONE frame from ONE SU position.
#    Shape: features (32,), label (10,)
#    This is fundamentally different from SpectrumDataset which
#    requires 3 simultaneous SU frames per sample.
# ─────────────────────────────────────────────────────────────────

class MobileSUDataset(Dataset):
    """
    Builds a single-SU dataset from frames collected at
    multiple positions sequentially (the mobile SU experiment).

    periods: list of dicts, each containing:
        'frames':   list of SU frame dicts (same JSON format as stationary)
        'location': np.array([x, y]) — position during this period

    pu_logs_list: [pu1_logs, pu2_logs, pu3_logs]
    """

    def __init__(self, periods, pu_logs_list):
        self.samples = []

        for period in periods:
            frames   = period['frames']
            location = period['location']

            for frame in frames:
                # Extract 32-dim feature vector using existing function
                # Location is baked into features[30:32]
                features = extract_node_features(frame, location)

                # Global ground truth from PU logs at this timestamp
                ts    = frame['timestamp']
                label = build_global_label(
                    pu_logs_list[0],
                    pu_logs_list[1],
                    pu_logs_list[2],
                    ts
                )

                self.samples.append((
                    torch.tensor(features, dtype=torch.float32),  # (32,)
                    torch.tensor(label,    dtype=torch.float32),  # (10,)
                ))

        print(f"MobileSUDataset: {len(self.samples)} single-SU samples "
              f"across {len(periods)} positions")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]   # (features_32, label_10)


# ─────────────────────────────────────────────────────────────────
# 2. LOCAL ENCODER WITH PREDICTION HEAD
#    Wraps the existing LocalEncoder and adds a linear head
#    so it can be trained on single-SU global prediction task.
#    After pre-training, only the LocalEncoder weights are
#    transferred to STGNN — the head is discarded.
# ─────────────────────────────────────────────────────────────────

class LocalEncoderWithHead(nn.Module):
    """
    Pre-training wrapper around LocalEncoder.

    LocalEncoder:  (batch, 32) → (batch, 64)
    Head:          (batch, 64) → (batch, 10) logits

    The head teaches the encoder to produce representations that
    contain enough information for per-channel occupancy prediction
    from a single SU's observations.

    After pre-training, only self.encoder weights are used —
    the head is discarded before transferring to STGNN.
    """

    def __init__(self, input_dim=32, latent_dim=64, n_ch=10):
        super().__init__()
        self.encoder = LocalEncoder(input_dim, latent_dim)
        self.head    = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_ch),
        )

    def forward(self, x):
        # x: (batch, input_dim)
        h      = self.encoder(x)    # (batch, latent_dim)
        logits = self.head(h)       # (batch, n_ch)
        return logits


# ─────────────────────────────────────────────────────────────────
# 3. PHASE 1 — PRE-TRAIN THE ENCODER
#    Train LocalEncoderWithHead on mobile single-SU dataset.
#    The encoder learns what each position's PSD looks like
#    for each occupancy state across all three positions.
# ─────────────────────────────────────────────────────────────────

def pretrain_encoder(mobile_dataset, cfg, save_dir='./checkpoints/pretrain'):
    """
    Phase 1: Pre-train LocalEncoder on mobile single-SU data.

    Returns:
        pretrained LocalEncoderWithHead (full model, with head)
        The caller extracts .encoder from this for transfer.
    """

    os.makedirs(save_dir, exist_ok=True)
    device = cfg['device']

    # ── Split ───────────────────────────────────────────────
    n_total = len(mobile_dataset)
    n_test  = int(n_total * cfg['test_split'])
    n_val   = int(n_total * cfg['val_split'])
    n_train = n_total - n_val - n_test

    train_ds, val_ds, _ = random_split(
        mobile_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"\nPre-train split — train: {n_train}, val: {n_val}")

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

    # ── Model ───────────────────────────────────────────────
    model = LocalEncoderWithHead(
        input_dim  = cfg['input_dim'],
        latent_dim = cfg['latent_dim'],
        n_ch       = cfg['n_ch'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LocalEncoderWithHead parameters: {n_params:,}")

    # ── Loss with class-imbalance correction ────────────────
    # Compute pos_weight from training split
    all_labels = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
    n_pos      = all_labels.sum(dim=0).clamp(min=1)
    n_neg      = (1 - all_labels).sum(dim=0).clamp(min=1)
    pos_weight = (n_neg / n_pos).to(device)
    print(f"Pre-train pos_weight: {pos_weight.cpu().numpy().round(2)}")

    criterion = WeightedChannelBCE(pos_weight, cfg['n_ch']).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg['lr'],
        weight_decay = cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # ── Training loop ───────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch  = 0

    # Use more epochs for pre-training since dataset is much larger
    pretrain_epochs = cfg.get('pretrain_epochs', 30)

    print(f"\n── Phase 1: Pre-training LocalEncoder "
          f"for {pretrain_epochs} epochs ──")

    for epoch in range(1, pretrain_epochs + 1):

        # Train
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for features, labels in train_loader:
            features = features.to(device)   # (batch, 32)
            labels   = labels.to(device)     # (batch, 10)

            optimizer.zero_grad()
            logits = model(features)         # (batch, 10)
            loss   = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        train_loss /= n_batches

        # Validate
        model.eval()
        val_loss   = 0.0
        n_val_b    = 0
        all_logits = []
        all_lbls   = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels   = labels.to(device)
                logits   = model(features)
                loss     = criterion(logits, labels)

                val_loss   += loss.item()
                n_val_b    += 1
                all_logits.append(logits.cpu().numpy())
                all_lbls.append(labels.cpu().numpy())

        val_loss   /= n_val_b
        all_logits  = np.concatenate(all_logits, axis=0)
        all_lbls    = np.concatenate(all_lbls,   axis=0)

        probs   = torch.sigmoid(torch.tensor(all_logits)).numpy()
        preds   = (probs >= 0.5).astype(int)
        val_f1  = f1_score(all_lbls.flatten(), preds.flatten(), zero_division=0)

        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, 'best_encoder.pt')
            )

        if epoch % cfg.get('log_every', 5) == 0 or epoch == 1:
            print(
                f"  Pretrain Epoch {epoch:3d}/{pretrain_epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_F1: {val_f1:.4f}"
            )

    print(f"\n  Best pre-train val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Load best checkpoint
    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'best_encoder.pt'))
    )

    return model   # caller uses model.encoder for transfer


# ─────────────────────────────────────────────────────────────────
# 4. TRANSFER ENCODER WEIGHTS INTO STGNN
#    Copies the pre-trained LocalEncoder weights into the STGNN.
#    The STGNN.encoder is the same LocalEncoder class —
#    weights are directly compatible.
# ─────────────────────────────────────────────────────────────────

def transfer_encoder_weights(pretrained_with_head, stgnn):
    """
    Copies LocalEncoder weights from pre-trained wrapper into STGNN.

    pretrained_with_head: LocalEncoderWithHead — output of pretrain_encoder()
    stgnn:                STGNN — target model

    The encoder submodule in both models is a LocalEncoder instance
    with the same architecture, so state_dict keys match exactly.
    """

    pretrained_encoder_state = pretrained_with_head.encoder.state_dict()
    stgnn.encoder.load_state_dict(pretrained_encoder_state)

    n_transferred = sum(p.numel() for p in stgnn.encoder.parameters())
    print(f"\nTransferred {n_transferred:,} encoder parameters to STGNN")


# ─────────────────────────────────────────────────────────────────
# 5. STGNN TRAINING WITH OPTIONAL ENCODER FREEZE
#    Reuses your existing train() structure but adds freeze control.
#    Phase 2: freeze_encoder=True  — only GNN layers + classifier train
#    Phase 3: freeze_encoder=False — all layers fine-tune together
# ─────────────────────────────────────────────────────────────────

def train_stgnn(model, dataset, cfg,
                freeze_encoder=False,
                phase_name='GNN',
                save_dir='./checkpoints'):
    """
    Trains STGNN on the stationary 3-SU dataset.
    Identical to your existing train() but with encoder freeze control.

    freeze_encoder=True  → Phase 2 (only GNN layers and classifier train)
    freeze_encoder=False → Phase 3 (all layers fine-tune, encoder at low LR)
    """

    os.makedirs(save_dir, exist_ok=True)
    device = cfg['device']
    model  = model.to(device)

    # ── Freeze or unfreeze encoder ──────────────────────────
    for param in model.encoder.parameters():
        param.requires_grad = not freeze_encoder

    frozen_params  = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"\n── Phase {phase_name} ──")
    print(f"  Encoder frozen: {freeze_encoder}")
    print(f"  Trainable params: {trainable_params:,} | "
          f"Frozen params: {frozen_params:,}")

    # ── Split dataset ───────────────────────────────────────
    n_total = len(dataset)
    n_test  = int(n_total * cfg['test_split'])
    n_val   = int(n_total * cfg['val_split'])
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Dataset split — train: {n_train}, val: {n_val}, test: {n_test}")

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

    # ── Loss ────────────────────────────────────────────────
    pos_weight = compute_pos_weight(train_ds, cfg['n_ch'], device)
    criterion  = WeightedChannelBCE(pos_weight, cfg['n_ch']).to(device)

    # ── Optimizer ────────────────────────────────────────────
    if freeze_encoder:
        # Phase 2: only GNN layers and classifier
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = cfg['lr'],
            weight_decay = cfg['weight_decay']
        )
    else:
        # Phase 3: encoder at low LR, rest at normal LR
        optimizer = torch.optim.AdamW([
            {
                'params': model.encoder.parameters(),
                'lr':     cfg['lr'] * 0.1,       # 10× smaller for encoder
            },
            {
                'params': list(model.gnn_layers.parameters()) +
                          list(model.classifier.parameters()),
                'lr':     cfg['lr'],
            },
        ], weight_decay=cfg['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # ── Training loop ────────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch  = 0
    history     = []
    ckpt_path   = os.path.join(save_dir, f'best_model_{phase_name}.pt')

    n_epochs = cfg.get(
        'n_epochs_phase2' if freeze_encoder else 'n_epochs_phase3',
        cfg['n_epochs']
    )

    for epoch in range(1, n_epochs + 1):

        # Train step
        model.train()
        train_loss = 0.0
        n_batches  = 0

        for node_feats, labels, edge_weights in train_loader:
            node_feats   = node_feats.to(device)
            labels       = labels.to(device)
            edge_weights = edge_weights[0].to(device)

            optimizer.zero_grad()
            logits = model(node_feats, edge_weights)
            loss   = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

        train_loss /= n_batches

        # Val step
        model.eval()
        val_loss   = 0.0
        n_val_b    = 0
        all_logits = []
        all_lbls   = []

        with torch.no_grad():
            for node_feats, labels, edge_weights in val_loader:
                node_feats   = node_feats.to(device)
                labels       = labels.to(device)
                edge_weights = edge_weights[0].to(device)

                logits = model(node_feats, edge_weights)
                loss   = criterion(logits, labels)

                val_loss   += loss.item()
                n_val_b    += 1
                all_logits.append(logits.cpu().numpy())
                all_lbls.append(labels.cpu().numpy())

        val_loss  /= n_val_b
        all_logits = np.concatenate(all_logits, axis=0)
        all_lbls   = np.concatenate(all_lbls,   axis=0)
        val_metrics = compute_metrics(all_logits, all_lbls)

        scheduler.step(val_loss)

        history.append({
            'epoch':      epoch,
            'phase':      phase_name,
            'train_loss': round(train_loss, 5),
            'val_loss':   round(val_loss, 5),
            'val_f1':     round(val_metrics['f1'], 4),
            'val_pd':     round(val_metrics['mean_pd'], 4),
            'val_pfa':    round(val_metrics['mean_pfa'], 4),
        })

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch  = epoch
            torch.save({
                'epoch':        epoch,
                'phase':        phase_name,
                'model_state':  model.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'val_f1':       best_val_f1,
                'cfg':          cfg,
            }, ckpt_path)

        if epoch % cfg.get('log_every', 5) == 0 or epoch == 1:
            print(
                f"  [{phase_name}] Epoch {epoch:3d}/{n_epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_F1: {val_metrics['f1']:.4f} | "
                f"val_PD: {val_metrics['mean_pd']:.4f} | "
                f"val_PFA: {val_metrics['mean_pfa']:.4f}"
            )

    print(f"\n  Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Load best checkpoint and run test evaluation
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    model.eval()
    all_logits = []
    all_lbls   = []

    with torch.no_grad():
        for node_feats, labels, edge_weights in test_loader:
            node_feats   = node_feats.to(device)
            labels       = labels.to(device)
            edge_weights = edge_weights[0].to(device)
            logits       = model(node_feats, edge_weights)
            all_logits.append(logits.cpu().numpy())
            all_lbls.append(labels.cpu().numpy())

    all_logits  = np.concatenate(all_logits, axis=0)
    all_lbls    = np.concatenate(all_lbls,   axis=0)
    test_metrics = compute_metrics(all_logits, all_lbls)

    print(f"\n── [{phase_name}] Test Set Results ──")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Mean PD:   {test_metrics['mean_pd']:.4f}")
    print(f"  Mean PFA:  {test_metrics['mean_pfa']:.4f}")
    print(f"  Per-ch PD:  {test_metrics['per_channel_pd']}")
    print(f"  Per-ch PFA: {test_metrics['per_channel_pfa']}")

    return model, history, test_metrics


# ─────────────────────────────────────────────────────────────────
# 6. COMBINED TRAINING ORCHESTRATOR
#    Phase 1 → Phase 2 → Phase 3 in sequence.
#    Returns the final fine-tuned STGNN and full history.
# ─────────────────────────────────────────────────────────────────

def train_combined(mobile_dataset, stationary_dataset, cfg,
                   save_dir='./checkpoints'):
    """
    Full three-phase combined training pipeline.

    Phase 1: Pre-train LocalEncoder on mobile single-SU data
             → learns spectral representations across 3 positions
    Phase 2: Freeze encoder, train GNN layers on stationary 3-SU data
             → learns graph collaboration with good frozen encoder
    Phase 3: Fine-tune all layers on stationary 3-SU data
             → end-to-end refinement at differentiated learning rates

    Returns: trained STGNN, combined history dict, final test metrics
    """

    full_history = {}

    # ── Phase 1: Pre-train encoder ──────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: Pre-training LocalEncoder on mobile SU data")
    print("="*60)

    pretrained = pretrain_encoder(
        mobile_dataset,
        cfg,
        save_dir=os.path.join(save_dir, 'pretrain')
    )

    # ── Build STGNN and transfer encoder weights ─────────────
    stgnn = STGNN(
        input_dim  = cfg['input_dim'],
        latent_dim = cfg['latent_dim'],
        n_su       = cfg['n_su'],
        n_ch       = cfg['n_ch'],
        n_rounds   = cfg['n_rounds']
    )

    transfer_encoder_weights(pretrained, stgnn)

    # ── Phase 2: Freeze encoder, train GNN ──────────────────
    print("\n" + "="*60)
    print("PHASE 2: Training GNN layers (encoder frozen)")
    print("="*60)

    stgnn, history_p2, metrics_p2 = train_stgnn(
        stgnn,
        stationary_dataset,
        cfg,
        freeze_encoder = True,
        phase_name     = 'Phase2_FrozenEncoder',
        save_dir       = os.path.join(save_dir, 'phase2')
    )
    full_history['phase2'] = history_p2

    # ── Phase 3: Fine-tune all layers ───────────────────────
    print("\n" + "="*60)
    print("PHASE 3: Fine-tuning all layers end-to-end")
    print("="*60)

    stgnn, history_p3, metrics_p3 = train_stgnn(
        stgnn,
        stationary_dataset,
        cfg,
        freeze_encoder = False,
        phase_name     = 'Phase3_FineTune',
        save_dir       = os.path.join(save_dir, 'phase3')
    )
    full_history['phase3'] = history_p3

    # ── Final comparison ─────────────────────────────────────
    print("\n" + "="*60)
    print("COMBINED TRAINING COMPLETE")
    print("="*60)
    print(f"\n{'Metric':<15} {'Phase2':>10} {'Phase3':>10}")
    print("-" * 37)
    for key in ['f1', 'mean_pd', 'mean_pfa']:
        print(f"{key:<15} {metrics_p2[key]:>10.4f} {metrics_p3[key]:>10.4f}")

    # Save combined history
    history_path = os.path.join(save_dir, 'combined_history.json')
    with open(history_path, 'w') as f:
        json.dump(full_history, f, indent=2)
    print(f"\nFull history saved to {history_path}")

    return stgnn, full_history, metrics_p3


# ─────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    from training.GNN.dataloader import SpectrumDataset

    def load_json(path):
        with open(path) as f:
            return json.load(f)

    # ── PU ground truth (shared across both experiments) ─────
    pu_logs_list = [
        load_json('./dataset/fusion/JSON/pu-001.json'),
        load_json('./dataset/fusion/JSON/pu-002.json'),
        load_json('./dataset/fusion/JSON/pu-003.json'),
    ]

    # ── Mobile SU dataset (Scenario 2) ───────────────────────
    # Three periods: same SU at three different positions
    # Locations must match the positions SU was placed at
    # during the mobile experiment. Adjust coordinates to match
    # your actual lab measurements.
    mobile_periods = [
        {
            'frames':   load_json('./dataset/mobile/JSON/su-mobile-pos1.json'),
            'location': np.array([0.0, 0.0]),    # left position
        },
        {
            'frames':   load_json('./dataset/mobile/JSON/su-mobile-pos2.json'),
            'location': np.array([5.0, -2.87]),  # centre position
        },
        {
            'frames':   load_json('./dataset/mobile/JSON/su-mobile-pos3.json'),
            'location': np.array([10.0, 0.0]),   # right position
        },
    ]

    mobile_dataset = MobileSUDataset(mobile_periods, pu_logs_list)

    # ── Stationary 3-SU dataset (Scenario 1) ─────────────────
    su_frames_list = [
        load_json('./dataset/fusion/JSON/su-001.json'),
        load_json('./dataset/fusion/JSON/su-002.json'),
        load_json('./dataset/fusion/JSON/su-003.json'),
    ]

    stationary_dataset = SpectrumDataset(
        su_frames_list,
        pu_logs_list,
        SU_LOCATIONS
    )

    print(f"Mobile dataset:     {len(mobile_dataset):,} single-SU samples")
    print(f"Stationary dataset: {len(stationary_dataset):,} 3-SU samples")

    # ── Extended config for combined training ─────────────────
    # Add phase-specific epoch counts to existing CFG
    COMBINED_CFG = {
        **CFG,
        'pretrain_epochs':  30,   # Phase 1: more epochs, large dataset
        'n_epochs_phase2':  30,   # Phase 2: moderate, frozen encoder
        'n_epochs_phase3':  20,   # Phase 3: fewer, fine-tuning
    }

    # ── Run combined training ─────────────────────────────────
    trained_model, history, test_metrics = train_combined(
        mobile_dataset,
        stationary_dataset,
        COMBINED_CFG,
        save_dir='./checkpoints/combined'
    )