import torch
import pandas as pd
from bisect import bisect_right
import numpy as np
from scipy import signal
from training.GNN.graph import compute_edge_weights
from torch.utils.data import Dataset

def get_pu_state_at(pu_logs, timestamp):
    """
    Given sorted PU log entries and a timestamp,
    find which channels are occupied at that moment.
    Returns: binary array of length 10
    """
    # Find the log entry active at this timestamp
    idx = bisect_right([e['timestamp'] for e in pu_logs], timestamp) - 1
    if idx < 0:
        return [False] * 10
    return pu_logs[idx]['channel_occupation']

def build_global_label(pu1_logs, pu2_logs, pu3_logs, timestamp):
    """
    Global ground truth = union of all PU occupations at this timestamp
    """
    state1 = get_pu_state_at(pu1_logs, timestamp)
    state2 = get_pu_state_at(pu2_logs, timestamp)
    state3 = get_pu_state_at(pu3_logs, timestamp)
    return [a or b or c for a, b, c in zip(state1, state2, state3)]

def extract_node_features(su_frame, location):
    """
    su_frame: one entry from SU JSON
      - iq:       (1024, 2) list of [I, Q] pairs
      - snr_db:   list of 10 values, some null
      - channels: list of 10 booleans

    location: (x, y) — known fixed position of this SU in your lab
    Returns: numpy array of shape (32,)
    """
    # --- Feature 1: Per-channel PSD energy (10 values) ---
    iq_complex = np.array([c[0] + 1j*c[1] for c in su_frame['iq']])
    _, psd = signal.welch(iq_complex, fs=20e6, nperseg=640,
                          noverlap=320, nfft=640,
                          return_onesided=False)
    psd = np.fft.fftshift(np.abs(psd))
    psd_db = 10 * np.log10(psd + 1e-12)
    # Mean energy per channel (10 values)
    psd_per_channel = psd_db.reshape(10, 64).mean(axis=1)
    # Normalise
    psd_per_channel = (psd_per_channel - psd_per_channel.mean()) / \
                      (psd_per_channel.std() + 1e-8)

    # --- Feature 2: SNR vector (10 values, nulls → -30 dB) ---
    snr = np.array([
        v if v is not None else -30.0
        for v in su_frame['snr_db']
    ], dtype=np.float32)
    # Normalise (optional but helps)
    snr_norm = (snr - snr.mean()) / (snr.std() + 1e-8)

    # --- Feature 3: Binary detection (10 values) ---
    binary = np.array(su_frame['channels'], dtype=np.float32)

    # --- Feature 4: Location (2 values) ---
    loc = np.array(location, dtype=np.float32)

    return np.concatenate([
        psd_per_channel,   # 10
        snr_norm,          # 10
        binary,            # 10
        loc                # 2
    ])  # total: 32

class SpectrumDataset(Dataset):
    def __init__(self, su_frames_list, pu_logs_list, locations):
        """
        su_frames_list: list of 3 lists (one per SU), each containing frames
        pu_logs_list:   list of 3 PU log lists
        locations:      dict of SU positions
        """
        self.edge_weights = torch.tensor(
            compute_edge_weights([
                locations['su1'], locations['su2'], locations['su3']
            ])
        )

        self.samples = []

        # Align frames by timestamp across SUs
        # Assume su_frames_list[0][i], [1][i], [2][i] are temporally aligned
        n_frames = min(len(f) for f in su_frames_list)

        loc_list = [locations['su1'], locations['su2'], locations['su3']]

        for i in range(n_frames):
            node_feats = np.stack([
                extract_node_features(su_frames_list[j][i], loc_list[j])
                for j in range(3)
            ])  # (3, 32)

            # Global ground truth from PU logs
            ts = su_frames_list[0][i]['timestamp']
            label = build_global_label(
                pu_logs_list[0], pu_logs_list[1], pu_logs_list[2], ts
            )

            self.samples.append((
                torch.tensor(node_feats, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        node_feats, label = self.samples[idx]
        return node_feats, label, self.edge_weights