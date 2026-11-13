import torch
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
import json
import os
import time
import warnings
import numpy as np
from scipy import signal
from training.GNN.graph import compute_edge_weights
from torch.utils.data import Dataset

FEATURE_SAMPLE_RATE_HZ = 10e6
_missing_local_channels_warning_emitted = False


def parse_timestamp(value):
    """Parse the timestamp formats emitted by the experiment writers."""
    return datetime.fromisoformat(value.replace("Z", "+00:00").replace("+0000", "+00:00"))


class JsonlFrameIndex:
    """Byte-offset index for a large JSONL stream.

    IQ samples are read only when a training item is requested, avoiding a
    multi-gigabyte in-memory dataset.
    """

    def __init__(self, path):
        self.path = str(path)
        self.offsets = []
        self.timestamps = []
        started = time.monotonic()
        with open(self.path, "rb") as file:
            frame_number = 0
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                frame_number += 1
                marker = b'"timestamp":"'
                start = line.find(marker)
                if start < 0:
                    raise ValueError(f"Missing timestamp in {self.path} at byte {offset}")
                start += len(marker)
                end = line.find(b'"', start)
                timestamp = line[start:end].decode("ascii")
                self.offsets.append(offset)
                self.timestamps.append(parse_timestamp(timestamp))
                if frame_number % 50000 == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"[data] indexed {self.path}: {frame_number:,} frames "
                        f"({elapsed:.1f}s)",
                        flush=True,
                    )
        if not self.timestamps:
            raise ValueError(f"JSONL stream is empty: {self.path}")
        print(
            f"[data] indexed {self.path}: {len(self.timestamps):,} frames",
            flush=True,
        )
        self._handles = {}

    def __len__(self):
        return len(self.offsets)

    def read(self, index):
        process_id = os.getpid()
        file = self._handles.get(process_id)
        if file is None:
            file = open(self.path, "rb")
            self._handles[process_id] = file
        file.seek(self.offsets[index])
        return json.loads(file.readline())


def _nearest_index(timestamps, target):
    """Return the index nearest to target in sorted timestamps."""
    index = bisect_left(timestamps, target)
    if index == 0:
        return 0
    if index == len(timestamps):
        return index - 1
    before = index - 1
    return before if target - timestamps[before] <= timestamps[index] - target else index


def prepare_pu_log(pu_log):
    """Convert a PU event log to a reusable timestamp/state representation."""
    return (
        [parse_timestamp(entry["timestamp"]) for entry in pu_log],
        [entry["channel_occupation"] for entry in pu_log],
    )


def align_su_streams(su_indexes, pu_logs, max_gap_s=0.25):
    """Align asynchronous SU streams and retain only PU-covered samples."""
    coverage_start = max(
        max(index.timestamps[0] for index in su_indexes),
        max(
            (log[0][0] if isinstance(log, tuple)
             else parse_timestamp(log[0]["timestamp"]))
            for log in pu_logs if log
        ),
    )
    coverage_end = min(
        min(index.timestamps[-1] for index in su_indexes),
        min(
            (log[0][-1] if isinstance(log, tuple)
             else parse_timestamp(log[-1]["timestamp"]))
            for log in pu_logs if log
        ),
    )
    samples = []
    reference = su_indexes[0]
    for reference_index, timestamp in enumerate(reference.timestamps):
        if timestamp < coverage_start or timestamp > coverage_end:
            continue
        indices = [reference_index]
        if any(index is reference for index in su_indexes[1:]):
            raise ValueError("SU stream indexes must be distinct")
        for index in su_indexes[1:]:
            match = _nearest_index(index.timestamps, timestamp)
            if abs(index.timestamps[match] - timestamp).total_seconds() > max_gap_s:
                break
            indices.append(match)
        else:
            samples.append((tuple(indices), timestamp))
    if not samples:
        raise ValueError("No timestamp-aligned SU samples remain after filtering")
    return samples


def get_pu_state_at(pu_logs, timestamp):
    """
    Given sorted PU log entries and a timestamp,
    find which channels are occupied at that moment.
    Returns: binary array of length 10
    """
    if not isinstance(timestamp, datetime):
        timestamp = parse_timestamp(timestamp)

    # Find the log entry active at this timestamp
    if isinstance(pu_logs, tuple):
        timestamps, states = pu_logs
    elif isinstance(timestamp, datetime):
        timestamps = [parse_timestamp(e["timestamp"]) for e in pu_logs]
        states = [e["channel_occupation"] for e in pu_logs]
    else:
        timestamps = [e["timestamp"] for e in pu_logs]
        states = [e["channel_occupation"] for e in pu_logs]
    idx = bisect_right(timestamps, timestamp) - 1
    if idx < 0:
        return [False] * 10
    return states[idx]

def build_global_label(*pu_logs, timestamp=None, offset_s=0.0):
    """
    Global ground truth is the union of all PU occupations at this timestamp.

    The number of PU logs is scenario-dependent.  Passing no logs represents
    an empty ground-truth source and returns an all-false label.
    """
    # Keep compatibility with the former
    # build_global_label(pu1, pu2, pu3, timestamp) call shape.
    if timestamp is None and pu_logs and not isinstance(pu_logs[-1], (list, tuple)):
        timestamp = pu_logs[-1]
        pu_logs = pu_logs[:-1]

    if timestamp is None:
        raise TypeError("timestamp is required")
    if not isinstance(timestamp, datetime):
        timestamp = parse_timestamp(timestamp)
    timestamp += timedelta(seconds=offset_s)
    if not pu_logs:
        return [False] * 10

    states = [get_pu_state_at(logs, timestamp) for logs in pu_logs]
    return [any(state[channel] for state in states) for channel in range(10)]

def extract_node_features(
    su_frame,
    location,
    sample_rate_hz=FEATURE_SAMPLE_RATE_HZ,
):
    """
    su_frame: one entry from SU JSON
      - iq:       (1024, 2) list of [I, Q] pairs
      - snr_db:   list of 10 values, some null
      - channels: SU-local channel detection list of 10 booleans
      - local_channels: SU-local energy-detector result, when available

    location: (x, y) — known fixed position of this SU in your lab
    Returns: numpy array of shape (32,)
    """
    # --- Feature 1: Per-channel PSD energy (10 values) ---
    iq_complex = np.array([c[0] + 1j*c[1] for c in su_frame['iq']])
    _, psd = signal.welch(iq_complex, fs=sample_rate_hz, nperseg=640,
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
    # Some SU files use "channels" for the local detector output. Global
    # labels are built separately from PU logs in SpectrumDataset.
    global _missing_local_channels_warning_emitted
    local_channels = su_frame.get("local_channels", su_frame.get("channels"))
    if local_channels is None:
        if not _missing_local_channels_warning_emitted:
            warnings.warn(
                "Frame has no local channel observation; using an all-zero "
                "observation feature.",
                RuntimeWarning,
                stacklevel=2,
            )
            _missing_local_channels_warning_emitted = True
        binary = np.zeros(10, dtype=np.float32)
    else:
        binary = np.asarray(local_channels, dtype=np.float32)
        if binary.shape != (10,):
            raise ValueError(
                "Local channel observations must contain exactly 10 values"
            )

    # --- Feature 4: Location (2 values) ---
    loc = np.array(location, dtype=np.float32)

    return np.concatenate([
        psd_per_channel,   # 10
        snr_norm,          # 10
        binary,            # 10
        loc                # 2
    ])  # total: 32

class SpectrumDataset(Dataset):
    def __init__(
        self,
        su_frames_list,
        pu_logs_list,
        locations,
        max_alignment_gap_s=0.25,
        feature_sample_rate_hz=FEATURE_SAMPLE_RATE_HZ,
        label_time_offset_s=0.0,
    ):
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

        self._locations = [
            locations['su1'], locations['su2'], locations['su3']
        ]
        self.feature_sample_rate_hz = feature_sample_rate_hz
        self.label_time_offset_s = label_time_offset_s
        self._indexes = [
            frames if isinstance(frames, JsonlFrameIndex)
            else _ListFrameIndex(frames)
            for frames in su_frames_list
        ]
        self._pu_logs = [prepare_pu_log(log) for log in pu_logs_list]
        self.samples = align_su_streams(
            self._indexes, self._pu_logs, max_alignment_gap_s
        )
        self.labels = torch.tensor(
            [
                build_global_label(
                    *self._pu_logs,
                    timestamp=timestamp,
                    offset_s=label_time_offset_s,
                )
                for _, timestamp in self.samples
            ],
            dtype=torch.float32,
        )
        self.features = torch.empty(
            (len(self.samples), len(self._indexes), 32), dtype=torch.float32
        )
        started = time.monotonic()
        for sample_index, (frame_indices, _) in enumerate(self.samples):
            self.features[sample_index] = torch.from_numpy(np.stack([
                extract_node_features(
                    self._indexes[node].read(frame_indices[node]),
                    self._locations[node],
                    sample_rate_hz=feature_sample_rate_hz,
                )
                for node in range(len(self._indexes))
            ]))
            if (sample_index + 1) % 50000 == 0:
                print(
                    f"[data] cached features: {sample_index + 1:,}/"
                    f"{len(self.samples):,} "
                    f"({time.monotonic() - started:.1f}s)",
                    flush=True,
                )
        print(
            f"[data] aligned {len(self.samples):,} samples "
            f"(max SU gap {max_alignment_gap_s:.2f}s)",
            flush=True,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.labels[idx],
            self.edge_weights,
        )

    def audit_local_detection_agreement(self):
        """Report local-detector availability and agreement with labels.

        Agreement is computed only for frames that actually contain
        ``local_channels``. Missing values are never treated as observations.
        """
        report = []
        for node, index in enumerate(self._indexes):
            available = 0
            missing = 0
            equal = 0
            total = 0
            for sample_index, (frame_indices, _) in enumerate(self.samples):
                frame = index.read(frame_indices[node])
                local_channels = frame.get(
                    "local_channels", frame.get("channels")
                )
                if local_channels is None:
                    missing += 1
                    continue
                available += 1
                observed = np.asarray(local_channels, dtype=bool)
                labels = self.labels[sample_index].numpy().astype(bool)
                equal += int(np.sum(observed == labels))
                total += labels.size
            report.append({
                "su_index": node,
                "available_frames": available,
                "missing_frames": missing,
                "channel_agreement": equal / total if total else None,
            })
        return report


def label_shift_sweep(dataset, offsets_s):
    """Build labels for candidate frame-to-PU timestamp offsets."""
    return {
        float(offset): torch.tensor(
            [
                build_global_label(
                    *dataset._pu_logs,
                    timestamp=timestamp,
                    offset_s=float(offset),
                )
                for _, timestamp in dataset.samples
            ],
            dtype=torch.float32,
        )
        for offset in offsets_s
    }


class _ListFrameIndex:
    """Compatibility adapter for existing in-memory frame lists."""

    def __init__(self, frames):
        self.frames = frames
        self.timestamps = [parse_timestamp(frame["timestamp"]) for frame in frames]

    def __len__(self):
        return len(self.frames)

    def read(self, index):
        return self.frames[index]