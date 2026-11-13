"""Data loading and dataset construction for local encoder pretraining."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from training.GNN.dataloader import build_global_label, extract_node_features


def load_json(path: str | Path) -> Any:
    """Load one JSON dataset file."""
    with Path(path).open() as file:
        return json.load(file)


def _timestamp_in_window(
    timestamp: str,
    start_timestamp: str | None,
    end_timestamp: str | None,
) -> bool:
    """Return whether a frame belongs to a manifest time window."""
    frame_time = datetime.fromisoformat(timestamp.replace("+0000", "+00:00"))
    start_time = (
        datetime.fromisoformat(start_timestamp.replace("+0000", "+00:00"))
        if start_timestamp is not None
        else None
    )
    end_time = (
        datetime.fromisoformat(end_timestamp.replace("+0000", "+00:00"))
        if end_timestamp is not None
        else None
    )
    if start_time is not None and frame_time < start_time:
        return False
    if end_time is not None and frame_time >= end_time:
        return False
    return True


class MobileSUDataset(Dataset):
    """Single-SU samples collected over one or more mobile positions.

    Each period must contain ``frames`` and ``location``.  A separate list of
    PU logs is supplied for the scenario, allowing mobile and stationary
    experiments to use different ground truth.
    """

    def __init__(
        self,
        periods: Sequence[dict[str, Any]],
        pu_logs: Sequence[Sequence[dict[str, Any]]],
    ) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for period in periods:
            location = period["location"]
            for frame in period["frames"]:
                features = extract_node_features(frame, location)
                label = build_global_label(
                    *pu_logs, timestamp=frame["timestamp"]
                )
                self.samples.append(
                    (
                        torch.tensor(features, dtype=torch.float32),
                        torch.tensor(label, dtype=torch.float32),
                    )
                )

        if not self.samples:
            raise ValueError("Mobile pretraining data contains no frames")

        print(
            f"MobileSUDataset: {len(self.samples)} single-SU samples "
            f"across {len(periods)} positions"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a scenario manifest and resolve all referenced JSON files.

    Expected shape for separate files::

        {
          "mobile": [{"frames": "mobile-a.json", "location": [0, 0]}],
          "mobile_pu_logs": ["mobile-pu-001.json"],
          "stationary_su_frames": ["stationary-su-001.json", "..."],
          "stationary_pu_logs": ["stationary-pu-001.json", "..."],
          "stationary_locations": {"su1": [0, 0], "su2": [10, 0], "su3": [5, -2.87]}
        }

    A single mobile SU stream can instead be split into location windows::

        {
          "mobile_frames": "mobile-su.json",
          "mobile": [
            {
              "location": [0, 0],
              "start_timestamp": "2026-09-23T14:00:00+00:00",
              "end_timestamp": "2026-09-23T17:00:00+00:00"
            },
            {
              "location": [10, 0],
              "start_timestamp": "2026-09-23T17:00:00+00:00",
              "end_timestamp": "2026-09-23T20:00:00+00:00"
            }
          ],
          "mobile_pu_logs": ["mobile-pu-001.json", "..."]
        }

    Windows use inclusive start and exclusive end boundaries.  ISO-8601
    timestamps should use one consistent timezone representation.
    """
    manifest_path = Path(path).resolve()
    manifest = load_json(manifest_path)
    base = manifest_path.parent

    def resolve_json(value: str | Path) -> Any:
        return load_json(base / value)

    if "mobile_frames" in manifest:
        mobile_frames = resolve_json(manifest["mobile_frames"])
        mobile_periods = []
        for period in manifest["mobile"]:
            frames = [
                frame
                for frame in mobile_frames
                if _timestamp_in_window(
                    frame["timestamp"],
                    period.get("start_timestamp"),
                    period.get("end_timestamp"),
                )
            ]
            if not frames:
                raise ValueError(
                    "Mobile location window contains no frames: "
                    f"{period.get('start_timestamp')} - "
                    f"{period.get('end_timestamp')}"
                )
            mobile_periods.append(
                {"frames": frames, "location": period["location"]}
            )
    else:
        mobile_periods = [
            {
                "frames": resolve_json(period["frames"]),
                "location": period["location"],
            }
            for period in manifest["mobile"]
        ]
    return {
        "mobile_periods": mobile_periods,
        "mobile_pu_logs": [
            resolve_json(log_path) for log_path in manifest["mobile_pu_logs"]
        ],
        "stationary_su_frames": [
            resolve_json(frame_path)
            for frame_path in manifest["stationary_su_frames"]
        ],
        "stationary_pu_logs": [
            resolve_json(log_path)
            for log_path in manifest["stationary_pu_logs"]
        ],
        "stationary_locations": manifest["stationary_locations"],
    }
