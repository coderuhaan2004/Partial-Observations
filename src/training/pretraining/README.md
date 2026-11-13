# Scenario-aware training

This package keeps mobile local-encoder pretraining separate from stationary
STGNN training. The two scenarios can provide different PU log files.

Create a JSON manifest next to the dataset files:

```json
{
  "mobile_frames": "mobile-su.json",
  "mobile": [
    {
      "location": [0.0, 0.0],
      "start_timestamp": "2026-09-23T14:00:00+00:00",
      "end_timestamp": "2026-09-23T17:00:00+00:00"
    },
    {
      "location": [10.0, 0.0],
      "start_timestamp": "2026-09-23T17:00:00+00:00",
      "end_timestamp": "2026-09-23T20:00:00+00:00"
    },
    {
      "location": [5.0, -2.87],
      "start_timestamp": "2026-09-23T20:00:00+00:00",
      "end_timestamp": "2026-09-23T23:00:00+00:00"
    }
  ],
  "mobile_pu_logs": ["mobile-pu-001.json", "mobile-pu-002.json", "mobile-pu-003.json"],
  "stationary_su_frames": [
    "stationary-su-001.json",
    "stationary-su-002.json",
    "stationary-su-003.json"
  ],
  "stationary_pu_logs": [
    "stationary-pu-001.json",
    "stationary-pu-002.json",
    "stationary-pu-003.json"
  ],
  "stationary_locations": {
    "su1": [0.0, 0.0],
    "su2": [10.0, 0.0],
    "su3": [5.0, -2.87]
  }
}
```

Windows are inclusive at `start_timestamp` and exclusive at
`end_timestamp`. Use the same timezone representation in the manifest and
the source JSON. A separate-file mobile manifest is also supported by using
`"frames"` inside each `mobile` entry instead of `"mobile_frames"` and time
windows.

Run from the repository root:

```bash
PYTHONPATH=src python -m training.pretraining.main path/to/manifest.json
```

The encoder checkpoint is written under `checkpoints/pretrain/`; STGNN
artifacts are written under `checkpoints/stgnn/`.
