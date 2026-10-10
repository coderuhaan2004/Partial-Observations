import json
from datetime import datetime
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[2] / 'dataset' / 'fusion' / 'JSON'

def parse_ts(ts_str):
    # Handle both offset formats your data uses
    ts_str = ts_str.replace('+0000', '+00:00')
    return datetime.fromisoformat(ts_str)

# Load and inspect each SU file
for su_id in ['001', '002', '003']:
    with open(DATASET_DIR / f'su-{su_id}.json') as f:
        data = json.load(f)

    print(f"\n── {su_id} ──")
    print(f"  Type of loaded object: {type(data)}")

    if isinstance(data, list):
        print(f"  Number of frames: {len(data)}")
        if len(data) > 1:
            t0 = parse_ts(data[0]['timestamp'])
            t1 = parse_ts(data[1]['timestamp'])
            tN = parse_ts(data[-1]['timestamp'])
            gap = (t1 - t0).total_seconds()
            dur = (tN - t0).total_seconds()
            expected = dur / gap
            print(f"  First timestamp:  {data[0]['timestamp']}")
            print(f"  Last timestamp:   {data[-1]['timestamp']}")
            print(f"  Duration:         {dur:.1f} seconds")
            print(f"  Inter-frame gap:  {gap:.4f} seconds")
            print(f"  Expected frames:  {expected:.0f}")
            print(f"  Missing frames:   {expected - len(data):.0f}")
    elif isinstance(data, dict):
        print(f"  Keys: {list(data.keys())[:5]}")
        print("  ← This is a dict, not a list. "
              "Your dataset builder cannot iterate this correctly.")

# Load and inspect each PU file
for pu_id in ['001', '002', '003']:
    with open(DATASET_DIR / f'pu-{pu_id}.json') as f:
        data = json.load(f)

    print(f"\n── {pu_id} ──")
    print(f"  Type: {type(data)}")
    if isinstance(data, list):
        print(f"  Number of occupation entries: {len(data)}")
        t0 = parse_ts(data[0]['timestamp'])
        tN = parse_ts(data[-1]['timestamp'])
        print(f"  Duration covered: {(tN-t0).total_seconds()/60:.1f} minutes")