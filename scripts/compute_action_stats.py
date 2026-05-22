"""Compute per-dim q01/q99 action stats from LIBERO-Spatial demos.

OpenVLA-style normalization: maps each action dim to roughly [-1, 1] using
its own quantiles, so the 256-bin action tokenizer uses full resolution
across dims with different scales (rotation deltas are O(0.04), translation
deltas are O(0.5) — without per-dim norm the rotational dims use ~5% of bins).

Saves to /nyx-storage1/hanliu/mirage_ckpts/action_stats.json with:
  {"q01": [..7..], "q99": [..7..], "mean": [...], "std": [...], "n_samples": N}
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import h5py

DATASET_DIR = Path("/nyx-storage1/hanliu/envs/mirage_venv/libero/libero/datasets/libero_spatial")
OUT = Path("/nyx-storage1/hanliu/mirage_ckpts/action_stats.json")

all_actions = []
for p in sorted(DATASET_DIR.glob("*.hdf5")):
    with h5py.File(p, "r") as f:
        for demo_key in f["data"].keys():
            all_actions.append(f["data"][demo_key]["actions"][:])
A = np.concatenate(all_actions, axis=0)
print(f"loaded {A.shape[0]} action samples across {A.shape[1]} dims")
stats = {
    "q01": np.quantile(A, 0.01, axis=0).tolist(),
    "q99": np.quantile(A, 0.99, axis=0).tolist(),
    "mean": A.mean(axis=0).tolist(),
    "std": A.std(axis=0).tolist(),
    "min": A.min(axis=0).tolist(),
    "max": A.max(axis=0).tolist(),
    "n_samples": int(A.shape[0]),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(stats, indent=2))
print(f"wrote {OUT}")
for k, v in stats.items():
    if isinstance(v, list):
        print(f"  {k}: {[str(round(x, 3)) for x in v]}")
    else:
        print(f"  {k}: {v}")
