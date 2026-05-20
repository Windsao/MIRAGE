"""Stream (obs, action) frames from LIBERO demonstration HDF5s.

Each LIBERO demo HDF5 is organised as ``data/demo_<i>/{obs/agentview_rgb,
actions, ...}`` (T-step trajectories). We flatten all (file, demo, t) triples
into a flat index and return one example per ``__getitem__``.

For SFT smoke we only return the *next* action (chunk size 1). The chunked
variant can layer on later by stacking ``action[t:t+C]``.

Robosuite renders ``agentview_rgb`` vertically flipped relative to what a
human (and Show-o2) expects; we flip on load.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from torch.utils.data import Dataset


class LiberoActionDataset(Dataset):
    """Flat (obs, task, action) examples from one or more LIBERO HDF5 files.

    Args:
        hdf5_paths: list of demo file paths (one per task in a suite).
        task_descriptions: parallel list of natural-language task descriptions.
        max_steps_per_demo: cap per-demo length to avoid the long tails of slow
            trajectories dominating SFT. ``None`` keeps all steps.
        image_key: which observation channel to use as the policy obs image.
    """

    def __init__(
        self,
        hdf5_paths: Iterable[str],
        task_descriptions: Iterable[str],
        max_steps_per_demo: int | None = None,
        image_key: str = "agentview_rgb",
    ) -> None:
        super().__init__()
        self.hdf5_paths = [str(p) for p in hdf5_paths]
        self.task_descriptions = list(task_descriptions)
        if len(self.hdf5_paths) != len(self.task_descriptions):
            raise ValueError(
                f"got {len(self.hdf5_paths)} files but {len(self.task_descriptions)} "
                f"task descriptions; they must align"
            )
        self.image_key = image_key
        self.max_steps_per_demo = max_steps_per_demo

        # Build a flat index of (file_idx, demo_key, t).
        self.index: list[tuple[int, str, int]] = []
        for fi, path in enumerate(self.hdf5_paths):
            with h5py.File(path, "r") as f:
                for demo_key in f["data"].keys():
                    T = int(f["data"][demo_key]["actions"].shape[0])
                    if self.max_steps_per_demo is not None:
                        T = min(T, self.max_steps_per_demo)
                    self.index.extend((fi, demo_key, t) for t in range(T))
        if not self.index:
            raise RuntimeError(f"no demos found across {self.hdf5_paths}")
        self._file_cache: dict[int, h5py.File] = {}

    # --- torch Dataset protocol ------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        fi, demo_key, t = self.index[idx]
        f = self._open(fi)
        demo = f["data"][demo_key]
        # robosuite renders agentview upside down; flip vertically.
        img = np.flipud(demo["obs"][self.image_key][t]).copy()           # [H, W, 3] uint8
        action = demo["actions"][t].astype(np.float32)                   # [7]
        return {
            "image": img,
            "action": action,
            "task": self.task_descriptions[fi],
        }

    # --- helpers ----------------------------------------------------------

    def _open(self, fi: int) -> h5py.File:
        if fi not in self._file_cache:
            self._file_cache[fi] = h5py.File(self.hdf5_paths[fi], "r", libver="latest", swmr=True)
        return self._file_cache[fi]

    def close(self) -> None:
        for f in self._file_cache.values():
            f.close()
        self._file_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --- convenience factory ------------------------------------------------

def libero_spatial_dataset(
    dataset_dir: str | os.PathLike,
    max_steps_per_demo: int | None = None,
) -> LiberoActionDataset:
    """Build a dataset from the standard libero_spatial demo dir.

    Expects one ``.hdf5`` per task with the canonical LIBERO file name. The
    natural-language task description is read from the LIBERO benchmark dict
    so file paths and tasks stay aligned.
    """
    from libero.libero import benchmark

    bd = benchmark.get_benchmark_dict()
    spatial = bd["libero_spatial"]()
    paths: list[str] = []
    tasks: list[str] = []
    dataset_dir = Path(dataset_dir)
    for i in range(spatial.n_tasks):
        t = spatial.get_task(i)
        # LIBERO names demos after the BDDL file with `_demo.hdf5` suffix.
        demo_name = Path(t.bddl_file).stem + "_demo.hdf5"
        p = dataset_dir / "libero_spatial" / demo_name
        if not p.exists():
            raise FileNotFoundError(p)
        paths.append(str(p))
        tasks.append(t.language)
    return LiberoActionDataset(
        hdf5_paths=paths,
        task_descriptions=tasks,
        max_steps_per_demo=max_steps_per_demo,
    )
