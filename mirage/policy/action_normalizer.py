"""Per-dim action normalization using q01/q99 quantiles from training demos.

OpenVLA-style: each action dim gets mapped from [q01[i], q99[i]] to [-1, 1]
before being binned into 256 tokens. Without this, narrow-range dims (rotation
deltas at ~5% of the full action range) waste 95% of their bin resolution.

Load stats with `ActionNormalizer.from_json(path)`, normalize() before encoding,
denormalize() after decoding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActionNormalizer:
    """Maps each action dim to/from [-1, 1] using per-dim q01/q99.

    For dim i with quantiles (q01[i], q99[i]):
        normalize: a' = clip(2*(a - q01) / (q99 - q01) - 1, -1, 1)
        denormalize: a = (a' + 1) * (q99 - q01) / 2 + q01
    """

    def __init__(self, q01: np.ndarray, q99: np.ndarray) -> None:
        self.q01 = np.asarray(q01, dtype=np.float32)
        self.q99 = np.asarray(q99, dtype=np.float32)
        if self.q01.shape != self.q99.shape:
            raise ValueError(f"q01 shape {self.q01.shape} != q99 shape {self.q99.shape}")
        rng = self.q99 - self.q01
        if (rng <= 0).any():
            raise ValueError(f"degenerate q01/q99 range: {self.q01=}, {self.q99=}")
        self._range = rng

    @classmethod
    def from_json(cls, path: str | Path) -> "ActionNormalizer":
        data = json.loads(Path(path).read_text())
        return cls(q01=data["q01"], q99=data["q99"])

    @property
    def action_dim(self) -> int:
        return int(self.q01.shape[0])

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """Map raw action (..., A) -> normalized in [-1, 1] (clipped)."""
        a = np.asarray(action, dtype=np.float32)
        # Broadcast across leading dims.
        a_norm = 2.0 * (a - self.q01) / self._range - 1.0
        return np.clip(a_norm, -1.0, 1.0).astype(np.float32)

    def denormalize(self, action_norm: np.ndarray) -> np.ndarray:
        """Map normalized action (..., A) back to raw action units."""
        a_norm = np.asarray(action_norm, dtype=np.float32)
        a = (a_norm + 1.0) * self._range / 2.0 + self.q01
        return a.astype(np.float32)
