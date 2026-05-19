"""Bin continuous robot actions into discrete tokens at the tail of an LM vocab.

Convention (matches OpenVLA): the last ``bins`` token IDs of the LM vocabulary
double as the action vocabulary. For each scalar action dimension, the value is
discretized into ``bins`` uniform-width buckets across ``[min_action, max_action]``,
and the resulting bin index ``k`` maps to token ID ``vocab_size - 1 - k``.

This is the simplest representation that:
  * keeps the LM head untouched (no resize),
  * gives token-level log-probs for GRPO,
  * lets the model interleave imagined-image tokens before action tokens (the
    visual-CoT property the rest of MIRAGE is built around).

For LIBERO with 7-DoF actions + 8-step chunks, a single forward emits
``7 * 8 = 56`` action tokens after the imagined-frame chunk.
"""

from __future__ import annotations

import numpy as np
import torch


class ActionTokenizer:
    """Discrete action bins at the tail of an LM vocabulary.

    Args:
        vocab_size: tokenizer's full vocab size (e.g. 151_669 for Show-o2).
        bins: number of discrete bins per action dim (default 256).
        min_action / max_action: clip range for the continuous actions before
            digitizing. Default ``[-1, 1]`` matches OpenVLA's normalization.
    """

    def __init__(
        self,
        vocab_size: int,
        bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ) -> None:
        if bins >= vocab_size:
            raise ValueError(f"bins={bins} must be < vocab_size={vocab_size}")
        self.vocab_size = int(vocab_size)
        self.bins = int(bins)
        self.min_action = float(min_action)
        self.max_action = float(max_action)

        # Bin edges and bin centers (np for digitize; tensor versions cached lazily).
        self._edges = np.linspace(self.min_action, self.max_action, self.bins + 1)
        self._centers = (self._edges[:-1] + self._edges[1:]) / 2.0

    # --- properties --------------------------------------------------------

    @property
    def action_token_id_range(self) -> tuple[int, int]:
        """Inclusive [lo, hi] range of token IDs reserved as action bins."""
        hi = self.vocab_size - 1
        lo = self.vocab_size - self.bins
        return lo, hi

    def is_action_token(self, token_id: int) -> bool:
        lo, hi = self.action_token_id_range
        return lo <= int(token_id) <= hi

    # --- actions -> tokens -------------------------------------------------

    def encode(self, action: np.ndarray | torch.Tensor) -> np.ndarray:
        """Discretize an action vector to token IDs.

        Args:
            action: array-like with shape ``[..., action_dim]`` (or ``[..., A, C]``
                for chunked actions); values are clipped to ``[min, max]``.

        Returns:
            ``np.int64`` array of the same leading shape with each scalar
            replaced by its action token ID.
        """
        a = np.asarray(action.detach().cpu() if torch.is_tensor(action) else action,
                       dtype=np.float64)
        a = np.clip(a, self.min_action, self.max_action)
        # digitize is 1-indexed; subtract 1 to get [0, bins-1].
        bin_idx = np.digitize(a, self._edges) - 1
        bin_idx = np.clip(bin_idx, 0, self.bins - 1)
        token_ids = self.vocab_size - 1 - bin_idx
        return token_ids.astype(np.int64)

    # --- tokens -> actions -------------------------------------------------

    def decode(self, token_ids: np.ndarray | torch.Tensor) -> np.ndarray:
        """Map action token IDs back to continuous bin-center values.

        Out-of-range token IDs decode to ``np.nan`` so the caller can detect
        ill-formed generations (e.g. the model emitted a text token where an
        action token was expected).
        """
        t = np.asarray(token_ids.detach().cpu() if torch.is_tensor(token_ids) else token_ids,
                       dtype=np.int64)
        lo, hi = self.action_token_id_range
        bin_idx = self.vocab_size - 1 - t
        valid = (t >= lo) & (t <= hi)
        out = np.full(t.shape, np.nan, dtype=np.float32)
        out[valid] = self._centers[bin_idx[valid]]
        return out
