"""Phase 2a smoke — confirm Show-o2 venv + LIBERO env coexist on a GPU node.

No model load, no GRPO; just verify the two stacks can live in one process.
"""

from __future__ import annotations

import os

import torch
import transformers
import diffusers
import flash_attn
import cv2
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main() -> None:
    print(f"torch={torch.__version__} tx={transformers.__version__} flash={flash_attn.__version__}")
    print(f"diffusers={diffusers.__version__} cv2={cv2.__version__} np={np.__version__}")

    bd = benchmark.get_benchmark_dict()
    spatial = bd["libero_spatial"]()
    print(f"libero_spatial tasks: {spatial.n_tasks}")
    t = spatial.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
    )
    obs = env.reset()
    img_shape = obs["agentview_image"].shape
    print(f"task: {t.language}")
    print(f"agentview_image.shape = {img_shape}")

    # One env step with zero action to confirm step() works.
    action = np.zeros(7, dtype=np.float32)
    obs, reward, done, info = env.step(action)
    print(f"after zero-action step: reward={reward} done={done} new_img_sum={obs['agentview_image'].sum()}")
    env.close()

    print("PHASE 2A.0 PASSED: Show-o2 venv + LIBERO env coexist")


if __name__ == "__main__":
    main()
