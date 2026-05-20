"""Phase 2A.1 — closed-loop smoke: Show-o2 + LIBERO in one process.

Goal of this smoke is **structural**, not behavioral:
    obs image  -> Show-o2 forward
              -> logits at last position
              -> constrain to action-bin token IDs and sample 7 dims
              -> ActionTokenizer.decode -> 7-D action
              -> env.step(action)
    repeat for T steps.

The raw Show-o2-1.5B has never been trained on action tokens, so the
sampled actions will be random within the bin grid. That is fine — we
just want to prove that the (perceive -> emit action token -> step env)
loop closes without errors. Success rate is meaningless here.

Gate: prints ``PHASE 2A.1 PASSED`` if 16 env steps complete without a
crash and the agent's obs sum changes step-to-step (env is actually
advancing).

Run via: ``bash scripts/phase2a_closed_loop.sh``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

SHOWO2_PATH = Path("/home/mzh1800/Show-o-repo/show-o2")
sys.path.insert(0, str(SHOWO2_PATH))
sys.path.insert(0, str(Path("/home/mzh1800/MIRAGE")))

from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive                   # noqa: E402
from models.misc import get_text_tokenizer                                       # noqa: E402
from utils import get_hyper_params, path_to_llm_name                             # noqa: E402
from datasets.utils import image_transform                                       # noqa: E402

from libero.libero import benchmark, get_libero_path                             # noqa: E402
from libero.libero.envs import OffScreenRenderEnv                                # noqa: E402

from mirage.policy import ActionTokenizer                                        # noqa: E402
from PIL import Image                                                            # noqa: E402

CONFIG_PATH = Path("/home/mzh1800/MIRAGE/configs/showo2_smoke.yaml")

ACTION_DIM = 7
T_STEPS = 16              # env steps (not full episode; just enough to prove the loop)
TASK_INDEX = 0


def encode_obs_image_to_embeds(model, vae, image_np, config, weight_dtype, device):
    """Take a raw uint8 [H, W, 3] obs image and produce Show-o2 image embeds."""
    pil = Image.fromarray(image_np[..., ::-1] if False else image_np).convert("RGB")
    # robosuite returns images upside-down; flip vertically.
    pil = pil.transpose(Image.FLIP_TOP_BOTTOM)
    img = image_transform(pil, resolution=config.dataset.preprocessing.resolution).to(device).unsqueeze(0)
    image_latents = vae.sample(img.unsqueeze(2)).squeeze(2).to(weight_dtype)
    image_embeds_und = model.image_embedder_und(image_latents)
    image_embeds_gen = model.image_embedder_gen(image_latents)
    image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
    image_embeds_und = model.und_trans(image_embeds_und)["last_hidden_state"]
    return model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))


def build_prefix_embeds(model, vae, text_tokenizer, showo_token_ids, config,
                        device, weight_dtype, image_np, task_text):
    image_embeds = encode_obs_image_to_embeds(model, vae, image_np, config, weight_dtype, device)
    sys_prompt_ids = text_tokenizer(
        "system\nYou are a helpful assistant.<|im_end|>",
        add_special_tokens=False,
    )["input_ids"]
    role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    role_b = text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    q_ids = text_tokenizer(
        f"What action should the robot take to {task_text.lower()}?",
        add_special_tokens=False,
    ).input_ids

    text_tokens_a = torch.tensor(
        [showo_token_ids["bos_id"]] + sys_prompt_ids + role_a, device=device
    )[None, :]
    text_tokens_b = torch.tensor(
        [showo_token_ids["boi_id"], showo_token_ids["eoi_id"]] + q_ids + role_b,
        device=device,
    )[None, :]
    text_embeds_a = model.showo.get_input_embeddings()(text_tokens_a)
    text_embeds_b = model.showo.get_input_embeddings()(text_tokens_b)

    _, num_mmu_image_tokens, *_ = get_hyper_params(config, text_tokenizer, showo_token_ids)

    if config.model.showo.add_time_embeds:
        time_embeds = model.time_embed(torch.tensor([[1.0]], device=device), text_embeds_a.dtype)
        if hasattr(model, "time_embed_proj"):
            time_embeds = model.time_embed_proj(time_embeds)
        prefix_embeds = torch.cat(
            [text_embeds_a, text_embeds_b[:, :1], time_embeds, image_embeds, text_embeds_b[:, 1:]],
            dim=1,
        ).to(weight_dtype)
        modality_positions = torch.tensor(
            [text_tokens_a.shape[1] + 2, num_mmu_image_tokens], device=device,
        )[None, None, :]
    else:
        prefix_embeds = torch.cat(
            [text_embeds_a, text_embeds_b[:, :1], image_embeds, text_embeds_b[:, 1:]],
            dim=1,
        ).to(weight_dtype)
        modality_positions = torch.tensor(
            [text_tokens_a.shape[1] + 1, num_mmu_image_tokens], device=device,
        )[None, None, :]

    return prefix_embeds, modality_positions


@torch.no_grad()
def sample_action(model, prefix_embeds, modality_positions, action_tokenizer, device, weight_dtype):
    """Autoregressively sample ACTION_DIM action tokens; decode to a 7-D action."""
    lo, hi = action_tokenizer.action_token_id_range

    cur_embeds = prefix_embeds
    L_prefix = prefix_embeds.shape[1]

    # Build prefix attention mask once (additive, [1, 1, L_prefix, L_prefix]).
    prefix_attn = omni_attn_mask_naive(
        B=1, LEN=L_prefix, modalities=modality_positions,
        device=device, inverted=True,
    )
    sampled_token_ids: list[int] = []

    for k in range(ACTION_DIM):
        # Attention mask grows: [1, 1, L_total, L_total]. Prefix block already
        # built; action positions attend causally to everything before.
        L_total = cur_embeds.shape[1]
        neg = torch.iinfo(torch.long).min
        attn = torch.zeros(1, 1, L_total, L_total, device=device, dtype=torch.long)
        attn[:, :, :L_prefix, :L_prefix] = prefix_attn
        # Action block (so far) is purely causal among themselves.
        n_action_so_far = L_total - L_prefix
        if n_action_so_far > 0:
            causal = torch.triu(
                torch.full((n_action_so_far, n_action_so_far), neg,
                           device=device, dtype=torch.long),
                diagonal=1,
            )
            attn[:, :, L_prefix:, L_prefix:] = causal
        attn = attn.to(weight_dtype)

        out = model.showo(inputs_embeds=cur_embeds, attention_mask=attn)
        next_logits = out.logits[:, -1, :].float()                # [1, V]

        # Restrict to action-bin token IDs.
        masked = torch.full_like(next_logits, float("-inf"))
        masked[:, lo:hi + 1] = next_logits[:, lo:hi + 1]
        probs = F.softmax(masked, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        sampled_token_ids.append(next_id)

        # Append the embedding of the chosen token to drive the next step.
        next_emb = model.showo.get_input_embeddings()(
            torch.tensor([[next_id]], device=device)
        ).to(weight_dtype)
        cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)

    token_arr = np.asarray(sampled_token_ids, dtype=np.int64)
    decoded = action_tokenizer.decode(token_arr)
    return decoded.astype(np.float32), token_arr


def main() -> None:
    config = OmegaConf.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if config.model.weight_type == "bfloat16" else torch.float32
    torch.manual_seed(0)
    np.random.seed(0)

    print("[2a.1] loading tokenizer", flush=True)
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    vocab_size = len(text_tokenizer)

    print("[2a.1] loading Wan-VAE", flush=True)
    vae = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_dtype,
        device=device,
    )

    print("[2a.1] loading Show-o2-1.5B", flush=True)
    model = Showo2Qwen2_5.from_pretrained(
        config.model.showo.pretrained_model_path,
        use_safetensors=False,
    ).to(device).to(weight_dtype)
    model.eval()

    action_tokenizer = ActionTokenizer(vocab_size=vocab_size, bins=256)
    print(f"[2a.1] action token range: {action_tokenizer.action_token_id_range}", flush=True)

    # -- LIBERO env -------------------------------------------------------
    bd = benchmark.get_benchmark_dict()
    spatial = bd["libero_spatial"]()
    t = spatial.get_task(TASK_INDEX)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    print(f"[2a.1] task: {t.language}", flush=True)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    obs = env.reset()

    sums: list[int] = []
    actions_emitted: list[np.ndarray] = []
    for step in range(T_STEPS):
        prefix_embeds, modality_positions = build_prefix_embeds(
            model, vae, text_tokenizer, showo_token_ids, config,
            device, weight_dtype, obs["agentview_image"], t.language,
        )
        action, tokens = sample_action(
            model, prefix_embeds, modality_positions,
            action_tokenizer, device, weight_dtype,
        )
        actions_emitted.append(action)
        obs, reward, done, info = env.step(action)
        sums.append(int(obs["agentview_image"].sum()))
        print(
            f"[2a.1] step {step:02d} action={np.round(action, 3).tolist()} "
            f"tokens={tokens.tolist()} reward={reward} done={done}",
            flush=True,
        )
        if done:
            print(f"[2a.1] env reported done at step {step}", flush=True)
            break

    env.close()

    distinct_sums = len(set(sums))
    print("=" * 60, flush=True)
    print(f"steps run            : {len(sums)}", flush=True)
    print(f"distinct obs.sum()   : {distinct_sums} (out of {len(sums)})", flush=True)
    print(f"first action emitted : {actions_emitted[0]}", flush=True)
    if distinct_sums < 2:
        raise RuntimeError("obs did not change across steps — env is stuck or actions are noops")
    print("PHASE 2A.1 PASSED", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
