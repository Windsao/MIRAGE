"""Phase 1.4 — does the GRPO math close on Show-o2-1.5B?

This is the minimum-viable test that one full GRPO step does not crash:
forward through the model, log-probs over generated action tokens, group-
relative advantage, policy-gradient loss, ``backward()``, ``optimizer.step()``.

Scope:
  * Single GPU, synthetic batch (no env, no RLinf Ray workers).
  * Show-o2-1.5B as the policy backbone; last 256 vocab IDs reused as
    discrete action bins via ``mirage.policy.ActionTokenizer``.
  * Group size N=4 fake trajectories per "prompt"; binary task-success reward
    sampled from Bernoulli(0.5) so advantages are non-zero.
  * No KL penalty, no PPO clipping (we want to surface NaN / shape bugs, not
    test the full PPO/GRPO loss surface yet — that's Phase 2).

Gate: prints ``"PHASE 1.4 PASSED"`` if loss is finite, gradient norm is
finite, ``optimizer.step()`` completes. Any NaN / inf / runtime error fails.

Run via: ``bash scripts/phase1_4_grpo_step.sh``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

# Show-o2 source on the cluster.
SHOWO2_PATH = Path("/home/mzh1800/Show-o-repo/show-o2")
sys.path.insert(0, str(SHOWO2_PATH))

# MIRAGE package (also on the cluster — we git-pull it into /home/mzh1800/MIRAGE).
MIRAGE_PATH = Path("/home/mzh1800/MIRAGE")
sys.path.insert(0, str(MIRAGE_PATH))

from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive                   # noqa: E402
from models.misc import get_text_tokenizer                                       # noqa: E402
from utils import get_hyper_params, path_to_llm_name                             # noqa: E402
from datasets.utils import image_transform                                       # noqa: E402

from mirage.policy import ActionTokenizer                                        # noqa: E402

CONFIG_PATH = Path("/home/mzh1800/MIRAGE/configs/showo2_smoke.yaml")
DEMO_IMAGE = SHOWO2_PATH / "docs/mmu/pexels-pixabay-207983.jpg"

# LIBERO conventions (matches RLinf's openvla_oft config).
ACTION_DIM = 7
NUM_ACTION_CHUNKS = 8
NUM_ACTION_TOKENS = ACTION_DIM * NUM_ACTION_CHUNKS    # 56
GROUP_SIZE = 4                                        # GRPO group size
TASK_DESCRIPTION = "pick up the red block"


def build_input_embeds(model, vae, text_tokenizer, showo_token_ids, config, device,
                       weight_dtype, image_path, task_text):
    """Reproduce the embedding pipeline from inference_mmu.py / phase1_0."""
    img = Image.open(image_path).convert("RGB")
    img = image_transform(img, resolution=config.dataset.preprocessing.resolution).to(device).unsqueeze(0)
    image_latents = vae.sample(img.unsqueeze(2)).squeeze(2).to(weight_dtype)

    image_embeds_und = model.image_embedder_und(image_latents)
    image_embeds_gen = model.image_embedder_gen(image_latents)
    image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
    image_embeds_und = model.und_trans(image_embeds_und)["last_hidden_state"]
    image_embeds = model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))

    sys_prompt_ids = text_tokenizer(
        "system\nYou are a helpful assistant.<|im_end|>",
        add_special_tokens=False,
    )["input_ids"]
    role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    role_b = text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    q_ids = text_tokenizer(task_text, add_special_tokens=False).input_ids

    text_tokens_a = torch.tensor(
        [showo_token_ids["bos_id"]] + sys_prompt_ids + role_a, device=device
    )[None, :]
    text_tokens_b = torch.tensor(
        [showo_token_ids["boi_id"], showo_token_ids["eoi_id"]] + q_ids + role_b,
        device=device,
    )[None, :]
    text_embeds_a = model.showo.get_input_embeddings()(text_tokens_a)
    text_embeds_b = model.showo.get_input_embeddings()(text_tokens_b)

    if config.model.showo.add_time_embeds:
        time_embeds = model.time_embed(torch.tensor([[1.0]], device=device), text_embeds_a.dtype)
        if hasattr(model, "time_embed_proj"):
            time_embeds = model.time_embed_proj(time_embeds)
        prefix_embeds = torch.cat(
            [text_embeds_a, text_embeds_b[:, :1], time_embeds, image_embeds, text_embeds_b[:, 1:]],
            dim=1,
        ).to(weight_dtype)
        _, num_mmu_image_tokens, *_ = get_hyper_params(config, text_tokenizer, showo_token_ids)
        modality_positions = torch.tensor(
            [text_tokens_a.shape[1] + 2, num_mmu_image_tokens], device=device,
        )[None, None, :]
    else:
        prefix_embeds = torch.cat(
            [text_embeds_a, text_embeds_b[:, :1], image_embeds, text_embeds_b[:, 1:]],
            dim=1,
        ).to(weight_dtype)
        _, num_mmu_image_tokens, *_ = get_hyper_params(config, text_tokenizer, showo_token_ids)
        modality_positions = torch.tensor(
            [text_tokens_a.shape[1] + 1, num_mmu_image_tokens], device=device,
        )[None, None, :]

    return prefix_embeds, modality_positions


def main() -> None:
    config = OmegaConf.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if config.model.weight_type == "bfloat16" else torch.float32
    torch.manual_seed(0)
    np.random.seed(0)

    # -- Load tokenizer + model + VAE -------------------------------------
    print("[1.4] loading tokenizer", flush=True)
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    vocab_size = len(text_tokenizer)

    print("[1.4] loading Wan-VAE", flush=True)
    vae = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_dtype,
        device=device,
    )

    print("[1.4] loading Show-o2-1.5B", flush=True)
    model = Showo2Qwen2_5.from_pretrained(
        config.model.showo.pretrained_model_path,
        use_safetensors=False,
    ).to(device).to(weight_dtype)
    # Train mode so .backward can flow.
    model.train()

    # -- Build the prompt-side embeddings (no grad needed yet) -----------
    print("[1.4] building prompt embeds (image + task description)", flush=True)
    with torch.no_grad():
        prefix_embeds, modality_positions = build_input_embeds(
            model, vae, text_tokenizer, showo_token_ids, config, device,
            weight_dtype, DEMO_IMAGE, TASK_DESCRIPTION,
        )
    # prefix_embeds: [1, L_prefix, H]

    # -- Sample synthetic action tokens (GROUP_SIZE rollouts) -------------
    # In a real rollout these would come from autoregressive generation;
    # here we fabricate them so we can test the GRPO math in one batch.
    action_tokenizer = ActionTokenizer(vocab_size=vocab_size, bins=256, hi_id=151642)
    print(f"[1.4] action token range: {action_tokenizer.action_token_id_range}", flush=True)

    synthetic_actions = np.random.uniform(
        -1.0, 1.0, size=(GROUP_SIZE, NUM_ACTION_TOKENS),
    )
    action_token_ids = torch.from_numpy(
        action_tokenizer.encode(synthetic_actions)
    ).to(device)                                            # [N, 56]
    action_embeds = model.showo.get_input_embeddings()(action_token_ids).to(weight_dtype)

    # Tile prefix across the group, then concat action embeds at the tail.
    prefix_batched = prefix_embeds.expand(GROUP_SIZE, -1, -1).contiguous()
    input_embeds = torch.cat([prefix_batched, action_embeds], dim=1)
    L_prefix = prefix_batched.shape[1]

    # Attention mask:
    #   omni_attn_mask_naive returns [B, 1, L, L] additive mask (0 attended,
    #   iinfo.min masked) when inverted=True.
    # Build a fresh [GROUP_SIZE, 1, L_total, L_total] additive mask that:
    #   - reproduces the omni prefix mask on the prefix block,
    #   - lets every action position attend to the full prefix,
    #   - causally attends to earlier action positions.
    L_total = input_embeds.shape[1]
    prefix_attn = omni_attn_mask_naive(
        B=1, LEN=L_prefix, modalities=modality_positions, device=device, inverted=True,
    )                                                       # [1, 1, L_prefix, L_prefix]
    neg = torch.iinfo(torch.long).min
    # Start fully unmasked, then set the prefix block from the omni mask and
    # paint masking on the upper triangle of the action block.
    attn = torch.zeros(GROUP_SIZE, 1, L_total, L_total, device=device, dtype=torch.long)
    attn[:, :, :L_prefix, :L_prefix] = prefix_attn.expand(GROUP_SIZE, -1, -1, -1)
    # Action rows attend freely to the prefix (left of their column), and
    # causally among themselves (upper-tri inside the action block is masked).
    action_causal = torch.triu(
        torch.full((NUM_ACTION_TOKENS, NUM_ACTION_TOKENS), neg, device=device, dtype=torch.long),
        diagonal=1,
    )
    attn[:, :, L_prefix:, L_prefix:] = action_causal
    # cast to model dtype so the additive mask plays well with bf16/fp16 paths.
    attn = attn.to(input_embeds.dtype)

    # -- Forward through the underlying Qwen LM ---------------------------
    print(f"[1.4] forward through model (input shape: {tuple(input_embeds.shape)})", flush=True)
    out = model.showo(
        inputs_embeds=input_embeds,
        attention_mask=attn,
    )
    logits = out.logits                                     # [N, L_total, V]
    # Action-token positions: predict token at position i from hidden at i-1.
    # action tokens live at positions [L_prefix, L_prefix+NUM_ACTION_TOKENS)
    # so we read logits at positions [L_prefix-1, L_prefix+NUM_ACTION_TOKENS-1)
    action_logits = logits[:, L_prefix - 1:L_prefix - 1 + NUM_ACTION_TOKENS, :].float()
    targets = action_token_ids                              # [N, 56]
    log_probs = F.log_softmax(action_logits, dim=-1)
    chosen_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)    # [N, 56]
    # Per-trajectory log-prob sum.
    traj_lp = chosen_lp.sum(dim=-1)                         # [N]
    print(f"[1.4] per-trajectory log-probs: {traj_lp.detach().float().cpu().tolist()}", flush=True)

    # -- Synthetic GRPO advantage ----------------------------------------
    rewards = torch.from_numpy(
        np.random.binomial(1, 0.5, size=GROUP_SIZE).astype(np.float32)
    ).to(device)
    print(f"[1.4] rewards: {rewards.cpu().tolist()}", flush=True)
    mean_r = rewards.mean()
    std_r = rewards.std().clamp_min(1e-6)
    advantage = (rewards - mean_r) / std_r                  # [N]
    # Policy gradient with no clipping, no KL.
    loss = -(advantage * traj_lp).mean()
    print(f"[1.4] GRPO loss: {loss.item():.6f}", flush=True)

    if not torch.isfinite(loss):
        raise RuntimeError(f"loss is not finite: {loss.item()}")

    # -- Backward + optimizer.step() -------------------------------------
    print("[1.4] backward + optimizer.step()", flush=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-6,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError(f"grad_norm is not finite: {grad_norm}")
    optimizer.step()
    print(f"[1.4] grad_norm: {grad_norm.item():.6f}", flush=True)

    print("=" * 60, flush=True)
    print("PHASE 1.4 PASSED", flush=True)
    print(f"  loss={loss.item():.6f}", flush=True)
    print(f"  grad_norm={grad_norm.item():.6f}", flush=True)
    print(f"  group_size={GROUP_SIZE}, action_tokens_per_traj={NUM_ACTION_TOKENS}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
