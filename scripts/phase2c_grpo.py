"""Phase 2C — real-env GRPO smoke on LIBERO-Spatial.

Loads an SFT'd Show-o2 checkpoint, samples ``--group-size`` trajectories per
task, computes group-relative advantage from binary task success, and runs
``--num-steps`` GRPO updates. Eval scaffold is intentionally minimal: this is
a *smoke* to confirm the math works on real rollouts, not a full benchmark
run.

Per GRPO step on 1 task:
    for k in range(group_size):
        reset env; roll out up to ``--max-steps`` (chunked inference)
        record action token ids, prefix tensors, reward (final success only)
    advantage = (reward - mean) / std    # group-relative
    forward each rollout's stored prefix+tokens through model,
      compute action-token log-probs and policy-gradient loss
    backward + optimizer.step()

Caveats this smoke deliberately ignores (Phase 2D+):
  * no KL penalty or PPO clip (vanilla policy gradient)
  * no LoRA; full-model finetune (memory-heavy but simple)
  * single task per GRPO step (no inter-task batching)
  * group rewards come from a single rollout each (no per-step credit)

CLI:
  --ckpt          required; SFT'd state_dict
  --task-idx      LIBERO-Spatial task index to run on (default 0)
  --group-size    N trajectories per GRPO step (default 4)
  --num-steps     GRPO update steps (default 5)
  --max-steps     env steps per trajectory (default 120)
  --lr            optimizer lr (default 1e-6)
  --temperature   rollout sampling temperature (default 1.0)
  --num-chunks    action chunk size (default 8; matches SFT)
  --save-dir      output dir; saves last checkpoint and a JSON summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

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

CONFIG_PATH = Path("/home/mzh1800/MIRAGE/configs/showo2_smoke.yaml")
ACTION_DIM = 7


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--num-chunks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--save-dir", type=str,
                    default="/nyx-storage1/hanliu/mirage_ckpts/phase2c_grpo")
    return ap.parse_args()


# ----- prefix builder (no grad; rollout-time inference path) -----------

def build_prefix_embeds_no_grad(model, vae, text_tokenizer, showo_token_ids,
                                config, device, weight_dtype, image_np, task_text):
    with torch.no_grad():
        return build_prefix_embeds(model, vae, text_tokenizer, showo_token_ids,
                                   config, device, weight_dtype, image_np, task_text)


def build_prefix_embeds(model, vae, text_tokenizer, showo_token_ids, config,
                        device, weight_dtype, image_np, task_text):
    pil = Image.fromarray(image_np).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
    img = image_transform(pil, resolution=config.dataset.preprocessing.resolution).to(device).unsqueeze(0)
    with torch.no_grad():
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


# ----- autoregressive sampler (collects per-token log-probs) -----------

@torch.no_grad()
def sample_action_chunks(model, prefix_embeds, modality_positions, action_tokenizer,
                          device, weight_dtype, num_chunks: int, temperature: float):
    lo, hi = action_tokenizer.action_token_id_range
    cur_embeds = prefix_embeds
    L_prefix = prefix_embeds.shape[1]
    total = num_chunks * ACTION_DIM
    prefix_attn = omni_attn_mask_naive(
        B=1, LEN=L_prefix, modalities=modality_positions, device=device, inverted=True,
    )
    sampled: list[int] = []

    for _ in range(total):
        L_total = cur_embeds.shape[1]
        neg = torch.iinfo(torch.long).min
        attn = torch.zeros(1, 1, L_total, L_total, device=device, dtype=torch.long)
        attn[:, :, :L_prefix, :L_prefix] = prefix_attn
        n = L_total - L_prefix
        if n > 0:
            causal = torch.triu(
                torch.full((n, n), neg, device=device, dtype=torch.long),
                diagonal=1,
            )
            attn[:, :, L_prefix:, L_prefix:] = causal
        attn = attn.to(weight_dtype)

        out = model.showo(inputs_embeds=cur_embeds, attention_mask=attn)
        next_logits = out.logits[:, -1, :].float()
        masked = torch.full_like(next_logits, float("-inf"))
        masked[:, lo:hi + 1] = next_logits[:, lo:hi + 1]
        if temperature <= 0:
            probs = torch.zeros_like(next_logits)
            probs[:, masked.argmax(dim=-1)] = 1.0
            next_id = int(masked.argmax(dim=-1).item())
        else:
            probs = F.softmax(masked / max(temperature, 1e-6), dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        sampled.append(next_id)

        next_emb = model.showo.get_input_embeddings()(
            torch.tensor([[next_id]], device=device)
        ).to(weight_dtype)
        cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)

    tokens = np.asarray(sampled, dtype=np.int64)
    decoded = action_tokenizer.decode(tokens).astype(np.float32)
    chunks = decoded.reshape(num_chunks, ACTION_DIM)
    return chunks, tokens


# ----- training-time log-prob (with grad) over a stored prefix + token seq -----

def teacher_force_logprobs(model, prefix_embeds, modality_positions,
                            target_token_ids, action_tokenizer,
                            device, weight_dtype):
    """Returns sum of log-probs of the target action tokens given prefix.

    Restrict the softmax to action-bin token IDs to match the rollout-time
    policy (constrained sampling).
    """
    lo, hi = action_tokenizer.action_token_id_range
    L_prefix = prefix_embeds.shape[1]
    A = int(target_token_ids.shape[0])
    target_embeds = model.showo.get_input_embeddings()(
        target_token_ids[None, :]
    ).to(weight_dtype)
    input_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)
    L_total = input_embeds.shape[1]

    prefix_attn = omni_attn_mask_naive(
        B=1, LEN=L_prefix, modalities=modality_positions, device=device, inverted=True,
    )
    neg = torch.iinfo(torch.long).min
    attn = torch.zeros(1, 1, L_total, L_total, device=device, dtype=torch.long)
    attn[:, :, :L_prefix, :L_prefix] = prefix_attn
    causal = torch.triu(
        torch.full((A, A), neg, device=device, dtype=torch.long),
        diagonal=1,
    )
    attn[:, :, L_prefix:, L_prefix:] = causal
    attn = attn.to(weight_dtype)

    out = model.showo(inputs_embeds=input_embeds, attention_mask=attn)
    logits = out.logits[:, L_prefix - 1:L_prefix - 1 + A, :].float()      # [1, A, V]
    # Restrict softmax to action-bin IDs (match rollout-time policy).
    masked = torch.full_like(logits, float("-inf"))
    masked[..., lo:hi + 1] = logits[..., lo:hi + 1]
    log_probs = F.log_softmax(masked, dim=-1)
    chosen = log_probs.gather(-1, target_token_ids.view(1, A, 1)).squeeze(-1)  # [1, A]
    return chosen.sum(dim=-1).squeeze(0)                                       # scalar


def rollout_one(env, model, vae, text_tokenizer, showo_token_ids, config,
                device, weight_dtype, action_tokenizer, task_text,
                num_chunks: int, max_steps: int, temperature: float, init_state=None):
    """One rollout. Returns
        (success, total_steps, chunk_prefixes, chunk_modality_positions,
         chunk_action_token_ids).
    The lists are aligned and let us recompute log-probs on the saved
    (prefix, target_tokens) pairs for the GRPO update.
    """
    env.reset()
    if init_state is not None:
        env.set_init_state(init_state)
    obs = None
    for _ in range(10):
        zero = np.zeros(7, dtype=np.float32)
        zero[-1] = -1.0
        obs, _r, _d, _info = env.step(zero)
    prefixes: list[torch.Tensor] = []
    mod_positions: list[torch.Tensor] = []
    action_token_seqs: list[torch.Tensor] = []
    success = False
    info = {}
    total_steps = 0
    reward = 0.0
    while total_steps < max_steps:
        prefix_embeds, modality_positions = build_prefix_embeds_no_grad(
            model, vae, text_tokenizer, showo_token_ids, config,
            device, weight_dtype, obs["agentview_image"], task_text,
        )
        chunks, tokens = sample_action_chunks(
            model, prefix_embeds, modality_positions, action_tokenizer,
            device, weight_dtype, num_chunks, temperature=temperature,
        )
        prefixes.append(prefix_embeds.detach())
        mod_positions.append(modality_positions.detach())
        action_token_seqs.append(torch.from_numpy(tokens).to(device).long())

        done = False
        for k in range(num_chunks):
            if total_steps >= max_steps:
                break
            act = chunks[k]
            if not np.all(np.isfinite(act)):
                act = np.nan_to_num(act, nan=0.0)
            obs, reward, done, info = env.step(act.astype(np.float32))
            total_steps += 1
            if done:
                break
        if done:
            success = bool(info.get("success", reward > 0))
            if not success and reward > 0:
                success = True
            break
    return success, total_steps, prefixes, mod_positions, action_token_seqs


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16
    torch.manual_seed(0)
    np.random.seed(0)

    config = OmegaConf.load(CONFIG_PATH)
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    vocab_size = len(text_tokenizer)

    print("[2c] loading Wan-VAE", flush=True)
    vae = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_dtype,
        device=device,
    )
    print(f"[2c] loading Show-o2 + SFT ckpt {args.ckpt}", flush=True)
    model = Showo2Qwen2_5.from_pretrained(
        config.model.showo.pretrained_model_path,
        use_safetensors=False,
    ).to(device).to(weight_dtype)
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir() and (ckpt_path / "adapter_config.json").exists():
        from peft import PeftModel
        model.showo = PeftModel.from_pretrained(
            model.showo, str(ckpt_path), is_trainable=True,
        )
        model = model.to(device).to(weight_dtype)
        # Freeze non-LoRA params (they were frozen during SFT).
        for n, p in model.named_parameters():
            if "showo" not in n or "lora_" not in n.lower():
                if "lora_" not in n.lower():
                    p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[2c] LoRA mode: trainable={trainable/1e6:.1f}M", flush=True)
    else:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
    model.train()

    action_tokenizer = ActionTokenizer(vocab_size=vocab_size, bins=256, hi_id=151642)

    bd = benchmark.get_benchmark_dict()
    spatial = bd["libero_spatial"]()
    t = spatial.get_task(args.task_idx)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    print(f"[2c] task {args.task_idx}: {t.language}", flush=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step_summaries: list[dict] = []
    for grpo_step in range(args.num_steps):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        successes: list[int] = []
        prefixes_all: list[list[torch.Tensor]] = []
        modp_all: list[list[torch.Tensor]] = []
        tokens_all: list[list[torch.Tensor]] = []
        init_states = spatial.get_task_init_states(args.task_idx)
        for k in range(args.group_size):
            t0 = time.time()
            init_state = init_states[k % len(init_states)]
            succ, n_steps, prefs, modps, toks = rollout_one(
                env, model, vae, text_tokenizer, showo_token_ids, config,
                device, weight_dtype, action_tokenizer, t.language,
                args.num_chunks, args.max_steps, args.temperature,
                init_state=init_state,
            )
            successes.append(int(succ))
            prefixes_all.append(prefs)
            modp_all.append(modps)
            tokens_all.append(toks)
            print(
                f"[2c] step {grpo_step} rollout {k}: success={succ} steps={n_steps} "
                f"wall={time.time()-t0:.0f}s",
                flush=True,
            )
        env.close()

        rewards = torch.tensor(successes, dtype=torch.float32, device=device)
        mean_r = rewards.mean()
        std_r = rewards.std().clamp_min(1e-6)
        advantages = (rewards - mean_r) / std_r                         # [N]
        if torch.allclose(rewards, rewards[0]):
            print("[2c] group rewards identical; advantage=0, no learning signal this step", flush=True)

        # Recompute log-probs with grad for each (prefix, tokens) and apply
        # policy-gradient loss weighted by per-rollout advantage.
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        n_inferences = 0
        for k in range(args.group_size):
            adv_k = advantages[k].item()
            if adv_k == 0:
                continue
            traj_lp = torch.zeros((), device=device, dtype=torch.float32)
            for prefix_embeds, mod_pos, tokens in zip(
                prefixes_all[k], modp_all[k], tokens_all[k],
            ):
                lp = teacher_force_logprobs(
                    model, prefix_embeds, mod_pos, tokens, action_tokenizer,
                    device, weight_dtype,
                )
                traj_lp = traj_lp + lp
                n_inferences += 1
            total_loss = total_loss + (-adv_k * traj_lp)
        total_loss = total_loss / max(n_inferences, 1)

        optim.zero_grad(set_to_none=True)
        if total_loss.requires_grad:
            total_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        else:
            gn = torch.tensor(0.0)
        optim.step()

        success_rate = float(np.mean(successes))
        print(
            f"[2c] === step {grpo_step}: success={sum(successes)}/{len(successes)} "
            f"({success_rate:.2f}) loss={float(total_loss):.4f} gn={float(gn):.2f} ===",
            flush=True,
        )
        step_summaries.append({
            "step": grpo_step,
            "successes": successes,
            "success_rate": success_rate,
            "loss": float(total_loss),
            "grad_norm": float(gn),
        })

    # save final state + summary. If LoRA was used, save adapter dir so eval can
    # auto-detect; else save full state_dict as a .pt.
    from peft import PeftModel
    if isinstance(model.showo, PeftModel):
        final_dir = save_dir / f"grpo_step_{args.num_steps}_lora"
        print(f"[2c] saving LoRA adapter to {final_dir}", flush=True)
        model.showo.save_pretrained(str(final_dir))
        torch.save({"step": args.num_steps, "args": vars(args)},
                   final_dir / "meta.pt")
        final_ckpt = final_dir
    else:
        final_ckpt = save_dir / "grpo_final.pt"
        torch.save({"step": args.num_steps, "model_state": model.state_dict(),
                    "args": vars(args)}, final_ckpt)
    with open(save_dir / "summary.json", "w") as f:
        json.dump({"task": t.language, "group_size": args.group_size,
                   "num_steps": args.num_steps, "steps": step_summaries}, f, indent=2)
    print(f"[2c] DONE. saved {final_ckpt}", flush=True)


if __name__ == "__main__":
    main()
