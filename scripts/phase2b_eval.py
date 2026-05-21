"""Phase 2B eval — closed-loop success rate of an SFT'd Show-o2 in LIBERO-Spatial.

For each of N tasks, run M trials:
    reset env
    while not done and steps < max_steps:
        obs image -> Show-o2 (with SFT'd LM head)
                  -> constrained sampling of 7 action tokens from action-bin range
                  -> ActionTokenizer.decode -> 7-D action
                  -> env.step
    record (task_id, trial, success)

Then print per-task and overall success rate.

CLI:
  --ckpt          path to SFT'd state_dict (.pt). Required to be informative.
  --num-tasks     how many LIBERO-Spatial tasks to evaluate (default 10).
  --trials-per-task   trials per task (default 3 for smoke; 10+ for paper).
  --max-steps     max env steps per trial (default 200).
  --temperature   sampling temperature (default 1.0; 0.0 for greedy).
  --output        JSON file to write summary into.

Result: writes <output> with per-task + overall success rate and prints the
same summary to stdout. Returns exit code 0.
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
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Either a .pt full-state checkpoint, or a directory holding a peft adapter.")
    ap.add_argument("--num-tasks", type=int, default=10)
    ap.add_argument("--trials-per-task", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--num-chunks", type=int, default=8,
                    help="actions per inference (matches SFT --num-chunks)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--output", type=str,
                    default="/home/mzh1800/MIRAGE/logs/phase2b_eval.json")
    return ap.parse_args()


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


@torch.no_grad()
def sample_action_chunks(model, prefix_embeds, modality_positions, action_tokenizer,
                          device, weight_dtype, num_chunks: int,
                          temperature: float = 1.0):
    """Autoregressively sample `num_chunks * ACTION_DIM` action tokens.

    Returns
    -------
    chunks : np.ndarray  shape [num_chunks, ACTION_DIM]
        Decoded continuous actions (NaN where the model emitted a non-action token).
    tokens : np.ndarray  shape [num_chunks * ACTION_DIM]
        Raw sampled token ids.
    """
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
    decoded = action_tokenizer.decode(tokens).astype(np.float32)        # [total]
    chunks = decoded.reshape(num_chunks, ACTION_DIM)
    return chunks, tokens


def run_trial(env, model, vae, text_tokenizer, showo_token_ids, config,
              device, weight_dtype, action_tokenizer, task_text, max_steps,
              num_chunks, temperature, init_state=None):
    env.reset()
    if init_state is not None:
        env.set_init_state(init_state)
    # Stabilize the initial config with a few zero / gripper-open dummy steps
    # (matches LIBERO/OpenVLA-OFT convention).
    obs = None
    for _ in range(10):
        zero = np.zeros(7, dtype=np.float32)
        zero[-1] = -1.0  # gripper open
        obs, _r, _d, _info = env.step(zero)
    success = False
    total_steps = 0
    while total_steps < max_steps:
        prefix_embeds, modality_positions = build_prefix_embeds(
            model, vae, text_tokenizer, showo_token_ids, config,
            device, weight_dtype, obs["agentview_image"], task_text,
        )
        chunks, _ = sample_action_chunks(
            model, prefix_embeds, modality_positions, action_tokenizer,
            device, weight_dtype, num_chunks, temperature=temperature,
        )
        # Replay decoded chunks back-to-back as an open-loop horizon.
        done = False
        info = {}
        reward = 0.0
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
    return success, total_steps


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16
    config = OmegaConf.load(CONFIG_PATH)

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    vocab_size = len(text_tokenizer)

    print("[2b-eval] loading Wan-VAE", flush=True)
    vae = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_dtype,
        device=device,
    )
    print(f"[2b-eval] loading Show-o2-1.5B (ckpt={args.ckpt})", flush=True)
    model = Showo2Qwen2_5.from_pretrained(
        config.model.showo.pretrained_model_path,
        use_safetensors=False,
    ).to(device).to(weight_dtype)
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_dir() and (ckpt_path / "adapter_config.json").exists():
            from peft import PeftModel
            model.showo = PeftModel.from_pretrained(
                model.showo, str(ckpt_path), is_trainable=False,
            )
            model = model.to(device).to(weight_dtype)
            meta = ckpt_path / "meta.pt"
            step = torch.load(meta)["step"] if meta.exists() else "?"
            print(f"[2b-eval] loaded LoRA adapter from step={step}", flush=True)
        else:
            state = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(state["model_state"])
            print(f"[2b-eval] loaded full state from step={state.get('step')}", flush=True)
    model.eval()
    action_tokenizer = ActionTokenizer(vocab_size=vocab_size, bins=256, hi_id=151642)

    bd = benchmark.get_benchmark_dict()
    spatial = bd["libero_spatial"]()
    n_tasks = min(args.num_tasks, spatial.n_tasks)
    per_task: dict[int, dict] = {}
    overall_success = 0
    overall_trials = 0
    t_start = time.time()

    for task_idx in range(n_tasks):
        t = spatial.get_task(task_idx)
        bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)

        succ_list: list[int] = []
        len_list: list[int] = []
        init_states = spatial.get_task_init_states(task_idx)
        for trial in range(args.trials_per_task):
            init_state = init_states[trial % len(init_states)]
            success, steps = run_trial(
                env, model, vae, text_tokenizer, showo_token_ids, config,
                device, weight_dtype, action_tokenizer,
                t.language, args.max_steps, args.num_chunks, args.temperature,
                init_state=init_state,
            )
            succ_list.append(int(success))
            len_list.append(steps)
            print(
                f"[2b-eval] task {task_idx} trial {trial}: success={success} steps={steps}",
                flush=True,
            )
        sr = float(np.mean(succ_list))
        per_task[task_idx] = {
            "language": t.language,
            "success_rate": sr,
            "mean_steps": float(np.mean(len_list)),
            "n_trials": args.trials_per_task,
        }
        overall_success += int(sum(succ_list))
        overall_trials += len(succ_list)
        env.close()
        print(f"[2b-eval] task {task_idx} success rate: {sr:.2f} (n={len(succ_list)})", flush=True)

    overall_sr = overall_success / max(overall_trials, 1)
    summary = {
        "overall_success_rate": overall_sr,
        "overall_trials": overall_trials,
        "per_task": per_task,
        "ckpt": args.ckpt,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "wall_seconds": time.time() - t_start,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(summary, fp, indent=2)

    print("=" * 60, flush=True)
    print(f"OVERALL success: {overall_success}/{overall_trials} = {overall_sr:.2%}", flush=True)
    print(f"Per-task:", flush=True)
    for k, v in per_task.items():
        print(f"  task {k}: {v['success_rate']:.2f}   {v['language'][:60]}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
