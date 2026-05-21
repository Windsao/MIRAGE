"""Phase 2B SFT: teach Show-o2 the action-token language on LIBERO demos.

Minimal teacher-forcing SFT loop:
  obs image -> Show-o2 multimodal embed pipeline
            -> concatenate the 7 *target* action-token embeddings at the tail
            -> single forward through model.showo
            -> cross-entropy on logits at the 7 action positions
            -> AdamW step

We deliberately *do not* use LoRA in this smoke — Show-o2-1.5B + AdamW fits in
~30 GB on an A40, and the full-model gradient lets us debug whether the LM
head can pick up the action token distribution at all.

Configurable via CLI:
  --num-tasks   how many LIBERO-Spatial tasks to draw from (default 10)
  --steps       number of training steps (default 200; ~30-45 min on 1 A40)
  --batch-size  examples per step (default 1; A40 mem-tight beyond 2)
  --lr          learning rate (default 1e-5)
  --save-dir    output checkpoint directory

Outputs:
  <save-dir>/sft_step_<step>.pt        # actor state_dict snapshots
  <save-dir>/loss_curve.csv            # one line per step: step, loss
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

SHOWO2_PATH = Path("/home/mzh1800/Show-o-repo/show-o2")
sys.path.insert(0, str(SHOWO2_PATH))
sys.path.insert(0, str(Path("/home/mzh1800/MIRAGE")))

from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive                   # noqa: E402
from models.misc import get_text_tokenizer                                       # noqa: E402
from utils import get_hyper_params, path_to_llm_name                             # noqa: E402
from datasets.utils import image_transform                                       # noqa: E402

from mirage.policy import ActionTokenizer                                        # noqa: E402
from mirage.data import libero_spatial_dataset                                   # noqa: E402

CONFIG_PATH = Path("/home/mzh1800/MIRAGE/configs/showo2_smoke.yaml")
DATASET_DIR = Path("/nyx-storage1/hanliu/envs/mirage_venv/libero/libero/datasets")
ACTION_DIM = 7


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tasks", type=int, default=10)
    ap.add_argument("--max-steps-per-demo", type=int, default=None,
                    help="cap each demo at this many timesteps (None = use all)")
    ap.add_argument("--num-chunks", type=int, default=8,
                    help="action chunk size (predict K consecutive 7-D actions per inference)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="for LoRA prefer 1e-4; for full ft prefer 1e-5")
    ap.add_argument("--lora-rank", type=int, default=0,
                    help=">0 enables LoRA on Qwen2 LM only (memory-friendly); 0 = full ft")
    ap.add_argument("--save-dir", type=str,
                    default="/nyx-storage1/hanliu/mirage_ckpts/phase2b_sft")
    ap.add_argument("--save-every", type=int, default=999999)
    return ap.parse_args()


def build_components(device, weight_dtype, lora_rank: int = 0):
    config = OmegaConf.load(CONFIG_PATH)
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    vocab_size = len(text_tokenizer)

    print("[2b] loading Wan-VAE", flush=True)
    vae = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_dtype,
        device=device,
    )
    print("[2b] loading Show-o2-1.5B", flush=True)
    model = Showo2Qwen2_5.from_pretrained(
        config.model.showo.pretrained_model_path,
        use_safetensors=False,
    ).to(device).to(weight_dtype)
    if lora_rank > 0:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        # Wrap only the Qwen2 LM. Image-side modules (image_embedder_und/gen,
        # und_trans, fusion_proj, position_embedding) stay frozen.
        model.showo = get_peft_model(model.showo, lora_cfg)
        # Freeze everything else explicitly.
        for n, p in model.named_parameters():
            if "showo" not in n:
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[2b] LoRA rank={lora_rank}  trainable={trainable/1e6:.1f}M / {total/1e6:.0f}M "
              f"({100 * trainable / total:.2f}%)", flush=True)
    model.train()
    action_tokenizer = ActionTokenizer(vocab_size=vocab_size, bins=256, hi_id=151642)
    return config, text_tokenizer, showo_token_ids, vae, model, action_tokenizer


def build_one_input(model, vae, text_tokenizer, showo_token_ids, config,
                    device, weight_dtype, image_np, task_text, action_token_ids):
    """Return (input_embeds, attn, L_prefix) for one example.

    input_embeds = [prompt_tokens..., A action token embeddings]
                   where A = len(action_token_ids) = ACTION_DIM * num_chunks
    attn        = additive omni mask on prefix, causal on actions
    L_prefix    = start of the action block; logits at
                  [L_prefix-1 .. L_prefix-1+A) supervise the A action tokens.
    """
    # ---- image -> embeds (no grad on VAE; model.image_embedder_* IS trained)
    img_pil = Image.fromarray(image_np).convert("RGB")
    img = image_transform(img_pil, resolution=config.dataset.preprocessing.resolution).to(device).unsqueeze(0)
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

    L_prefix = prefix_embeds.shape[1]
    A = int(action_token_ids.shape[0])
    action_embeds = model.showo.get_input_embeddings()(
        action_token_ids[None, :]
    ).to(weight_dtype)                                              # [1, A, H]
    input_embeds = torch.cat([prefix_embeds, action_embeds], dim=1)
    L_total = input_embeds.shape[1]

    # Build 4D additive attention mask.
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
    return input_embeds, attn, L_prefix


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    # ---- dataset
    print(
        f"[2b] loading LIBERO-Spatial dataset (cap={args.max_steps_per_demo} "
        f"steps/demo, num_chunks={args.num_chunks})",
        flush=True,
    )
    ds = libero_spatial_dataset(
        dataset_dir=DATASET_DIR,
        max_steps_per_demo=args.max_steps_per_demo,
        num_chunks=args.num_chunks,
    )
    print(f"[2b] dataset size = {len(ds)} (image, action_chunk, task) triples", flush=True)
    total_action_tokens = ACTION_DIM * args.num_chunks
    print(f"[2b] action tokens per example = {total_action_tokens}", flush=True)

    # Plain DataLoader; batch_size=1 since each example owns its tensors.
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    config, text_tokenizer, showo_token_ids, vae, model, action_tokenizer = (
        build_components(device, weight_dtype, lora_rank=args.lora_rank)
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = open(save_dir / "loss_curve.csv", "w", newline="")
    csv_w = csv.writer(loss_csv)
    csv_w.writerow(["step", "loss", "wall_s"])

    step = 0
    running = 0.0
    t0 = time.time()
    iter_loader = iter(loader)
    while step < args.steps:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)

        image_np = batch["image"][0].numpy()
        action_np = batch["action"][0].numpy()                          # [C, 7] or [7]
        task_text = batch["task"][0]
        # Flatten time-major: dims of chunk 0, then chunk 1, ... -> [C*7]
        flat = action_np.reshape(-1) if action_np.ndim > 1 else action_np
        action_token_ids = torch.from_numpy(
            action_tokenizer.encode(flat)
        ).to(device).long()                                            # [C*7]
        A = int(action_token_ids.shape[0])

        input_embeds, attn, L_prefix = build_one_input(
            model, vae, text_tokenizer, showo_token_ids, config,
            device, weight_dtype, image_np, task_text, action_token_ids,
        )
        out = model.showo(inputs_embeds=input_embeds, attention_mask=attn)
        logits = out.logits                                            # [1, L_total, V]
        # logits at positions [L_prefix-1 .. L_prefix-1+A) supervise the
        # A action token labels (next-token prediction).
        pred_logits = logits[:, L_prefix - 1:L_prefix - 1 + A, :].float()
        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            action_token_ids,
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        running += float(loss.item())
        step += 1
        wall = time.time() - t0
        csv_w.writerow([step, float(loss.item()), wall])

        if step % 10 == 0 or step == 1:
            avg = running / min(step, 10)
            print(
                f"[2b] step {step:04d}/{args.steps} "
                f"loss={float(loss.item()):.4f} avg10={avg:.4f} "
                f"gn={float(gn):.2f} t={wall:.0f}s",
                flush=True,
            )
            running = 0.0

        if step % args.save_every == 0 or step == args.steps:
            if args.lora_rank > 0:
                ckpt_dir = save_dir / f"sft_step_{step}_lora"
                print(f"[2b] saving LoRA adapter to {ckpt_dir}", flush=True)
                # peft's save_pretrained on model.showo writes adapter + config only.
                model.showo.save_pretrained(str(ckpt_dir))
                torch.save({"step": step, "args": vars(args)},
                            ckpt_dir / "meta.pt")
            else:
                ckpt_path = save_dir / f"sft_step_{step}.pt"
                print(f"[2b] saving full state to {ckpt_path}", flush=True)
                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )

    loss_csv.close()
    print(f"[2b] DONE. final loss={loss.item():.4f}, total wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
