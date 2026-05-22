"""Phase 2B diagnostic — verify the LoRA-SFT'd model recovers training actions.

For 10 randomly drawn (image, action_chunk, task) examples from the LIBERO
demo dataset, greedy-decode the model's predicted action chunk and compare
to the ground-truth chunk. Prints:
  - per-dim MSE  (predicted vs GT, summed over 8 chunks)
  - the GT action chunk side-by-side with the predicted chunk
  - distribution of predicted token ids (to detect mode collapse)

If MSE is small (~0.05): SFT learned well, eval failure is distribution-shift
                        (init states, env dynamics). Proceed to GRPO.
If MSE is large (~0.5+): SFT didn't learn, regardless of CE loss curve.
                        Investigate forward path / LoRA wrapper / image flip.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

SHOWO2_PATH = Path("/home/mzh1800/Show-o-repo/show-o2")
sys.path.insert(0, str(SHOWO2_PATH))
sys.path.insert(0, str(Path("/home/mzh1800/MIRAGE")))

from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive  # noqa: E402
from models.misc import get_text_tokenizer  # noqa: E402
from utils import get_hyper_params, path_to_llm_name  # noqa: E402
from datasets.utils import image_transform  # noqa: E402

from mirage.policy import ActionTokenizer, ActionNormalizer  # noqa: E402

ACTION_STATS_PATH = "/nyx-storage1/hanliu/mirage_ckpts/action_stats.json"
from mirage.data import libero_spatial_dataset  # noqa: E402

CONFIG_PATH = Path("/home/mzh1800/MIRAGE/configs/showo2_smoke.yaml")
DATASET_DIR = Path("/nyx-storage1/hanliu/envs/mirage_venv/libero/libero/datasets")
CKPT = Path("/nyx-storage1/hanliu/mirage_ckpts/phase2b_lora_chunk1/sft_step_30000_lora")
NUM_CHUNKS = 1
ACTION_DIM = 7
NUM_PROBE = 10
TOTAL = NUM_CHUNKS * ACTION_DIM


def build_prefix(model, vae, tok, sids, cfg, device, dtype, img_np, task_text):
    pil = Image.fromarray(img_np).convert("RGB")  # already flipped by dataset
    img = image_transform(pil, resolution=cfg.dataset.preprocessing.resolution).to(device).unsqueeze(0)
    with torch.no_grad():
        latents = vae.sample(img.unsqueeze(2)).squeeze(2).to(dtype)
    e_und = model.image_embedder_und(latents)
    e_gen = model.image_embedder_gen(latents)
    e_und = e_und + model.position_embedding(model.image_position_ids)
    e_und = model.und_trans(e_und)["last_hidden_state"]
    image_embeds = model.fusion_proj(torch.cat([e_und, e_gen], dim=-1))

    sysp = tok("system\nYou are a helpful assistant.<|im_end|>", add_special_tokens=False)["input_ids"]
    role_a = tok("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    role_b = tok("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    q = tok(f"What action should the robot take to {task_text.lower()}?",
            add_special_tokens=False).input_ids

    t_a = torch.tensor([sids["bos_id"]] + sysp + role_a, device=device)[None, :]
    t_b = torch.tensor([sids["boi_id"], sids["eoi_id"]] + q + role_b, device=device)[None, :]
    e_a = model.showo.get_input_embeddings()(t_a)
    e_b = model.showo.get_input_embeddings()(t_b)

    _, n_mmu, *_ = get_hyper_params(cfg, tok, sids)
    if cfg.model.showo.add_time_embeds:
        te = model.time_embed(torch.tensor([[1.0]], device=device), e_a.dtype)
        if hasattr(model, "time_embed_proj"):
            te = model.time_embed_proj(te)
        pref = torch.cat([e_a, e_b[:, :1], te, image_embeds, e_b[:, 1:]], dim=1).to(dtype)
        mp = torch.tensor([t_a.shape[1] + 2, n_mmu], device=device)[None, None, :]
    else:
        pref = torch.cat([e_a, e_b[:, :1], image_embeds, e_b[:, 1:]], dim=1).to(dtype)
        mp = torch.tensor([t_a.shape[1] + 1, n_mmu], device=device)[None, None, :]
    return pref, mp


@torch.no_grad()
def greedy_sample(model, prefix, mp, atok, device, dtype):
    lo, hi = atok.action_token_id_range
    cur = prefix
    L0 = prefix.shape[1]
    pref_attn = omni_attn_mask_naive(B=1, LEN=L0, modalities=mp, device=device, inverted=True)
    sampled = []
    for _ in range(TOTAL):
        Lt = cur.shape[1]
        neg = torch.iinfo(torch.long).min
        attn = torch.zeros(1, 1, Lt, Lt, device=device, dtype=torch.long)
        attn[:, :, :L0, :L0] = pref_attn
        n = Lt - L0
        if n > 0:
            causal = torch.triu(torch.full((n, n), neg, device=device, dtype=torch.long), diagonal=1)
            attn[:, :, L0:, L0:] = causal
            attn[:, :, :L0, L0:] = neg
        attn = attn.to(dtype)
        out = model.showo(inputs_embeds=cur, attention_mask=attn)
        logits = out.logits[:, -1, :].float()
        masked = torch.full_like(logits, float("-inf"))
        masked[:, lo:hi + 1] = logits[:, lo:hi + 1]
        nid = int(masked.argmax(dim=-1).item())
        sampled.append(nid)
        next_emb = model.showo.get_input_embeddings()(
            torch.tensor([[nid]], device=device)).to(dtype)
        cur = torch.cat([cur, next_emb], dim=1)
    return np.asarray(sampled, dtype=np.int64)


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = OmegaConf.load(CONFIG_PATH)

    tok, sids = get_text_tokenizer(
        cfg.model.showo.llm_model_path, add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[cfg.model.showo.llm_model_path],
    )
    cfg.model.showo.llm_vocab_size = len(tok)
    V = len(tok)

    print("[diag] loading Wan-VAE", flush=True)
    vae = WanVAE(vae_pth=cfg.model.vae_model.pretrained_model_path, dtype=dtype, device=device)

    print("[diag] loading Show-o2 + LoRA", flush=True)
    model = Showo2Qwen2_5.from_pretrained(
        cfg.model.showo.pretrained_model_path, use_safetensors=False,
    ).to(device).to(dtype)
    from peft import PeftModel
    model.showo = PeftModel.from_pretrained(model.showo, str(CKPT), is_trainable=False)
    model = model.to(device).to(dtype)
    model.eval()

    atok = ActionTokenizer(vocab_size=V, bins=256, hi_id=151642)
    anorm = ActionNormalizer.from_json(ACTION_STATS_PATH)
    lo, hi = atok.action_token_id_range
    print(f"[diag] action token range = [{lo}, {hi}]  (bins=256, V={V})", flush=True)

    ds = libero_spatial_dataset(dataset_dir=DATASET_DIR, num_chunks=NUM_CHUNKS)
    print(f"[diag] dataset size = {len(ds)}", flush=True)

    rng = np.random.default_rng(0)
    probe_ids = rng.choice(len(ds), size=NUM_PROBE, replace=False)

    all_pred_tokens = []
    all_gt_tokens = []
    per_chunk_mse = []
    print("=" * 80)
    for k, idx in enumerate(probe_ids):
        ex = ds[int(idx)]
        gt_action = ex["action"].astype(np.float32)  # [C, 7] raw
        gt_action_norm = anorm.normalize(gt_action)
        gt_flat = gt_action_norm.reshape(-1)
        gt_tokens = atok.encode(gt_flat).astype(np.int64)  # [56]
        prefix, mp = build_prefix(model, vae, tok, sids, cfg, device, dtype,
                                  ex["image"], ex["task"])
        pred_tokens = greedy_sample(model, prefix, mp, atok, device, dtype)
        pred_action_norm = atok.decode(pred_tokens).reshape(NUM_CHUNKS, ACTION_DIM).astype(np.float32)
        pred_action = anorm.denormalize(pred_action_norm)

        mse = float(np.mean((pred_action - gt_action) ** 2))
        per_chunk_mse.append(mse)
        all_pred_tokens.extend(pred_tokens.tolist())
        all_gt_tokens.extend(gt_tokens.tolist())
        token_match = int((pred_tokens == gt_tokens).sum())
        print(f"[probe {k} idx={idx} task='{ex['task'][:40]}'] "
              f"token_match={token_match}/56  chunk_mse={mse:.4f}")
        print(f"  GT  chunk0: {gt_action[0]}")
        print(f"  PRED chunk0: {pred_action[0]}")
        print(f"  GT  chunk4: {gt_action[4]}")
        print(f"  PRED chunk4: {pred_action[4]}")

    print("=" * 80)
    print(f"MEAN chunk_mse = {np.mean(per_chunk_mse):.4f}")
    print(f"MEDIAN chunk_mse = {np.median(per_chunk_mse):.4f}")

    # Mode collapse check
    pred_counter = Counter(all_pred_tokens)
    top_pred = pred_counter.most_common(5)
    print(f"Top-5 predicted tokens: {top_pred}")
    print(f"Unique predicted tokens: {len(pred_counter)} (out of {hi-lo+1} possible)")
    gt_counter = Counter(all_gt_tokens)
    top_gt = gt_counter.most_common(5)
    print(f"Top-5 GT tokens: {top_gt}")
    print(f"Unique GT tokens: {len(gt_counter)}")


if __name__ == "__main__":
    main()
