"""Image-conditioning probe: does the chunk=1 30k SFT model use the input image?

For each of N held-out training examples, predict ONLY the first action token
(7 distinct images, 7 distinct actions). If predictions differ across images,
the model conditions on visual content. If they're identical, image input is
being ignored.

Also computes image-embed L2 distances to verify the image encoder produces
distinct embeddings (rules out "all images embed to same vector" bug).
"""
from __future__ import annotations
import sys
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

from mirage.policy import ActionTokenizer  # noqa: E402
from mirage.data import libero_spatial_dataset  # noqa: E402

CONFIG_PATH = Path("/home/mzh1800/MIRAGE/configs/showo2_smoke.yaml")
DATASET_DIR = Path("/nyx-storage1/hanliu/envs/mirage_venv/libero/libero/datasets")
CKPT = Path("/nyx-storage1/hanliu/mirage_ckpts/phase2b_lora_v6_normaction/sft_step_30000_lora")
N = 10


def build_prefix(model, vae, tok, sids, cfg, device, dtype, img_np, task_text):
    pil = Image.fromarray(img_np).convert("RGB")
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
    return pref, mp, image_embeds


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

    print("[probe] loading Wan-VAE", flush=True)
    vae = WanVAE(vae_pth=cfg.model.vae_model.pretrained_model_path, dtype=dtype, device=device)

    print(f"[probe] loading Show-o2 + LoRA from {CKPT}", flush=True)
    model = Showo2Qwen2_5.from_pretrained(
        cfg.model.showo.pretrained_model_path, use_safetensors=False,
    ).to(device).to(dtype)
    from peft import PeftModel
    model.showo = PeftModel.from_pretrained(model.showo, str(CKPT), is_trainable=False)
    model = model.to(device).to(dtype)
    model.eval()

    atok = ActionTokenizer(vocab_size=V, bins=256, hi_id=151642)
    lo, hi = atok.action_token_id_range

    ds = libero_spatial_dataset(dataset_dir=DATASET_DIR, num_chunks=1)
    print(f"[probe] dataset size = {len(ds)}", flush=True)

    rng = np.random.default_rng(0)
    probe_ids = rng.choice(len(ds), size=N, replace=False)

    # Collect first-token predictions and image embeds for each probe
    img_embeds_all = []
    first_token_logits_all = []
    first_token_argmax_all = []
    gt_first_token_all = []
    gt_actions = []
    print("\n=== Image-conditioning probe ===")
    print(f"action token range = [{lo}, {hi}]\n")

    for k, idx in enumerate(probe_ids):
        ex = ds[int(idx)]
        gt = ex["action"]  # shape [7]
        gt_actions.append(gt)
        gt_tok = atok.encode(gt)[0]  # first dim's GT token
        gt_first_token_all.append(int(gt_tok))

        with torch.no_grad():
            prefix, mp, img_embed = build_prefix(model, vae, tok, sids, cfg, device, dtype, ex["image"], ex["task"])
            img_embeds_all.append(img_embed.detach().float().cpu().numpy())

            # Single forward to predict first action token (no autoregressive)
            L0 = prefix.shape[1]
            attn = omni_attn_mask_naive(B=1, LEN=L0, modalities=mp, device=device, inverted=True).to(dtype)
            out = model.showo(inputs_embeds=prefix, attention_mask=attn)
            logits = out.logits[:, -1, :].float()  # [1, V]
            masked = torch.full_like(logits, float("-inf"))
            masked[:, lo:hi+1] = logits[:, lo:hi+1]
            argmax_tok = int(masked.argmax(dim=-1).item())
            first_token_argmax_all.append(argmax_tok)
            # Also keep top-5 probs over action vocab
            probs = F.softmax(masked, dim=-1)[0, lo:hi+1]
            top5 = torch.topk(probs, k=5)
            print(f"[probe {k}] idx={idx} task='{ex['task'][:40]}...'")
            print(f"  gt_action[0]={gt[0]:.4f}  gt_token={gt_tok}  decoded={atok.decode(np.asarray([gt_tok]))[0]:.4f}")
            print(f"  argmax_token={argmax_tok}  decoded={atok.decode(np.asarray([argmax_tok]))[0]:.4f}")
            print(f"  top-5 action token IDs: {[int(t+lo) for t in top5.indices.tolist()]}")
            print(f"  top-5 probs:            {[f'{p:.3f}' for p in top5.values.tolist()]}")
            print(f"  image_embed L2 norm: {np.linalg.norm(img_embed.float().cpu().numpy()):.2f}")

    print("\n=== Cross-image comparison ===")
    # Image-embed pairwise distances
    img_arr = np.stack([e.squeeze() for e in img_embeds_all])  # [N, T_img, H]
    img_flat = img_arr.reshape(N, -1)
    print(f"img_embed shape per probe: {img_embeds_all[0].shape}")
    print(f"img-embed mean L2 norm: {np.linalg.norm(img_flat, axis=1).mean():.2f}")
    print(f"img-embed pairwise L2 distance (off-diag mean): "
          f"{(np.linalg.norm(img_flat[:, None] - img_flat[None, :], axis=-1).sum() / (N*(N-1))):.2f}")

    # Token agreement
    unique_argmax = set(first_token_argmax_all)
    print(f"\nunique argmax first-tokens across {N} different images: {len(unique_argmax)}")
    print(f"  argmax_tokens: {first_token_argmax_all}")
    print(f"  gt_tokens:     {gt_first_token_all}")
    if len(unique_argmax) == 1:
        print("\n*** MODEL EMITS THE SAME FIRST-TOKEN REGARDLESS OF IMAGE. ***")
        print("    Image is not being used by the action head.")
    else:
        print(f"\nModel DOES vary first-token by image ({len(unique_argmax)}/{N} unique).")


if __name__ == "__main__":
    main()
