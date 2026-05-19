"""Phase 1.0 smoke — minimal Show-o2 mmu_generate that prints the caption.

Reuses Show-o2's own pipeline (loading, embedding, generation) but cuts wandb
and outputs the caption to stdout so we can sanity-check it. The whole point
is to surface whether the vision encoder warnings during weight loading
actually broke captioning, or whether the model still produces sensible text.

This is also the de-facto inference scaffold we'll lift into a policy wrapper
in Phase 1.1 — same flow, different output format.

Run via: scripts/phase1_0_print_caption.sh (sources mirage_showo_venv, sets
PYTHONPATH to /home/mzh1800/Show-o-repo/show-o2, picks a free GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

# Show-o2 source isn't pip-installed; add it to PYTHONPATH at runtime.
SHOWO2_PATH = Path("/home/mzh1800/Show-o-repo/show-o2")
sys.path.insert(0, str(SHOWO2_PATH))

from models import Showo2Qwen2_5, WanVAE                                          # noqa: E402
from models.misc import load_state_dict                                          # noqa: E402
from utils import (                                                              # noqa: E402
    get_text_tokenizer,
    get_hyper_params,
    image_transform,
    path_to_llm_name,
    omni_attn_mask_naive,
)


CONFIG_PATH = Path("/home/mzh1800/MIRAGE/configs/showo2_smoke.yaml")
DEMO_IMAGE = SHOWO2_PATH / "docs/mmu/pexels-pixabay-207983.jpg"
QUESTION = "Describe this image in one short sentence."


def main() -> None:
    config = OmegaConf.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 if config.model.weight_type == "bfloat16" else torch.float32

    # VAE
    print(f"[smoke] loading Wan VAE from {config.model.vae_model.pretrained_model_path}", flush=True)
    vae = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_dtype,
        device=device,
    )

    # Tokenizer + Show-o token ids
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    # Show-o2 model
    print(f"[smoke] loading Show-o2 from {config.model.showo.pretrained_model_path}", flush=True)
    model = Showo2Qwen2_5.from_pretrained(
        config.model.showo.pretrained_model_path,
        use_safetensors=False,
    ).to(device).to(weight_dtype)
    model.eval()

    # Hyperparams (token offsets, max lens, etc.)
    (
        _, num_mmu_image_tokens, _, _, _, _, _,
        _, _, _, bos_id, eos_id, boi_id, eoi_id, _, _, _, _, _,
    ) = get_hyper_params(config, text_tokenizer, showo_token_ids)

    # Image -> latents -> multimodal image embeds
    img = Image.open(DEMO_IMAGE).convert("RGB")
    img = image_transform(img, resolution=config.dataset.preprocessing.resolution).to(device).unsqueeze(0)
    image_latents = vae.sample(img.unsqueeze(2)).squeeze(2).to(weight_dtype)

    image_embeds_und = model.image_embedder_und(image_latents)
    image_embeds_gen = model.image_embedder_gen(image_latents)
    image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
    image_embeds_und = model.und_trans(image_embeds_und)["last_hidden_state"]
    image_embeds = model.fusion_proj(
        torch.cat([image_embeds_und, image_embeds_gen], dim=-1)
    )

    # Build prompt: <bos><sys><role_user>\n<boi><eoi><question><role_assistant>
    sys_prompt_ids = text_tokenizer(
        "system\nYou are a helpful assistant.<|im_end|>",
        add_special_tokens=False,
    )["input_ids"]
    role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    role_b = text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    q_ids = text_tokenizer(QUESTION, add_special_tokens=False).input_ids

    text_tokens_a = torch.tensor(
        [showo_token_ids["bos_id"]] + sys_prompt_ids + role_a, device=device
    )[None, :]
    text_tokens_b = torch.tensor(
        [showo_token_ids["boi_id"], showo_token_ids["eoi_id"]] + q_ids + role_b,
        device=device,
    )[None, :]
    text_embeds_a = model.showo.model.embed_tokens(text_tokens_a)
    text_embeds_b = model.showo.model.embed_tokens(text_tokens_b)

    if config.model.showo.add_time_embeds:
        time_embeds = model.time_embed(torch.tensor([[1.0]], device=device), text_embeds_a.dtype)
        if hasattr(model, "time_embed_proj"):
            time_embeds = model.time_embed_proj(time_embeds)
        input_embeds = torch.cat(
            [text_embeds_a, text_embeds_b[:, :1], time_embeds, image_embeds, text_embeds_b[:, 1:]],
            dim=1,
        ).to(weight_dtype)
        modality_positions = torch.tensor(
            [text_tokens_a.shape[1] + 2, num_mmu_image_tokens], device=device
        )[None, None, :]
    else:
        input_embeds = torch.cat(
            [text_embeds_a, text_embeds_b[:, :1], image_embeds, text_embeds_b[:, 1:]],
            dim=1,
        ).to(weight_dtype)
        modality_positions = torch.tensor(
            [text_tokens_a.shape[1] + 1, num_mmu_image_tokens], device=device
        )[None, None, :]

    attention_mask = omni_attn_mask_naive(
        B=input_embeds.size(0),
        LEN=input_embeds.size(1),
        modalities=modality_positions,
        device=device,
        inverted=True,
    ).to(input_embeds.dtype)

    print(f"[smoke] generating (max_new_tokens=200)...", flush=True)
    output_tokens = model.mmu_generate(
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        top_k=1,
        max_new_tokens=200,
        eos_token=text_tokenizer.eos_token_id,
    )
    output_tokens = torch.stack(output_tokens).squeeze()[None]
    caption = text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    print("=" * 60, flush=True)
    print(f"IMAGE   : {DEMO_IMAGE.name}", flush=True)
    print(f"QUESTION: {QUESTION}", flush=True)
    print(f"ANSWER  : {caption}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
