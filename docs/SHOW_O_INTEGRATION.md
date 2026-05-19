# Show-o2 Integration Design

**Date:** 2026-05-19
**Status:** decisions needed before implementation

This document maps the engineering surface for integrating Show-o2-1.5B as the MIRAGE backbone in RLinf-VLA. Three load-bearing design decisions are open; see §5.

---

## 1. What we have

- **Show-o2-1.5B weights**: `/nyx-storage1/hanliu/hf/models/show-o2-1.5B/` (8.4 GB)
  - `pytorch_model.bin`, `config.json`. No tokenizer (uses Qwen2.5-1.5B-Instruct's; needs `add_showo_tokens=True`).
- **Show-o2 source**: `/home/mzh1800/Show-o-repo/show-o2/` (cloned from `github.com/showlab/Show-o`)
- **Model class**: `Showo2Qwen2_5(ModelMixin, ConfigMixin)` at `show-o2/models/modeling_showo2_qwen2_5.py`
- **VAE**: Wan-VAE (downloaded separately at runtime via `WanVAE(vae_pth=...)`)
- **Vision encoder**: SigLIP (`google/siglip-so400m-patch14-384`)

## 2. Show-o2 architecture (relevant parts)

```
        ┌─────────────────────────────────────────┐
        │ obs image (RGB)                         │
        └────────────────────┬────────────────────┘
                             │ Wan-VAE.sample()
                             ▼
                  image_latents [B, 16, 27, 27]
                             │ image_embedder_und + image_embedder_gen
                             │ + position_embedding + und_trans + fusion_proj
                             ▼
                image_embeds [B, num_image_tokens, 1536]

        ┌─────────────────────────────────────────┐
        │ text tokens (Qwen tokenizer)            │
        └────────────────────┬────────────────────┘
                             │ embed_tokens
                             ▼
                  text_embeds [B, T, 1536]

         Concat: [bos, sys, role_user, boi, image_embeds, eoi,
                  user_text, role_assistant, ...]

                             │ Showo2Qwen2_5.forward / mmu_generate
                             ▼
              hidden states [B, L, 1536] → lm_head → logits over Qwen vocab
              (no native action output; for image gen, separate flow head)
```

**Heads available:**
- **Language head (lm_head)** — predicts next text token autoregressively
- **Flow matching head** — for image generation (3D Causal VAE space)
- **No action head** — would need to be added or actions encoded as text/tokens

## 3. Three open design decisions

### Decision A — Action representation

The blocking question: how does the unified MLLM "speak action"?

| Option | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **A1: Text-formatted actions** | Model emits `"[0.1, -0.2, 0.0, ...]"` style text; we parse it back | Zero architectural change; reuses existing LM head fully; works in any inference framework | Brittle (parse errors); tokens-per-action high; GRPO over messy token stream |
| **A2: Discrete action tokens** (OpenVLA pattern) | Allocate 256 unused token IDs as `<action_bin_N>` tokens; emit `action_dim × num_action_chunks` of them | Compact; matches RLinf/OpenVLA assumption; clean token-level GRPO; **enables visual-CoT before action tokens** | Need to extend tokenizer + retrain embeddings (small) |
| **A3: MLP action head** | Add `Linear(1536, action_dim × num_chunks)` after last hidden state | No tokenization; clean continuous action | Breaks GRPO token-level discipline; PPO instead; can't interleave imagined image as CoT before action |

**Recommendation: A2** — matches OpenVLA-OFT's pattern (so RLinf's existing workers, loss functions, and rollout logic apply with minimal changes), and is the only option compatible with the *visual chain-of-thought* core thesis (you can't interleave imagined-image tokens and action tokens if actions aren't tokens).

### Decision B — Env strategy (transformers version conflict)

Show-o2's `build_env.sh` pins `transformers==4.47.0`. Our `mirage_venv` is at `4.40.1` (forced by `moojink/transformers-openvla-oft` for OpenVLA-OFT). They are incompatible — OFT uses ancient model registration patterns the 4.47 codebase removed.

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| **B1: Separate `mirage_showo_venv`** | Fresh uv venv with transformers 4.47; install RLinf + Show-o2 deps; drop OFT support | Clean; no dep conflicts | Lose ability to run OpenVLA-OFT baselines from the same env. Need extra ~10 GB env on /nyx-storage1. |
| **B2: Patch Show-o2 to run on 4.40** | Backport / monkey-patch any 4.47-specific APIs | Single env, dual baseline | Fragile; Show-o2 may use new Qwen2.5-specific kernel paths not in 4.40 |
| **B3: Patch OFT to run on 4.47** | Update `moojink/transformers-openvla-oft` to current transformers | Cleanest long-term | Touching someone else's fork, upstream risk |

**Recommendation: B1** — accept the parallel-env cost. We can always activate `mirage_venv` for baselines and `mirage_showo_venv` for MIRAGE proper. Disk is the only "cost" and we have 1.3 TB free.

### Decision C — Integration shape with RLinf

| Option | Approach | Pros | Cons |
|--------|----------|------|------|
| **C1: Subclass `BasePolicy` directly, register as new model_type** | `class Showo2ForRLActionPrediction(Showo2Qwen2_5, BasePolicy)`; register `"showo2_action"` model_type; works inside RLinf's existing HF worker | Reuses RLinf rollout/training scaffolding wholesale; minimal new code | Multi-inheritance with diffusers' `ModelMixin` may clash; need to override checkpoint loading (config-driven instead of pretrained_dir) |
| **C2: Custom rollout worker (`mirage.workers.show_o_worker`)** | Bypass RLinf's HF worker; write our own | Full control; clean Show-o2 idioms | Reimplement half of `huggingface_worker.py`; lose RLinf's GRPO loss path; lots of code |
| **C3: Hybrid** | Subclass `BasePolicy` (C1) but wrap Show-o2 inside a thin proxy module that exposes `transformers`-shaped `forward(input_ids, ...)` | Lower coupling than C1; doesn't fight diffusers | One extra adapter layer |

**Recommendation: C1** — accept some friction; OpenVLA-OFT had similar multi-inheritance and made it work.

## 4. Phase 1 milestone breakdown (revised)

| Sub-phase | Goal | Gate | Estimated work |
|-----------|------|------|----------------|
| **1.0** | Smoke: load Show-o2-1.5B in a fresh venv on nyx and run `mmu_generate` on a single LIBERO obs image | Captioned output, no crash | 1-2 days (mostly env install) |
| **1.1** | Wrap as `Showo2ForRLActionPrediction(Showo2Qwen2_5, BasePolicy)` with discretized action tokens (A2); register in RLinf | Imports cleanly via RLinf `get_model` | 3-5 days |
| **1.2** | Implement `predict_action_batch` + `default_forward` (mirror OpenVLA-OFT pattern); add MLP **value head** for GRPO advantage | One LIBERO env step end-to-end, no crash | 3-5 days |
| **1.3** | Implement SFT warm-start on LIBERO demos so the model knows what action tokens mean | Non-random LIBERO success rate after 1 epoch SFT | 1-2 days |
| **1.4** | Integration smoke: one full GRPO training step in RLinf | Step completes, loss is finite | 1-2 days |

**Total Phase 1:** 9-15 working days from a settled design. The hardest sub-phase is 1.1 — getting Show-o2's Qwen2-MoT + flow-matching architecture to satisfy `BasePolicy` via `Showo2Qwen2_5` multi-inheritance.

## 5. What I need from you

1. **Decision A (action representation)**: A1 / A2 / A3? My vote: A2.
2. **Decision B (env strategy)**: B1 / B2 / B3? My vote: B1.
3. **Decision C (integration shape)**: C1 / C2 / C3? My vote: C1.
4. **Scope: should Phase 1 include SFT warm-start (sub-phase 1.3) or stop at 1.2 "loop closes"?**
5. **Show-o2-1.5B vs Show-o2-1.5B-HQ vs Show-o2-7B?** Currently downloaded 1.5B base. HQ has higher quality image gen but may be the same understanding capacity. 7B exceeds A40 single-GPU memory comfortably.

## 6. Recommended next concrete step (after decisions)

If A2 + B1 + C1 are confirmed:
1. Create `/nyx-storage1/hanliu/envs/mirage_showo_venv/` with Show-o2 deps (~1-2 hr)
2. Add `mirage/policy/show_o2_policy.py` with stub `Showo2ForRLActionPrediction` class
3. Write `scripts/phase1_0_smoke.py` — load Show-o2, run `mmu_generate` on a LIBERO-Spatial first-frame image, print the caption
4. Iterate from there

Decision required before I do any of this — these are 2+ week paths and going down the wrong one is expensive.
