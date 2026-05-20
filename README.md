# MIRAGE

**M**ultimodal **I**magination **R**eward **A**nd **G**enerative **E**mbodiment

> Training small unified multimodal models for embodied manipulation via imagination-verified, self-consistent reinforcement learning.

---

## One-paragraph pitch

Vision-Language-Action (VLA) models map perception to action directly with no reasoning step; text-CoT RL adds linguistic reasoning that loses visual structure; closed unified-model RL (Emu3.5) demonstrates visual imagination but provides no open training code and no objective that *makes the imagination useful*. MIRAGE introduces a GRPO variant with two unified-model-specific reward terms — **imagination-verified** (does the predicted next-frame match reality?) and **self-consistency** (does the model's own success-scoring agree with the realized outcome?) — that explicitly train the visual chain-of-thought capability of unified architectures. Applied to a small (1.5B) unified backbone via the open RLinf-VLA framework, MIRAGE targets parity with 7B VLAs and the 34B closed Emu3.5 on standard manipulation benchmarks, on a fully open training pipeline.

---

## Stack

| Layer | Choice |
|-------|--------|
| RL framework | [RLinf-VLA](https://github.com/RLinf/RLinf) — multi-turn embodied rollouts, GRPO/PPO/DAPO |
| Backbone (primary) | [Show-o-1.5B](https://github.com/showlab/Show-o) — discrete-token unified MLLM, LLaMA-3.2-1B base |
| Backbone (scaling ablation) | [InternVL-U-4B](https://github.com/OpenGVLab/InternVL-U) |
| Sim environments | LIBERO (primary), MetaWorld (diversity), CALVIN (long-horizon) |

---

## Repo layout

```
MIRAGE/
├── README.md
├── configs/               # YAML configs (Show-o2 smoke)
├── mirage/                # main package
│   └── policy/            # action tokenizer; unified-model policy wrappers
└── scripts/               # phase-by-phase launchers
```

---

## License

To be decided before release. Likely Apache-2.0 (matches Show-o / RLinf-VLA).
