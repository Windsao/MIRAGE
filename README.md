# MIRAGE

**M**ultimodal **I**magination **R**eward **A**nd **G**enerative **E**mbodiment

> Training small unified multimodal models for embodied manipulation via imagination-verified, self-consistent reinforcement learning.

**Status:** scaffolding (Week 0). Code is empty; see `docs/PLAN.md` for the full design and execution roadmap.

**Target venue:** ICLR 2027 (submission Sep–Oct 2026).

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
| Compute | Northwestern CS cluster, A40 nodes (`erebus`/`hemera`/`nyx`) |

---

## Repo layout (planned)

```
MIRAGE/
├── README.md              # this file
├── docs/
│   ├── PLAN.md            # design doc + chapter plan + INSIGHT log
│   └── PHASES.md          # week-by-week execution milestones
├── configs/               # YAML configs for SFT + GRPO runs
├── mirage/                # main package
│   ├── policy/            # unified-model → RLinf policy wrapper
│   ├── rewards/           # IV (imagination-verified) + SC (self-consistency) terms
│   ├── rollout/           # multi-turn rollout loop, env wrappers
│   ├── train/             # SFT warm-start, GRPO training
│   └── eval/              # success-rate + interpretability metrics
├── scripts/               # slurm/SSH launchers, one per phase
├── tests/                 # unit tests
└── pyproject.toml         # package metadata
```

Modules will be filled in as each phase completes — see `docs/PHASES.md`.

---

## Quick links

- **Plan + INSIGHT log:** [`docs/PLAN.md`](docs/PLAN.md)
- **Phase roadmap:** [`docs/PHASES.md`](docs/PHASES.md)
- **Closest prior work:**
  - [Emu3.5: Native Multimodal Models are World Learners](https://arxiv.org/abs/2510.26583) — closed 34B unified+RL, ate the headline niche
  - [LaST-R1: Reinforcing Robotic Manipulation via Adaptive Physical Latent Reasoning](https://arxiv.org/abs/2604.28192) — text-CoT RL baseline
  - [Embodied-R1: Reinforced Embodied Reasoning](https://arxiv.org/abs/2508.13998) — RFT curriculum template
  - [RLinf-VLA: Unified Framework for VLA+RL](https://arxiv.org/abs/2510.06710) — execution platform

---

## License

To be decided before first non-scaffold release. Likely Apache-2.0 (matches BAGEL / Show-o / RLinf-VLA).
