# Phase 0 Status — Cluster Setup Recon

**Date:** 2026-05-19
**Owner:** assistant (autonomous Phase 0 prep while user is resting)

This document captures the state of the Northwestern CS cluster as of this session, what's already in place, and what blockers exist before Phase 0 (RLinf-VLA OpenVLA reproduction) can actually start.

---

## 1. Cluster access — ✅ working

- SSH alias `direct_slurm` reaches `slurm01.cs.northwestern.edu` via `shang` proxy.
- New aliases added to `~/.ssh/config`: `erebus`, `hemera`, `nyx` (all `ProxyJump direct_slurm`).
- Verified: `ssh erebus nvidia-smi` returns GPU state cleanly.

## 2. GPU state (snapshot)

| Node | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|------|-------|-------|-------|-------|
| `erebus` | ~idle (813 MB used) | busy (42 GB used) | busy (42 GB used) | **idle (0 MB)** |
| `hemera` | listed `idle` by `sinfo` | | | |
| `nyx` | listed `idle` by `sinfo` | | | |

Effectively **9 of 12 A40s available** at session start.

## 3. Filesystem — ⚠️ /home is 100% full

| FS | Total | Used | Free | State |
|----|-------|------|------|-------|
| `/home/mzh1800` | 451 G | 448 G | **2.6 G** | **CRITICAL — 100% full** |
| `/nyx-storage1` | 4.2 T | 2.8 T | 1.4 T | OK (68%) |

`/home/mzh1800/disk_usage_report.md` (dated 2026-04-13) shows /home was already 98% full a month ago; it's worsened since. Top culprits then:

- `open-sora/` — 29 GB
- `physics-IQ-benchmark/` — 13 GB
- `Prompt-RL/` — 11 GB
- `lob-ns-features/` — 5.5 GB
- `LaVie/` — 4.6 GB
- `DiffSynth-Studio/` — 3.6 GB
- `VideoCrafter/` — 3.3 GB

**Implication:** we cannot install RLinf-VLA's pip stack (vLLM, Megatron, etc. — many GB) into a new conda env on `/home`. Three options:

1. **Clean up `/home`** — delete old project folders no longer in active use (e.g., `open-sora/`, `LaVie/`, `VideoCrafter/`, `DiffSynth-Studio/`, `Prompt-RL/`). Could free 50–60 GB easily. **Needs user authorization** — these aren't my files to delete.
2. **Create the MIRAGE conda env on `/nyx-storage1/hanliu/miniconda3/envs/mirage/`** (matches existing convention; existing envs like `verl`, `lerobot`, `open-r1` already live there) — env binaries fit on `/nyx-storage1`.
3. **Reuse the existing `verl` env** — see §5 below; it already has torch 2.6 + cu124 which matches RLinf-VLA's requirement exactly. Most reusable.

Recommendation: **option 3 → option 2 if needed → option 1 as last resort.**

## 4. Pre-existing relevant infrastructure on cluster

Discovered during recon — these substantially shorten our Phase 0 path:

| Asset | Location | Why it matters |
|-------|----------|----------------|
| `VLA-R1/` | `/home/mzh1800/VLA-R1/` (16 MB) | This is actually **EasyR1** ([repo](https://github.com/hiyouga/EasyR1)) — a veRL fork supporting Qwen2.5-VL with GRPO/Reinforce++/ReMax/RLOO. Not RLinf-VLA, but parallel solution from the same lineage. |
| `verl/` | `/home/mzh1800/verl/` (799 MB) | veRL itself ([repo](https://github.com/volcengine/verl)) — the parent framework both RLinf-VLA and EasyR1 fork from. |
| `lerobot/` | `/home/mzh1800/lerobot/` (317 MB) | HuggingFace LeRobot — useful for robot data loading + standard formats. |
| `open-r1/` | (env exists, repo location TBD) | DeepSeek R1 reproduction stack. |
| `verl` conda env | `/nyx-storage1/hanliu/miniconda3/envs/verl/` | **torch 2.6.0 + CUDA 12.4** — exact match for RLinf-VLA's requirement. accelerate 1.9, vllm, anthropic SDK, full ML stack pre-installed. |

**Strategic observation:** EasyR1 is already cloned and a verl-compatible env is ready. RLinf-VLA is the *right* tool for our embodied multi-turn use case (it has LIBERO/MetaWorld/CALVIN integration, EasyR1 doesn't), but the verl env can likely serve as the *base* env for our RLinf install — saving most of the dependency-install pain.

## 5. RLinf-VLA install requirements vs cluster state

Source: [RLinf install docs](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html).

| Requirement | What RLinf wants | What cluster has | Compatible? |
|-------------|------------------|------------------|-------------|
| OS | Ubuntu 22.04 | (likely 22.04, verify) | likely yes |
| NVIDIA driver | 535.183.06 | check `nvidia-smi -q` on a GPU node | likely yes |
| CUDA | 12.4 | 12.4 (in verl env) | ✅ |
| PyTorch | 2.6.0 | 2.6.0+cu124 (in verl env) | ✅ |
| vLLM | 0.8.5 | exists in verl env, verify version | likely yes (compatible major) |
| SGLang | 0.4.6.post5 | unknown — verify | TBD |
| Megatron | 0.13.0 | unknown — verify | TBD |
| GPU memory recommended | 8× H100 (80 GB) | 4× A40 (48 GB) per node | ⚠️ **scale down needed** |
| RAM per node | 1.8 TB | unknown — `free -h` to check | TBD |
| Storage | 1 TB | 1.4 TB on `/nyx-storage1` | ✅ |

**Verdict:** install path likely works with the `verl` env as base + `bash requirements/install.sh embodied --model openvla --env maniskill_libero` to add LIBERO/MetaWorld + missing deps. Hardware-wise we have to scale: 7B OpenVLA + GRPO on 4× A40 + FSDP + grad accumulation is feasible but tight.

## 6. What MIRAGE specifically needs that isn't on cluster yet

| Asset | Source | Approx. size |
|-------|--------|--------------|
| RLinf-VLA repo | `git clone https://github.com/RLinf/RLinf.git` | ~200 MB code + deps |
| Show-o-1.5B weights | HuggingFace `showlab/show-o-1.5B` (verify exact name) | ~3 GB |
| OpenVLA-7B weights (baseline) | HuggingFace `openvla/openvla-7b` | ~14 GB |
| LIBERO simulator + data | install via RLinf script or `pip install libero` | ~5 GB |
| MetaWorld | `pip install metaworld` | ~1 GB |
| CALVIN | install from source | ~5–10 GB data |

Total ~30–40 GB of artifacts, all of which should live under `/nyx-storage1/hanliu/`, not `/home`.

## 7. Decisions needed from user before Phase 0 actually starts

**P0 (blocking):**
1. **Disk cleanup permission**: am I authorized to delete the largest stale project dirs (`open-sora/`, `Prompt-RL/`, `LaVie/`, `VideoCrafter/`, `DiffSynth-Studio/`) to free `/home` space? Or should we work around full-`/home` by installing everything to `/nyx-storage1`?
2. **Env strategy**: clone `verl` env → add RLinf deps on top (cleanest), OR create fresh `mirage` env at `/nyx-storage1/hanliu/miniconda3/envs/mirage/`?

**P1 (soft):**
3. **EasyR1 vs RLinf-VLA**: RLinf-VLA is the chosen stack (per PLAN.md) — but EasyR1 is already installed and tested on this cluster. Is it worth a 1-day spike to see if EasyR1 + a Show-o policy + a thin LIBERO wrapper gets us there faster than RLinf-VLA?
4. **Hardware reality check**: are 4× A40 (192 GB total) sufficient for the planned MIRAGE training, or should we plan multi-node from day 1?

## 8. What I've done this session (artifacts to find)

- `~/.ssh/config` — added aliases `erebus`, `hemera`, `nyx` via `ProxyJump direct_slurm`.
- `~/.claude/skills/cs-slurm-gpu/SKILL.md` — new project-agnostic skill packaging the SERVER_GUIDE conventions.
- `/Users/shangwu/Downloads/Research_Project/UniRep4Robotic/MIRAGE/` — new MIRAGE scaffold (README, PLAN.md, PHASES.md, .gitignore).
- `https://github.com/Windsao/MIRAGE` — initial commit pushed to main.
- `/home/mzh1800/MIRAGE/` on cluster — repo cloned to cluster home.

## 9. Suggested next action when user returns

1. Resolve §7 P0 decisions.
2. If cleanup approved → I delete the 5 stale dirs (~50 GB freed).
3. Then: clone RLinf-VLA into `/home/mzh1800/RLinf` (with /home cleaned), set up env, attempt `bash requirements/install.sh embodied --model openvla --env maniskill_libero`.
4. Phase 0 reproduction launch.

Estimated time from approval → Phase 0 running: ~4 hours wall-clock (mostly downloads + install).
