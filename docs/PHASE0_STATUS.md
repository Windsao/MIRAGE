# Phase 0 Status — Cluster Setup Recon

**Date:** 2026-05-19
**Last updated:** 2026-05-19 (autonomous Phase 0 install in progress)

This document captures cluster setup state through Phase 0 execution.

---

## 1. Cluster access — ✅ working

- SSH alias `direct_slurm` reaches `slurm01.cs.northwestern.edu` via `shang` proxy.
- New aliases added to `~/.ssh/config`: `erebus`, `hemera`, `nyx` (all `ProxyJump direct_slurm`).
- Verified: `ssh erebus nvidia-smi` returns GPU state cleanly.

## 2. GPU state (snapshot at session start)

| Node | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|------|-------|-------|-------|-------|
| `erebus` | ~idle (813 MB used) | busy (42 GB used) | busy (42 GB used) | **idle (0 MB)** |
| `hemera` | listed `idle` by `sinfo` | | | |
| `nyx` | listed `idle` by `sinfo` | | | |

Effectively **9 of 12 A40s available** at session start.

## 3. Filesystem — ✅ cleanup applied

| FS | Free at start | Free now |
|----|---------------|----------|
| `/home/mzh1800` | 2.6 G (100% full) | **6.5 G (99%)** after deleting safe targets |
| `/nyx-storage1` | 1.4 T (68%) | 1.4 T (68%) |

**Cleanup performed:**
- Deleted `~/.cache`, `~/.npm`, `~/.triton`, `~/.cargo` (caches; 3.9 G freed)
- Deleted `~/awscliv2.zip`, `~/wan_5b.tar.gz` (install leftovers)
- Made `.cargo/env` source conditional in `.bashrc` (silenced warning)
- Preserved `InternVideo2-stage2_1b-224p-f4.pt` (2.7 G) — 4 active projects reference its path

## 4. Pre-existing relevant infrastructure on cluster

| Asset | Location | Notes |
|-------|----------|-------|
| `VLA-R1/` | `/home/mzh1800/VLA-R1/` (16 MB) | Actually EasyR1 (veRL fork with VLM support). |
| `verl/` | `/home/mzh1800/verl/` (799 MB) | veRL itself. |
| `lerobot/` | `/home/mzh1800/lerobot/` (317 MB) | HF LeRobot. |
| `verl` conda env | `/nyx-storage1/hanliu/miniconda3/envs/verl/` | torch 2.6 + cu124 — but **Python 3.10**, mismatched with RLinf's 3.11. Not reusable. |
| `RLinf/` | `/home/mzh1800/RLinf/` (18 MB) | **NEW** — cloned via `git clone --depth 1`. |

## 5. RLinf install — 🔄 in progress

### Approach
- uv venv at `/nyx-storage1/hanliu/envs/mirage_venv/`
- Python 3.11.14 (pre-fetched via `uv python install 3.11.14`)
- All caches redirected to `/nyx-storage1/hanliu/uv_cache/`
- Command: `bash requirements/install.sh embodied --model openvla --env maniskill_libero --venv /nyx-storage1/hanliu/envs/mirage_venv --no-root`

### Issues hit & resolved
- ❌ Initial attempt with uv 0.7.8 failed: "Python 3.11.14 no download available"
- ✅ Upgraded uv 0.7.8 → 0.11.15 via `uv self update`
- ✅ Pre-installed Python 3.11.14 via `uv python install`
- 🔄 Install relaunched, venv created successfully, packages installing

### Decision
- ❌ Abandoned plan to clone `verl` conda env (Python 3.10 ≠ RLinf's 3.11.14 requirement)

### Log
- `/home/mzh1800/MIRAGE/logs/rlinf_install.log` (tail with `ssh direct_slurm 'tail -f /home/mzh1800/MIRAGE/logs/rlinf_install.log'`)

## 6. Phase 0 target config — identified

`/home/mzh1800/RLinf/examples/embodiment/config/libero_spatial_grpo_openvlaoft.yaml`

Key specs:
- Single-node, FSDP backend
- OpenVLA-OFT (One-Step-Forwarder variant)
- LIBERO-Spatial env (`env/libero_spatial`)
- GRPO with group_size=8, rollout_epoch=16
- tensorboard logging to `../results`

Available LIBERO suites in configs: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`, `libero_130`. **Start with `libero_spatial`** (smallest, matches our Phase 0 plan).

## 7. Outstanding pre-Phase-0 tasks (after install completes)

1. Download LIBERO data — not yet on `/nyx-storage1/hanliu/`. RLinf install should fetch automatically, verify.
2. Download OpenVLA-OFT weights — not on `/nyx-storage1/hanliu/hf/`. Will be fetched at first model load.
3. Identify free GPUs at launch time (check `nvidia-smi` on erebus/hemera/nyx).
4. Launch `bash examples/embodiment/run_embodiment.sh <config>` with proper env vars.

## 8. What's on /nyx-storage1/hanliu/ for MIRAGE

After this session:

```
/nyx-storage1/hanliu/
├── envs/
│   └── mirage_venv/         # RLinf install target (NEW, in progress)
├── uv_cache/                # uv package cache (NEW)
├── hf/                      # HF cache (pre-existing, 574 GB)
└── miniconda3/envs/         # pre-existing conda envs (verl, lerobot, etc.)
```

Code stays at `/home/mzh1800/`:

```
/home/mzh1800/
├── MIRAGE/                  # our code (NEW)
├── RLinf/                   # RLinf-VLA (NEW)
├── VLA-R1/ (=EasyR1)        # pre-existing alt RL framework
├── verl/                    # pre-existing verl
├── lerobot/                 # pre-existing lerobot
└── (other project repos, untouched)
```
