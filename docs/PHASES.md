# MIRAGE — Phase Roadmap

Week-by-week execution plan. Each phase has a clear gate: a concrete artifact and a binary pass/fail. Phases are sequential; later phases assume earlier ones passed.

Hardware budget: 2 idle NU CS cluster nodes (`hemera`, `nyx`) — 4× A40 (48 GB) each = 8 A40s available; `erebus` partial. Reserve 1 node for evaluation, train on remaining.

---

## Phase 0 — Infrastructure smoke test (Week 1)

**Goal:** Confirm RLinf-VLA works on our cluster before touching unified models.

**Steps:**
1. Clone RLinf-VLA on the cluster.
2. Install into a fresh conda env at `/nyx-storage1/hanliu/miniconda3/envs/mirage`.
3. Reproduce the RLinf-VLA OpenVLA + GRPO baseline on **LIBERO-Spatial** (one suite, smallest).
4. Confirm: training launches, GRPO loss decreases, eval rollouts run.

**Gate:** OpenVLA + GRPO on LIBERO-Spatial reaches ≥ 80% success rate within RLinf's documented training time. If not, debug infra before doing anything else.

**Risk:** RLinf's installation may have dependency conflicts with our cluster's torch / CUDA versions.

**Deliverable:** `scripts/phase0_repro_openvla.sh` + a results entry in `docs/RESULTS.md`.

---

## Phase 1 — Wrap Show-o as RLinf policy (Weeks 2–3)

**Goal:** Make Show-o-1.5B drive an RLinf rollout, even if untrained for robotics.

**Steps:**
1. Read RLinf-VLA's policy interface (likely a `Policy` base class with `act(obs) → action` and a log-prob API).
2. Implement `mirage/policy/show_o_policy.py`:
   - Load Show-o-1.5B weights from HF.
   - Tokenize obs (image) + instruction (text).
   - Generate `think_img_tokens` followed by `action_tokens`.
   - Map action tokens → action vector (Show-o's vocabulary needs an action-token allocation — see §1.1).
   - Expose log-prob over generated tokens for GRPO gradient.
3. Smoke test: a single LIBERO env step. The action will be garbage; the goal is "doesn't crash."

**1.1 Action-token allocation.** Three options to evaluate:
- (a) Quantize continuous actions to discrete bins, allocate dedicated tokens in Show-o's vocabulary.
- (b) Map action embeddings to image-token embeddings space (action-as-image).
- (c) Add a small action head on top of Show-o's last hidden state.

Default plan: (a) — cleanest match to discrete-token GRPO.

**Gate:** RLinf rollout with Show-o policy runs end-to-end without crashing, on at least one LIBERO task.

**Risk:** Show-o's tokenizer may not have room for action tokens — may need vocab extension.

**Deliverables:** `mirage/policy/show_o_policy.py`, `tests/test_show_o_rollout.py`.

---

## Phase 2 — SFT warm-start (Week 4)

**Goal:** Take Show-o from random embodied performance to non-zero success after SFT on demos.

**Steps:**
1. Download / locate LIBERO demos under `/nyx-storage1/`.
2. Format trajectories into the MIRAGE token schema:
   `[obs_t | task | think_img_{t+1} | action_t]` where `think_img_{t+1}` is the next observation from the demo.
3. SFT for ~1 epoch on each LIBERO suite. Monitor: train loss, held-out token accuracy, qualitative imagined-frame plausibility.
4. Eval: zero-shot rollouts on held-out tasks.

**Gate:** SFT warm-started Show-o reaches ≥ 20% success rate on LIBERO-Spatial without any RL. Confirms (a) the schema works, (b) Show-o can learn to act.

**Risk:** Show-o's pretraining may interfere — image-gen prior may produce decorative imagined frames that ignore robot scene structure.

**Deliverables:** `mirage/train/sft.py`, SFT'd checkpoint on `/nyx-storage1/`.

---

## Phase 3 — Vanilla GRPO baseline (Weeks 5–6)

**Goal:** Show-o-1.5B + vanilla GRPO (task reward only, no IV no SC) on LIBERO. This becomes the *MIRAGE ablation baseline*.

**Steps:**
1. Configure RLinf-VLA to use Show-o policy with vanilla GRPO (task success reward).
2. Train on LIBERO-Spatial, LIBERO-Object.
3. Eval success rate; compare against OpenVLA-7B baseline numbers.

**Gate:** Vanilla GRPO produces a measurable improvement over SFT-only checkpoint. If not, infra or schema bug — debug.

**Deliverables:** vanilla-GRPO checkpoint; Table 1 row.

---

## Phase 4 — MIRAGE: IV reward (Week 7)

**Goal:** Add imagination-verified reward; demonstrate imagination accuracy improves.

**Steps:**
1. Implement `mirage/rewards/imagination_verified.py`:
   - Decode `think_img_tokens` to RGB.
   - Compute LPIPS (or DreamSim) against the next observed obs.
   - Negate → reward term.
2. Plug into RLinf reward function with weight `α`.
3. Train on LIBERO-Spatial.
4. Metric: scatter plot LPIPS(imagined, actual) vs task success across training steps.

**Gate:** LPIPS improves *and* task success ≥ vanilla GRPO baseline. If task success drops, α is too high.

**Deliverables:** IV reward module; scatter plot for Chapter 5.

---

## Phase 5 — MIRAGE: SC reward (Week 8)

**Goal:** Add self-consistency reward; demonstrate self-score calibration.

**Steps:**
1. Implement `mirage/rewards/self_consistency.py`:
   - Re-run Show-o on `(task, imagined_frame)` in *scoring mode* (likelihood over success token, or learned head).
   - After episode, compute |s_self − s_actual|.
   - Negate → reward term.
2. Plug into RLinf reward function with weight `β`.
3. Warm-start SC for a few epochs on labeled demo outcomes (mitigation for I50 bootstrapping risk).
4. Train on LIBERO-Spatial.
5. Metric: reliability diagram for self-score vs actual outcome.

**Gate:** Reliability diagram improves *and* task success ≥ IV-only baseline.

**Deliverables:** SC reward module; reliability diagram for Chapter 5.

---

## Phase 6 — Scale to MetaWorld + CALVIN (Weeks 9–10)

**Goal:** Demonstrate MIRAGE generalizes beyond LIBERO.

**Steps:**
1. Configure RLinf-VLA for MetaWorld MT-50 and CALVIN.
2. Repeat SFT → vanilla GRPO → MIRAGE pipeline.
3. Eval against baselines (OpenVLA, LaST-R1, Embodied-R1 if their checkpoints are public).

**Gate:** MIRAGE beats vanilla GRPO on CALVIN long-horizon by a clear margin (≥ 5 pp). This is the *long-horizon advantage* claim — the differentiator from LaST-R1.

**Deliverables:** Tables 1, 2, 3 in the paper.

---

## Phase 7 — Scaling ablation & analyses (Week 11)

**Goal:** Strengthen paper with secondary results.

**Steps:**
1. MIRAGE-1.5B (Show-o) vs MIRAGE-4B (InternVL-U) on LIBERO.
2. Component ablation: vanilla / IV-only / SC-only / IV+SC.
3. Algorithm ablation: GRPO / PPO / DAPO with MIRAGE rewards.
4. Compute comparison vs OpenVLA, Emu3.5.
5. OOD generalization: novel objects, layouts.

**Deliverables:** All ablation tables for Chapter 5.

---

## Phase 8 — Writing (Weeks 12–14)

Standard ML paper writing cadence. ARS pipeline (`/ars-outline` → `/ars-full`) can scaffold a first draft once results stabilize.

**Deliverables:** ICLR 2027 submission.

---

## Cumulative timeline

| Phase | Weeks | Cumulative |
|-------|-------|-----------|
| 0 | 1 | 1 |
| 1 | 2 | 3 |
| 2 | 1 | 4 |
| 3 | 2 | 6 |
| 4 | 1 | 7 |
| 5 | 1 | 8 |
| 6 | 2 | 10 |
| 7 | 1 | 11 |
| 8 | 3 | 14 |

14 weeks (~3.5 months) from Phase 0 start to submission. Assuming start mid-May 2026 → late August 2026 — *before* the ICLR Sep–Oct submission window.

---

## Decision points (gates that may force re-planning)

- **End of Phase 0:** If RLinf doesn't reproduce OpenVLA result, our infra assumption is wrong. Fall back: minimal RL loop on top of RLinf primitives.
- **End of Phase 2:** If SFT warm-started Show-o can't reach 20%, the schema or backbone is wrong. Switch to InternVL-U or restructure the action token format.
- **End of Phase 5:** If MIRAGE doesn't beat vanilla GRPO, the central thesis is wrong. Pivot to a different framing of the contribution (e.g., open pipeline only).
- **End of Phase 6:** If MIRAGE doesn't show long-horizon advantage on CALVIN, drop CALVIN claim and emphasize methodological novelty + open-pipeline + interpretability instead.
