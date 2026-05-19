# MIRAGE — Design Plan + INSIGHT Log

This document distills the planning conversation that produced MIRAGE. It is the canonical reference for *why* every decision was made. Update as decisions evolve.

---

## 1. Thesis

> *We introduce **MIRAGE**, a GRPO variant that exploits unified multimodal models' joint generation+understanding capability: rewards are augmented by (i) **imagination accuracy** — does the model's predicted next-frame match reality? — and (ii) **self-consistency** — does the model's own scoring agree with task outcome? Applied to a small (1.5B) unified backbone via RLinf-VLA, MIRAGE matches the 34B closed Emu3.5 result on LIBERO and beats 7B VLA baselines on long-horizon manipulation.*

## 2. Why this is publishable

Sellable axes:

1. **Method** — first RL objective that *trains the visual chain-of-thought capability* of unified MLLMs, rather than treating imagined frames as decorative. The two reward terms are *only available* in unified architectures (VLAs can't generate; text-CoT can't predict images).
2. **Efficiency** — 1.5B unified backbone vs 7B VLAs and 34B Emu3.5.
3. **Interpretability** — generated imagined frames are debuggable artifacts. Reviewers can inspect whether the model "thinks" sensibly.
4. **Reproducibility** — Emu3.5's RL pipeline is closed. RLinf-VLA's backbone roster has no unified models. MIRAGE fills both gaps with open code.

## 3. Chapter Plan

### Chapter 1 — Introduction (~1 pg)

- Embodied policies need *visual* reasoning, not just language reasoning.
- Three current paradigms each miss something: VLAs (no reasoning), text-CoT RL (lossy translation of visual state), closed unified-RL (Emu3.5: visual imagination but no objective forcing it to be useful, code closed).
- **MIRAGE** adds two unified-model-specific reward terms (IV + SC) that force visual CoT to be functional.
- Three contributions: (i) MIRAGE objective; (ii) first open unified-model + RLinf pipeline; (iii) 1.5B matches 7-34B on a fully open stack.

### Chapter 2 — Related Work (~0.75 pg)

Four buckets, each ending with a gap:

| Bucket | Examples | Gap |
|--------|----------|-----|
| VLA models | OpenVLA, π₀, GR00T, RT-2 | no reasoning step |
| Reasoning RL for robotics | LaST-R1, Embodied-R1 | text reasoning loses visual structure; pointing alone misses planning |
| Unified MLLM + embodied | Emu3.5 | closed training code; no explicit visual-CoT objective |
| Unified MLLM + GRPO (gen) | UniGen-1.5, UniRL, UAE, ULM-R1 | single-turn image-gen rewards; not for multi-turn embodied |

### Chapter 3 — Method (~2 pg, heaviest)

#### 3.1 Preliminaries
Show-o discrete-token backbone; interleaved token format; GRPO recap.

#### 3.2 Policy schema
At each control step the model autoregressively emits:
```
[obs_tokens | task_text | think_img_tokens | action_tokens]
```
`think_img_tokens` decode to a predicted next-frame.

#### 3.3 SFT warm-start
Robot demos formatted as the above schema. Imagined frame = actual next obs from demo. ≥5K trajectories per benchmark.

#### 3.4 MIRAGE objective

**Imagination-Verified reward (IV).**
For imagined frame `I_t` and observed next frame `O_{t+1}`:
```
r_IV = - LPIPS( decode(I_t), O_{t+1} )         # or DreamSim
```
Trains imagination toward *predictively useful*, not decorative.

**Self-Consistency reward (SC).**
Use the *same* unified model in scoring mode to predict task success from the imagined frame:
```
s_self    = model_score( task | I_t )           # model judges its own dream
s_actual  = realized episode success
r_SC      = - | s_self - s_actual |             # calibration penalty
```
Only possible in unified architectures: same parameters generate and score.

**Combined GRPO update.**
For each task prompt sample N rollouts; combined reward:
```
r_k = r_task,k  +  α · r_IV,k  +  β · r_SC,k
A_k = (r_k - mean(r)) / std(r)
```
Standard GRPO policy gradient on combined log-probs over `think_img` and `action` tokens.

#### 3.5 Why unified-model-specific
- IV requires the model to *generate* the predicted frame → VLAs can't.
- SC requires the model to *score* its own generation → text-CoT models can't.
- The same parameters doing both is what makes MIRAGE distinctly a unified-architecture method.

### Chapter 4 — Experiments (~1.5 pg)

- **Primary**: LIBERO (4 suites).
- **Secondary**: MetaWorld MT-50.
- **Long-horizon**: CALVIN (where visual-CoT should help most).

**Baselines** (same demo budget):
- OpenVLA-7B
- LaST-R1 (text CoT, RL)
- Embodied-R1 (pointing CoT, RL)
- Show-o-1.5B + vanilla GRPO (ablation of MIRAGE's two terms)
- Emu3.5-34B (numbers from paper)

Target: MIRAGE-1.5B ≥ OpenVLA-7B, ≥ LaST-R1, ≥ Embodied-R1; competitive with Emu3.5-34B on LIBERO; clear win on long-horizon (CALVIN).

Scaling: MIRAGE-1.5B (Show-o) vs MIRAGE-4B (InternVL-U).

### Chapter 5 — Analysis & Ablation (~1.5 pg)

1. Component ablation: vanilla GRPO / IV only / SC only / MIRAGE.
2. **Imagination accuracy vs success** scatter — confirms IV makes imagination predictive.
3. **Self-score calibration reliability diagram** — confirms SC calibrates.
4. Qualitative rollouts (money figure): side-by-side imagined vs actual frames.
5. Failure-mode analysis: when imagination diverges, does policy fail?
6. Compute comparison vs OpenVLA, Emu3.5.
7. GRPO vs PPO vs DAPO with MIRAGE rewards.

### Chapter 6 — Discussion, Limitations (~0.5 pg)

- Sim-only; image generation adds inference latency; SC bootstrapping risk.
- Future: real-robot eval; BEHAVIOR; cross-embodiment; Visual-MCTS (hand off to LatentMind project).
- Broader impact: interpretable embodied AI; imagined frame as debugging artifact.

---

## 4. Stack Decisions

| Layer | Choice | Why |
|-------|--------|-----|
| Backbone (primary) | Show-o-1.5B | Smallest; clean discrete-token design; LLaMA-3.2-1B base |
| Backbone (scaling) | InternVL-U-4B | Claims to beat BAGEL-14B; stronger understanding |
| RL framework | RLinf-VLA | Multi-turn embodied rollouts; GRPO/PPO/DAPO built-in; LIBERO/MetaWorld/CALVIN integrated; 99% on LIBERO with OpenVLA |
| Sim envs | LIBERO + MetaWorld + CALVIN | Standard + diversity + long-horizon |
| Compute | NU CS cluster A40s | 4× A40 (48 GB each) per node, 3 nodes available |
| Method reference | UniRL paper, Embodied-R1 RFT, UniGen-1.5 reward design | Recipes only — UniRL code is mostly TODO |

---

## 5. Position vs Prior Work

| Prior work | What they did | What MIRAGE adds |
|-----------|---------------|------------------|
| **Emu3.5** | Unified+GRPO+embodied at 34B, closed code | Open pipeline; explicit visual-CoT objective; small backbone |
| **LaST-R1** | RL with text-CoT for manipulation | Visual-CoT instead of text-CoT |
| **Embodied-R1** | RFT with "pointing" intermediate representation | Imagined-frame intermediate instead of pointing |
| **UniGen-1.5** | GRPO on unified models for image gen/edit | Same shape of method, but multi-turn embodied + new reward terms |
| **UniRL** | SFT+GRPO for unified models (paper only) | Working code + embodied scope |
| **RLinf-VLA** | RL infra for VLA backbones | First unified-MLLM backbone in this ecosystem |

Drop "first unified+RL for embodied" framing — Emu3.5 has priority. Frame as "open methodology + visual-CoT objective + efficiency" instead.

---

## 6. INSIGHT Log

Append-only running collection of decisions, risks, and discoveries from the planning dialogue. Numbers preserved from chat for traceability.

| # | Type | Insight |
|---|------|---------|
| I1 | Decision | Novelty axis is visual-imagination CoT (the model generates an intermediate image as its reasoning step). |
| I2 | Decision | Drop JEPA-alignment component from this paper's story. Pure unified-model + RL. |
| I3 | Decision | Two-stage training: SFT warm-start → GRPO with sparse episode-level reward. |
| I4 | Decision | Imagination-consistency loss is the **headline reward** (became IV reward) — not a peripheral ablation. |
| I5 | Risk | LaST-R1 hits 99.9% on LIBERO. Use LIBERO as calibration; differentiate on long-horizon (CALVIN). |
| I6 | Risk | "Visual CoT emerges from RL without supervision" is a strong claim — verify in pilot before committing. |
| I7 | Open | Compute budget for full GRPO rollouts is non-trivial. Show-o-1.5B keeps it tractable. |
| I8 | Decision | Real-robot evaluation deferred to follow-up. |
| I9 | Decision | Target venue ICLR 2027. |
| I10 | Reversed | Earlier "BAGEL-7B-MoT" backbone decision — replaced by Show-o-1.5B for compute reasons. |
| I11 | Reversed | Earlier "OpenRLHF" RL scaffold — replaced by RLinf-VLA which is purpose-built for embodied. |
| I12 | Decision | Recipe template from Embodied-R1's two-stage RFT curriculum. |
| I13 | Risk | Wiring a unified-model policy interface to RLinf is the first engineering risk. |
| I14 | Risk | WVA code not publicly available; head-to-head comparison may require contacting authors. |
| I15 | Open | LaST-R1 project page (siriyep.github.io/last-r1) — check for code. |
| I16 | Discovery | No public BAGEL+RL repo. Closest precedent: UniGen-1.5 (paper only). |
| I17 | Risk | Diagnostic paper 2603.17044: offline DPO fails on VQ-based unified models. GRPO may behave better; verify stability. |
| I18 | Decision | Two-phase implementation: Phase 0 = validate RLinf+unified-model numerics; Phase 1 = embodied. |
| I19 | Open | Contact UniGen-1.5 authors for code. |
| I20 | Discovery | Unified+RL ecosystem has 5+ public repos: Emu3.5, UniRL, UAE, ULM-R1, MaskGRPO, DeepGen, InternVL-U. |
| I21 | Decision | Backbone reconsidered: Show-o-1.5B over BAGEL-7B for compute; over InternVL-U-4B for iteration speed. InternVL-U is scaling ablation. |
| I22 | Reframe | Novelty must shift from "first unified+RL for embodied" (Emu3.5 has priority) to "first open unified+RL pipeline with explicit visual-CoT objective." |
| I23 | Risk | Reviewers may ask "isn't this just Emu3.5's pipeline with a robotics reward?" — sharper methodological contribution required beyond reward swap. MIRAGE's IV+SC objective answers this. |
| I24 | Discovery | Emu3.5 already did unified+GRPO+embodied at 34B. Closed training code. Ate the headline niche. |
| I25 | Decision | Counter-position vs Emu3.5: small + open + visual-CoT-explicit + long-horizon. |
| I26 | Critical-risk | Must verify whether Emu3.5's embodied training includes visual-CoT-as-RL-signal. If yes, novelty collapses. Resolved by adding *self-consistency* reward (SC) — even if Emu3.5 has IV-flavored signal, SC is novel and unified-architecture-specific. |
| I27 | Decision | Repositioned thesis: small + open + visual-CoT-explicit + long-horizon. |
| I28 | Decision | Drop "first unified+RL for embodied" framing — Emu3.5 has priority. |
| I29 | Discovery | UniRL's training code is mostly TODO and single-turn — unsuitable for embodied execution. |
| I30 | Discovery | RLinf-VLA solves the embodied multi-turn RL infra problem. v0.2 (March 2026), Apache 2.0, 99% on LIBERO with OpenVLA. |
| I31 | Decision | Stack v3 locked: RLinf-VLA + Show-o-1.5B + LIBERO/MetaWorld/CALVIN. Engineering shifts from "build infra" to "integrate one new backbone." |
| I32 | Decision | Show-o-1.5B primary; InternVL-U-4B scaling ablation. |
| I33 | Discovery | No unified MLLM in RLinf-VLA's backbone roster. Filling that slot is the engineering contribution. |
| I34 | Decision | Thesis v2: small unified + RLinf-VLA + visual-CoT. |
| I35 | Risk | InternVL-U is fresh (Mar 2026). May have undocumented quirks. |
| I36 | Clarification | UniRL = method reference, not execution codebase. |
| I37 | Decision | Final stack: RLinf-VLA (execution) + small unified backbone (wrapped policy) + UniRL/Embodied-R1 (recipes). |
| I38 | Action | Two-week de-risking milestone before locking chapter plan. |
| I39 | Decision | Strategic pivot: design own RL method, not just integrate. Methodological contribution. |
| I40 | Decision | Backbone: Show-o-1.5B (primary) + InternVL-U-4B (scaling ablation). |
| I41 | Decision | RL method: IV + SC combined (MIRAGE). |
| I42 | Risk | Each method has a "does this actually train?" risk. Phase 0 pilot on a single LIBERO task mandatory. |
| I43 | Differentiation | Visual MCTS-RL deferred to LatentMind project to maintain project boundaries. |
| I44 | Decision | Method name: **MIRAGE**. Strong semantic fit; counters "won't it hallucinate?" reviewer instinct. |
| I45 | Decision | Backbone: Show-o-1.5B primary, InternVL-U-4B scaling ablation. |
| I46 | Decision | Method components: IV (LPIPS imagined↔actual) + SC (model self-score↔outcome calibration). Both unified-model-specific. |
| I47 | Decision | Execution stack: RLinf-VLA + Show-o policy wrapper + LIBERO/MetaWorld/CALVIN. |
| I48 | Decision | Position vs Emu3.5: smaller + open-pipeline + explicit visual-CoT objective. |
| I49 | Decision | Visual-MCTS deferred to LatentMind project. |
| I50 | Risk | SC reward has bootstrapping risk — initial untrained model's self-scores are unreliable. Mitigation: warm-start SC for a few epochs on labeled demo outcomes before letting it influence GRPO. |
| I51 | Risk | LPIPS between low-res imagined frame and high-res actual obs may be noisy. Mitigation: use DreamSim, or train at fixed shared resolution. |
| I52 | Risk | Show-o-1.5B was trained for image gen, not embodied. SFT warm-start step is load-bearing. Budget ≥5K demos per task. |
| I53 | Next | Phase 0 pilot: reproduce RLinf-VLA OpenVLA baseline on LIBERO-Spatial before locking MIRAGE training. |

---

## 7. Open Questions / Verifications Needed

- [ ] Exact license of Show-o-1.5B and InternVL-U-4B weights (probably Apache-2.0, verify).
- [ ] Whether RLinf-VLA's policy interface assumes continuous action heads or supports discrete-token action policies natively.
- [ ] LIBERO demo download path on the cluster — likely under `/nyx-storage1/hanliu/`, check or fetch.
- [ ] InternVL-U fine-tuning hooks: does the repo expose a clean training entrypoint, or do we need to build it?
- [ ] Whether LPIPS or DreamSim is more sensible for IV reward at the resolution Show-o operates on (256 or 512 px).
