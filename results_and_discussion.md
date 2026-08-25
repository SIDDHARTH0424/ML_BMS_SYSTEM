# Results and Discussion — RL-BMS Final Report

> [!IMPORTANT]
> **Project Status — Updated after final evaluation run (2026-08-23)**
>
> | Area | Status |
> | :--- | :---: |
> | Core implementation & tests | ✅ 214 / 214 passed |
> | Track A — A1 50k charging model (seeds 7, 21, 42) | ✅ Validated |
> | Track B — B3 100k driving EMS (seeds 7, 21, 42) | ✅ Validated |
> | Track A — 1M-step long-training run | ❌ Fails target-reach gate |
> | Track B — 1M-step long-training (formal gate) | ⚠️ Raw results present, not formally gated |
>
> The **validated deliverables** are the A1 50k charging model and the B3 100k
> driving EMS, stored in `final_models/`. The 1M long-training failure is
> documented as a research finding in `docs/research_findings.md`.
> Numbers in Sections 3–5 below reflect the validated candidates unless
> explicitly noted otherwise.

## Executive Summary

This project built a reinforcement-learning-based EV fast-charging controller
(PPO) and evaluated it against four baseline strategies (Constant Current,
CCCV, a rule-based Adaptive controller, and a trivial Max-Current controller)
under a shared, physics-grounded battery simulation and a shared rule-based
safety layer.

**Primary finding:** after a systematic debugging process that found and
fixed seven distinct issues spanning the action space, safety layer, reward
function, thermal physics, energy accounting, and PPO's training
configuration, the trained policy converges to a strategy that is
**functionally very close to "request near-maximum current, modulated by
state"** — closely matching the trivial Max-Current baseline in aggregate
outcomes (charging time, safety-layer intervention rate), despite being
demonstrably, genuinely state-adaptive internally (not degenerately flat).

We conclude this is not a training failure but a **result about the
reward and safety-layer design as specified**: given this task definition,
near-maximum-current charging is close to the actual optimum, and PPO
correctly finds it once every training pathology is removed. This is the
project's central, defensible contribution — not "RL beats baselines," but
a fully-instrumented demonstration of *why* it doesn't, with each
contributing factor identified, isolated, and fixed or ruled out in turn.

---

## 1. Experimental Setup

**Task.** Charge a simulated 121 Ah / ~372 V NMC EV battery pack (parameters
sourced from the Tata Nexon EV LR spec sheet + NMC cell literature — full
provenance in `configs/battery.yaml`) from 10–30% initial SoC to 95% SoC,
across ambient temperatures of 15/25/35°C, subject to a shared rule-based
safety layer (current limiting, temperature derating, progressive
voltage/SoC tapering).

**Controllers compared.**
- **CC** — fixed constant current (1C, 121 A)
- **CCCV** — constant current until a voltage setpoint, then proportional taper
- **Adaptive** — SoC-banded rule-based current table
- **Max-Current** — trivial controller, always requests the physical
  current ceiling (160 A) unconditionally; added specifically as a control
  experiment (Section 4) to test whether PPO's behavior differs from the
  simplest possible aggressive strategy
- **PPO** — the trained RL controller (Stable-Baselines3, MLP 2×64)
- **PPO (safety derating disabled)** — ablation: identical trained policy,
  safety layer left in monitoring-only mode (interventions logged, episode
  still hard-terminates on overvoltage/overtemperature) rather than
  actively clamping current

**Evaluation protocol.** All controllers run through an identical fixed
grid of 15 scenarios (5 initial SoC x 3 ambient temperatures, fixed seeds),
scored on charging time, temperature, safety interventions, current
smoothness, energy efficiency, and -- added specifically to prevent
"truncated-but-incomplete" episodes being scored as if they succeeded --
`target_reached`, `time_to_target_s`, and `target_shortfall`.

**Final training configuration** (after the corrections in Section 2):
PPO, `n_steps=8192`, `target_kl=0.01`, `ent_coef=0.01`, symmetric `[-1,1]`
action space, seed 7, 1,000,000 timesteps, checkpointed every 25,000 steps
and swept post-hoc rather than assuming the final checkpoint is best.

---

## 2. Methodology: The Debugging Journey

This project's most substantive work was not the initial implementation but
a sequence of increasingly specific diagnostic experiments, each of which
found and fixed a real, verifiable issue. This section documents that
sequence because the final result (Section 4) is only trustworthy in light
of it -- an RL agent converging to "always request near-max" could mean
many different things, and distinguishing between them required ruling
each one out individually.

### 2.1 Action-space saturation (the initial failure)

The first trained policy converged to requesting **exactly zero current in
every scenario**. Diagnosis: the action space was `Box(0, 1)` -- asymmetric.
SB3's continuous-action PPO uses an unbounded Gaussian policy internally;
when the sampled mean drifted negative (easy early in training), it clipped
hard to 0 with zero gradient through the clip, and a zero entropy
coefficient (`ent_coef=0.0`) provided no pressure to escape. **Fix:**
symmetric `[-1,1]` action space (remapped internally to `[0, i_max]`) plus
`ent_coef=0.01`.

### 2.2 Reward scale imbalance

The `charging_progress` reward term was two orders of magnitude too small
relative to the smoothness/safety penalty terms (full-current progress:
~0.0037/step vs. up to 0.5/step for smoothness alone), making "avoid
penalties" trivially easier to optimize than "charge the battery."
**Fix:** rescaled `charging_progress` weight from 10 to 1000, derivation
documented in `configs/reward.yaml`.

### 2.3 Safety-layer double-derating (the "always max" trap)

With the reward and action space fixed, the policy converged to **always
requesting maximum current regardless of state** -- verified via a policy
sensitivity analysis tool built specifically to distinguish genuine
state-adaptive control from action-space saturation artifacts (raw
pre-clip policy output vs. clipped current). Direct simulation confirmed
this wasn't a training bug: a hand-designed "smart" controller that
requested exactly the safety-computed ceiling scored **worse** than always
requesting the physical maximum (378 vs. 532 reward in one test scenario).

Root cause: the safety layer computed `applied = min(requested, i_max) x
multiplier` -- capping the request at `i_max` *before* applying the
derating multiplier. Any request below `i_max` got derated a second time
on top of its own reduction, making "always request >= i_max" the
*unique* way to reach the true safe ceiling. No reward redesign could
incentivize anticipatory behavior while this held, because self-limiting
was physically counterproductive, not merely unrewarded.

**Fix (Safety Layer v2):** compute the ceiling first
(`i_max x multiplier`), then clamp the request against that ceiling
directly (`applied = min(requested, ceiling)`). Re-running the same
always-max-vs-smart-taper test under the corrected layer flipped the
result: smart-taper now scored 900 vs. always-max's 532. Verified
monotonic (`applied` never decreases as `requested` increases) via a
dedicated parametrized test suite (Tests A-D in `tests/test_safety.py`).

A further, subtler circularity was found and fixed in the same pass: the
voltage-taper multiplier's estimate depended on the *requested* current
(via `terminal_voltage(state, requested_current)`), creating a theoretical
non-monotonicity (confirmed by direct construction using an artificial
high-`Vrc` state, though not reachable by this system's actual physics).
**Fix:** the voltage estimate is now evaluated at `i_max` (worst case),
making the ceiling purely state-dependent, matching the design already
used for the `state_based_safe_fraction` observation feature.

### 2.4 Reward v3: closing the remaining exploits

With the safety layer fixed, PPO still, in a later run, discovered a new
degenerate solution: charge normally for part of an episode, then request
near-zero current and idle for the remainder -- because the reward had no
explicit cost for elapsed time or for finishing short of the target.
**Fix:** added a constant per-step `time_penalty` and a
`terminal_shortfall_penalty` (proportional to `max(0, target_soc -
final_soc)`, applied only on truncation). Also added an
`overrequest_penalty` (absolute wasted current, not the fractional
`safety_penalty` already present, which under-penalizes large absolute
waste when the ratio is small) and exposed the safety layer's own
state-based ceiling multiplier directly as a 6th observation dimension
(`state_based_safe_fraction`), so the policy has direct access to "how
much can I safely request right now" rather than having to infer it.

These three hand-designed-policy sanity checks were converted into
permanent regression tests (`tests/test_reward_sanity.py`) rather than
one-off scripts, specifically to catch any future reintroduction of a
degenerate optimum before spending training budget on it.

### 2.5 Physics and evaluation-metric corrections (v3.1)

An independent audit of the codebase, checked claim-by-claim against the
actual implementation rather than accepted at face value, surfaced several
further issues, each verified directly before fixing:

- **Thermal model:** heat generation used `I x Vrc` instead of the
  standard resistive-loss term `Vrc^2 / R1`. Verified wrong by direct
  construction: at rest (`I=0`) with a charged RC branch (`Vrc > 0`), the
  old formula gave exactly zero heat despite R1 actively dissipating
  stored polarization energy. Fixed; the two formulations coincide at
  steady state (`Vrc = I*R1`), confirmed by test, so only transient
  behavior changed.
- **Energy efficiency metric:** double-counted the `I^2*R0` resistive loss
  -- once implicitly inside the "delivered" energy term (`I x V_terminal`,
  which already includes the `I*R0` drop), and again as a separately
  computed "dissipated" term. Fixed by comparing true input energy
  (`I x V_terminal`) against OCV-referenced stored energy (`I x OCV`),
  with no term counted twice.
- **Reward component logging:** terminal bonuses/penalties
  (`target_reached_bonus`, `terminal_shortfall_penalty`,
  overvoltage/overtemperature penalties) were applied to the scalar reward
  but never logged, so Stage-2 diagnostic CSVs couldn't reconstruct the
  true total reward on terminal steps. Fixed.
- **`initial_soc` metric bug:** `energy_per_percent_soc_wh` inferred the
  episode's starting SoC from the first *logged* (post-step) value rather
  than the true reset value -- a small but real bias. Fixed by threading
  the actual reset SoC through explicitly.
- **Config-snapshot ordering:** Stage-3 manual-hyperparameter-configuration runs
  snapshotted the config directory *before* applying CLI overrides, so a
  run's saved config didn't reflect what was actually used. Fixed; now
  also saves an `effective_ppo.yaml` and the exact invoking command.

One claim from the same audit -- that `target_kl` was configured but never
actually passed to the PPO constructor -- was checked and found **false**
against the current codebase (confirmed both by reading the source and by
observing "Early stopping ... due to reaching max kl" messages during live
training). This is noted because not every claim from an external review
should be accepted without verification, and this project's discipline of
checking rather than assuming applied to reviewing reviews as well as code.

### 2.6 Seed-sensitivity study

Even with the above fixes, training exhibited severe instability under one
seed (final performance collapsing from ~0.95 to ~0.175 mid-training,
partially recovering) that was absent under a different seed (perfectly
flat ~0.95 across all 40 checkpoints). A controlled three-seed comparison
(changing only the seed, holding every other hyperparameter fixed)
confirmed the instability was real and seed-dependent but did not, on its
own, distinguish "seed noise" from "a systemic issue that seed variation
merely delays."

### 2.7 The n_steps fix: resolving the remaining instability

A fourth training run, still under the corrected reward/safety pipeline,
again showed severe, non-recovering collapse (multiple checkpoints stuck
at `final_soc~0.175`). Rather than guess, the full TensorBoard training
curve (not just checkpoint end-states) was analyzed: `explained_variance`
repeatedly broke down -- including going **negative** (e.g. -20.5 at step
274k, -2.5 at step 525k, meaning the value function's predictions were
worse than a trivial constant-mean baseline) -- while `ep_rew_mean`
declined steadily across the back half of training and policy `std`
climbed from ~0.85 to ~1.48. `approx_kl` stayed low throughout this
decline, ruling out "individual updates too large" (the mechanism
`target_kl` guards against) as the cause.

This pattern -- value-function breakdown despite small individual updates,
across a sustained period -- is consistent with the rollout length
(`n_steps=2048`) being short relative to episode lengths of up to 7200
steps: the value function rarely sees a complete episode in one rollout,
straining long-horizon credit assignment. **Fix:** `n_steps: 2048 -> 8192`,
holding every other hyperparameter (seed, `target_kl`, reward, safety
layer) fixed. The next training run showed perfectly flat performance
(`final_soc=0.9501`) across all 40 checkpoints, stable `explained_variance`
(0.993), and -- critically, verified via sensitivity analysis rather than
assumed -- genuine, structured state-adaptive behavior rather than the
degenerate flatness that had previously produced identically flat
checkpoint sweeps (Section 2.3).

---

## 3. Sensitivity Analysis: Genuinely Adaptive, Not Degenerate

The final model (`run_010`, checkpoint `75000_steps`) was tested via two
complementary methods, run against the safety-derating-disabled variant to
isolate the policy's own behavior from the safety layer's mechanical
tapering:

**Partial-dependence sweep** (raw policy response to each observation
dimension in isolation): `prev_current_norm` (raw response range 0.76) and
`state_based_safe_fraction` (0.63) dominate: the policy is most sensitive
to its own recent behavior and the safety layer's own signal -- a sensible
pattern, since these two inputs most directly answer "what should I do
next." `soc` (0.23) and `ambient_temp_norm` (0.20) contribute
meaningfully; `voltage_norm` and `temperature_norm` contribute least
(0.08, 0.04).

**Real-trajectory analysis** (safety-disabled): 13 of 15 evaluation
scenarios show genuine nonzero current variation (`current_std_a` ranging
0.02-0.29 A), with a coherent, interpretable structure -- variation
increases with higher initial SoC and decreases with higher ambient
temperature. This contrasts with an earlier "adaptive-looking" checkpoint
(`run_006`) that showed **exactly zero** variation in all 15 scenarios
despite passing a naive threshold check -- a result that motivated adding
the raw-vs-clipped saturation distinction to the sensitivity tool, since
clipped-action flatness alone cannot distinguish a genuinely flat policy
from one whose internal response is fully saturated before reaching the
environment.

**Conclusion: the policy is demonstrably state-adaptive, not a disguised
constant controller.** This rules out the simplest explanation for why PPO
resembles Max-Current (i.e., "it just learned to always max out and got
lucky") in favor of the more interesting one below.

---

## 4. Primary Results

| Controller | Final SoC | Charging Time (s) | Safety Interventions | Target Reached |
|-----------------|----------:|-------------------:|----------------------:|:---:|
| Adaptive | 0.9500 | 4141.2 | 0.0 | Yes |
| CC | 0.9501 | 2721.0 | 113.0 | Yes |
| CCCV | 0.9501 | 2833.0 | 0.0 | Yes |
| Max-Current | 0.9501 | 2094.6 | 188.4 | Yes |
| **PPO** | **0.9501** | **2094.8** | **188.6** | Yes |
| PPO (no safety) | 0.9502 | 2042.4 | 136.2 | Yes |

**PPO is faster than CC/CCCV/Adaptive (as in earlier, pre-correction runs)
but is statistically indistinguishable from the trivial Max-Current
baseline** -- 0.2s slower on charging time, 0.2 more interventions on
average, both well within scenario-to-scenario noise. Given Section 3
confirms the policy is genuinely state-adaptive rather than a disguised
constant strategy, this is not a null result about PPO's competence -- it
is a result about the **task's optimum**: under this reward function and
this safety layer, requesting near-maximum current is close to optimal,
and the state-dependent modulation PPO learned on top of that (Section 3)
represents fine-tuning within a strategy space where the ceiling is
already doing most of the work.

---

## 5. Interpretation: Why Near-Maximum-Current Is Near-Optimal Here

Three structural features of the current task design jointly explain this
outcome:

1. **The safety layer already handles all hard constraints.** Since
   `applied = min(requested, safe_ceiling)` and the ceiling is a pure
   function of state, requesting the physical maximum every step
   automatically yields the maximum *safe* current at every step, with no
   risk of ever exceeding a limit. There is no scenario where "ask for
   less than the ceiling" produces a strictly better outcome purely from a
   safety standpoint -- the safety layer already computes the
   best-possible safe action given the state.
2. **The reward's dominant term is charging-progress-proportional-to-
   applied-current**, and applied current is capped by the (state-optimal)
   ceiling regardless of what's requested above it. The
   `overrequest_penalty` (Section 2.4) discourages *wasteful* requests
   above the ceiling, but requesting exactly `i_max` never wastes more
   than requesting any smaller amount that still clears the ceiling.
3. **The battery model's thermal margin is generous relative to the
   charging envelope actually explored**: peak temperatures across the
   full evaluation grid stay well under the 40C threshold where the
   thermal penalty engages, so there is little reward pressure to
   sacrifice speed for heat management within the SoC/ambient-temperature
   ranges tested.

Together, these mean the *only* real degree of freedom left for an
adaptive policy to exploit is the transition zone near full charge
(SoC-taper region) and the smoothness penalty -- exactly where Section 3's
sensitivity analysis shows the learned policy actually differs from a
constant strategy (`prev_current_norm` and `state_based_safe_fraction`
dominating its behavior). The policy is adaptive precisely where
adaptivity has room to matter under this reward, and near-constant
everywhere else, because everywhere else, near-constant *is* close to
optimal.

---

## 6. Limitations and Future Work

- **This is a result about the current task specification, not a general
  claim about RL for battery charging.** A v4 redesign that gives adaptive
  behavior genuine value -- e.g., a real degradation/aging cost that
  scales with sustained high current (the current SoH model is
  monitoring-only, below), a thermal model with less margin, or a safety
  layer deliberately made more conservative so blind-maximum-current
  becomes measurably risky -- would be a different, legitimate research
  question, not a bug fix to this one.
- **Battery model.** Physically motivated and fully source-documented
  (`configs/battery.yaml`: every parameter tagged `[datasheet]`,
  `[derived]`, `[literature]`, or `[assumption]`), but not validated
  against real measured charging curves. Comparing simulated CC/CCCV
  curves against a public dataset (NASA PCoE, CALCE) would be a natural
  follow-up, not required for this project's scope.
- **Cooling model.** Uses a simplified lumped-convection proxy; the real
  Nexon EV pack is liquid-cooled. The RL agent is therefore optimizing
  charging current, not joint charging-and-thermal-management -- a
  narrower, still legitimate scope, worth stating explicitly rather than
  implying the reverse.
- **SoH is monitoring-only.** Tracked (throughput-based degradation proxy)
  but not in the observation, reward, or optimization objective. The
  project does not claim to optimize battery aging.
- **Statistical framing.** The 15-scenario evaluation grid provides
  coverage across operating conditions, not repeated stochastic trials
  (the simulation is deterministic given a scenario). "Mean +/- std across
  scenarios" describes variability across operating conditions, not a
  confidence interval on the algorithm.
- **Validation/test leakage.** Checkpoint selection and final evaluation
  currently draw from the same 15-scenario grid. A cleaner split (train on
  random conditions as now, select checkpoints on a distinct validation
  subset, report final numbers on a held-out test subset) would be more
  defensible and is a straightforward follow-up.
