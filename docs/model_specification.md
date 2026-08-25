# RL-BMS Model Specification

Source of truth: this document is transcribed directly from the
implementation (`environment/ecm_model.py`, `environment/battery_env.py`,
`safety/safety_layer.py`, `configs/*.yaml`) as of this audit. Where the
implementation and any prior documentation disagreed, the code is treated
as authoritative (per audit Phase 8 instruction) and this document
reflects the code.

## 1. Battery assumptions

Single lumped 1RC-Thevenin-equivalent cell/pack model. Parameters
(`configs/battery.yaml`): nominal capacity (Ah), R0, R1, C1, an OCV(SoC)
lookup table (linear interpolation), pack mass, specific heat, convection
coefficient, surface area, and safety-relevant bounds (`v_max`, `v_min`,
`t_max_c`, `i_max_a`). **No parameter values were changed in this audit
pass** — see `configs/battery.yaml` for current values and any cited
sources.

## 2. Equivalent Circuit Model (ECM)

Implemented in `environment/ecm_model.py::BatteryECM._derivatives`.
Convention: **positive current = charging.**

- `dSoC/dt = I / (capacity_Ah * 3600)`
- `dV_rc/dt = I/C1 − V_rc/(R1*C1)`
- `V_t = OCV(SoC) + I*R0 + V_rc` (`terminal_voltage`)
- `Q_gen = I²*R0 + V_rc²/R1` (see code comment for why `I*V_rc` is
  deliberately NOT used — it double-counts energy stored in C1 during
  transients)
- `Q_loss = h * A * (T − T_ambient)`
- `dT/dt = (Q_gen − Q_loss) / (mass * cp)`

Integration: Euler (default) or RK4, selected via
`battery.integration_method`. Both paths are tested in
`tests/test_ecm.py::test_euler_and_rk4_agree_closely_for_smooth_dynamics`.

SoC is clamped to `[0, 1]` after every step. SoH bookkeeping
(`ah_throughput`, `soh`) is tracked but **not** fed into the reward (see
Section 7) — monitoring only.

## 3. Thermal model

Single lumped thermal mass with convective loss to ambient, as given by
`Q_loss` above. No spatial gradient, no active cooling model beyond the
convection term.

## 4. Safety model

Implemented in `safety/safety_layer.py::safety_layer` ("v2" semantics,
per its own docstring — see file for the v1→v2 fix history).

1. Compute independent derating multipliers in `[0,1]` from **state only**
   (no dependency on the request): `temp_mult` (linear derate between
   `t_derate_start_c` and `t_hard_cutoff_c`), `soc_mult` (linear derate
   between `soc_taper_start` and `soc_taper_full`), and optionally
   `volt_mult` if an `estimated_voltage` is supplied (linear derate
   between `v_taper_start` and `v_hard_max`).
2. `combined_mult = min(temp_mult, soc_mult, volt_mult)` (most-restrictive
   rule wins; multipliers are not multiplied together, to avoid
   unrealistic compounding).
3. `safe_ceiling = i_max * combined_mult`.
4. `applied_current = max(0, min(requested_current, safe_ceiling))`.

This is monotonic in the request (verified in
`tests/test_safety.py`'s `test_v2_monotonicity_*` tests): the applied
current never decreases as the requested current increases, for fixed
state. It is a **rule-based constraint layer with post-step
verification** — not a mathematically-proven safety guarantee. The
environment additionally hard-terminates an episode if terminal voltage
or temperature exceeds `v_max`/`t_max_c` despite the clamp (see
`_check_termination`), which can happen because the voltage estimate used
for clamping is evaluated at `i_max` (worst case), not at the actual
applied current — see `battery_env.py::step` comment for the specific
non-monotonicity edge case this is guarding against.

## 5. State space (observation)

`environment/battery_env.py`: `Box(low=0, high=1, shape=(6,))`, normalized:

1. SoC (already `[0,1]`)
2. Terminal voltage, normalized by `(V − v_min)/(v_max − v_min)`
3. Temperature, normalized by `(T − 0)/(t_max_c − 0)`
4. Previous applied current, normalized by `i_max`
5. Ambient temperature, normalized the same way as (3)
6. State-based safety multiplier (`state_based_current_multiplier`) — SoC
   and temperature only, **excludes** voltage tapering (see code comment:
   voltage tapering depends circularly on the request, so it's omitted
   from this observation feature; SoC tapering is a reasonable proxy for
   it in this system's dynamics)

## 6. Action space

`Box(low=-1, high=1, shape=(1,))` — symmetric, **not** `[0,1]`. Remapped
in `step()`: `action_val = (raw_action + 1)/2`, then
`requested_current = action_val * i_max`. The symmetric range is a
documented fix for Gaussian-policy saturation at an asymmetric `[0,1]`
boundary (see code comment and `results_and_discussion.md` Section 2.1).

## 7. Reward

See `audit/reward_audit.md` for the full per-component breakdown, weights,
and magnitude analysis. Total = `charging_progress − temperature_penalty
− safety_penalty − overrequest_penalty − smoothness_penalty − time_penalty
− thermal_reward`, plus one-time terminal bonuses/penalties on episode end.
**SoH is not a reward term** — confirmed by reading `_compute_reward` in
full; no reference to `soh` or `ah_throughput` appears anywhere in the
reward computation.

**Stable V3 (state-aware thermal reward):** `thermal_reward` is
config-gated (`reward.yaml: thermal_enabled`, default `false` — baseline
reward is unchanged when disabled). When enabled: quadratic in normalized
temperature excess above `thermal_reference_temp_c` (reused from the
existing `temperature_penalty_start_c`, 40°C), linear in the ECM's own
`heat_generation_w` normalized against `thermal_q_reference_w`. See
`audit/stable_v3_report.md` for the full equation, reference-value
derivation, tests, and diagnostic finding that this environment's normal
operating range (~25–32°C) never crosses 40°C, so the term is currently
inactive at its default reference — a known, documented limitation, not a
bug.

## 8. Training configuration

PPO via Stable-Baselines3, `agents/train_ppo.py`. Hyperparameters from
`configs/ppo.yaml` (learning rate uses a linear-decay schedule — see
`linear_schedule`). Four independent stages (sanity / reward-verification
/ manual HPO / full training) — see `audit/STAGE_PIPELINE.md`.

## 9. Evaluation metrics

`utils/metrics.py::summarize_episode` — see `audit/reward_audit.md` and
`utils/metrics.py` docstrings for the energy-accounting and
voltage-stability caveats already documented in-code (both already
correctly named/scoped as of this audit; no metric-definition change was
needed beyond `aggregate_runs`, see ISSUE-002).

## 10. Limitations

- Single lumped-parameter cell model, no cell-to-cell variation.
- No degradation/aging feedback into control (SoH is monitored, not
  optimized — see `audit/ISSUES.md` and `README.md`).
- No results in this project are currently verifiable against real
  training artifacts — see ISSUE-005.
- Voltage tapering in the safety layer depends on a worst-case (`i_max`)
  voltage estimate, not the actual requested current, as a deliberate
  trade-off to preserve monotonicity (see Section 4 and code comments for
  the specific edge case this avoids).
