# RL-BMS Model Contract

Purpose: a short, unambiguous reference for what flows into and out of the
environment, so future contributors (human or AI) don't have to
reverse-engineer it from `battery_env.py`. See `docs/model_specification.md`
for the full equations.

## Inputs (observation, 6-dim, all normalized to `[0,1]`)

1. SoC
2. Terminal voltage
3. Pack temperature
4. Previous applied current
5. Ambient temperature
6. State-based safety multiplier (SoC + temperature only; excludes
   voltage taper — see spec Section 5)

## Action (1-dim, `[-1, 1]`)

Requested charging current, remapped internally to `[0, i_max]` amps.
**Not** already safety-clamped — clamping happens inside the environment,
after the action is received.

## Safety transformation

```
requested_current (from action)
        │
        ▼
safety_layer(): safe_ceiling = i_max * min(temp_mult, soc_mult, volt_mult)
        │
        ▼
applied_current = clamp(requested_current, 0, safe_ceiling)
```

`volt_mult` uses a voltage estimate evaluated at `i_max` (worst case), not
at the actual request — see spec Section 4 for why.

## State transition

```
applied_current
        │
        ▼
BatteryECM.step(): SoC, V_rc, temperature updated via 1RC ECM + thermal model
        │
        ▼
next_state (+ terminal_voltage recomputed at applied_current)
```

## Reward

```
next_state, requested_current, applied_current, safety_info
        │
        ▼
_compute_reward(): 7 per-step components (see audit/reward_audit.md and,
for the 7th, audit/stable_v3_report.md — a config-gated state-aware
thermal term, `reward.yaml: thermal_enabled` default false)
        │
        ▼
+ terminal bonus/penalty on episode end (target reached / overvoltage /
  overtemperature / truncated-short-of-target)
        │
        ▼
total scalar reward
```

## Outputs (per `step()`)

- `obs` — the 6-dim observation above, for the *next* state
- `reward` — scalar total
- `terminated` / `truncated` — episode-end flags
- `info` dict: `safety_intervention` (full `SafetyInfo.as_dict()`),
  `terminal_voltage`, `termination_reason`, `reward_components` (dict, all
  6 per-step + 4 terminal components, zeroed unless active that step),
  `applied_current_a`, `target_reached` (bool), `final_soc_if_ended`

## What this environment is (and is not)

Per `README.md`'s already-accurate framing: a physics-based RL EV battery
**charging/control** simulation with a 1RC ECM, thermal model, PPO
controller, and rule-based safety constraints. It is **not** a full
vehicle energy-management system, does not model driving/discharge, and
does not optimize SoH/degradation (monitoring only — see
`audit/ISSUES.md`).
