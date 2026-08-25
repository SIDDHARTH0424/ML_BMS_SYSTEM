# Driving Energy Management — `environment/ev_energy_env.py`

Integrated Gymnasium environment for the driving EMS extension.
Does NOT replace `environment/battery_env.py` (the existing charging
environment, unmodified and still fully functional — 119 original tests
still passing, see `audit/stable_v3_final_validation.md`).

## Architecture

```
drive cycle -> vehicle forces -> wheel power -> drivetrain power
    -> PPO action -> desired battery power -> bidirectional safety
    layer -> feasible current -> ECM step -> reward -> observation
```

Hierarchical by construction: PPO proposes a *desired* battery power;
`safety.safety_layer_bidirectional()` (new, additive — see
`audit/vehicle_integration_plan.md`) enforces what's actually feasible.
PPO cannot bypass the safety layer — there is no code path that applies
an unclamped current to the ECM.

## Observation space (11-dim, `Box(-1, 1)`, exact order documented in
`environment/ev_energy_env.py`'s `OBSERVATION_FIELDS`)

```
0. soc                     [0,1]
1. voltage_norm             (V - v_min)/(v_max - v_min), clipped [0,1]
2. temperature_norm         T / t_max_c, clipped [0,1]
3. prev_battery_power_norm  signed, prev applied power / max(charge,discharge power)
4. speed_norm                v / 30 m/s (normalization-only reference)
5. acceleration_norm         a / 3 m/s^2 (normalization-only reference), clipped [-1,1]
6. grade_norm                 grade_rad / (pi/6), clipped [-1,1]
7. wheel_power_norm           P_wheel / max_discharge_power_w, clipped [-1,1]
8. available_regen_norm       available_regen_w / max_charge_power_w, clipped [0,1]
9. ambient_temp_norm          ambient_c / t_max_c, clipped [0,1]
10. trip_progress             current_time / total_cycle_time, [0,1]
```

No future drive-cycle values are exposed (task §17) — every field above
is computed from the environment's *current* index only.

## Action space

`Box(-1, 1)`, 1-dim, controls **desired battery power** (not raw current,
not motor torque, not accelerator pedal — task §18). Convention, verified
against the existing project rather than copied from the master task
prompt's illustrative example (which used the opposite sign):

```
action > 0  -> desired CHARGE power (into battery)
action < 0  -> desired DISCHARGE power (out of battery, propulsion)
```

This matches the existing project-wide convention ("positive current =
charging", `environment/ecm_model.py`). Mapping is asymmetric
(`max_desired_charge_power_w=25kW` vs. `max_desired_discharge_power_w=106.4kW`,
`configs/energy_management.yaml`) since the physical charge/discharge
power limits differ.

## Known, documented simplification

`power_deficit_w` (when the battery can't fully meet propulsion demand)
is computed and reported but **does not feed back into vehicle speed** —
the drive cycle's prescribed speed/acceleration is treated as achieved
regardless. A closed-loop driver/vehicle response to a power shortfall is
explicitly out of scope for this phase (task §20/§21: "do not build a
sophisticated driver-controller model yet").

## Reward

Five logged components (`tracking_error`, `energy_cost`,
`regen_recovery`, `thermal_stress`, `safety_penalty`) — weights in
`configs/energy_management.yaml` are an **explicitly unvalidated first
pass**, not a tuned reward (task §19's discipline: log and measure scale
before normalizing/trusting weights — the same approach used for the
charging Stable V3 thermal reward).

## Validation

- `tests/test_ev_energy_env.py`, 12 tests (task §27) — all passing.
- Rule-based baseline (`baselines/rule_based_ems.py`, task §21) validated
  end-to-end over a real episode: SoC decreases during propulsion,
  increases during regen braking, exactly as expected physically.
- **First-class driving-EMS metrics** (`utils/metrics.py`,
  `training/evaluate_drive_ems.py`): Wh/km, discharge/regen/net energy
  breakdown, regen recovery fraction, min SoC, max/avg temperature,
  safety interventions — replacing charging-time-style metrics, which
  don't transfer meaningfully to a driving problem. 7 tests
  (`tests/test_driving_ems_metrics.py`), all passing.
- PPO smoke test (`training/train_drive_ems.py --smoke-test`, task §9):
  environment init, PPO init, short training (2048 steps), checkpoint
  save/load, full-rollout finiteness, safety-layer reachability — all
  passed. No multi-seed diagnostic or longer training has been run yet
  (requires real drive-cycle data, not yet sourced — see
  `data/drive_cycles/README.md`).
