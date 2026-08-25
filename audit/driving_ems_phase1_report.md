# Driving EMS — Phase 1 Report

## 1. Existing charging-BMS baseline

Stable, tested, frozen (`stable-v3-frozen` tag, commit `70085fc`), 119
tests passing before any driving-EMS work began. See
`audit/stable_v3_final_validation.md`. Not modified by this extension —
confirmed by the full regression suite still passing (186 tests as of
this report, 119 original + 67 new, 0 failures) throughout every stage
of this work.

## 2. Why driving EMS is being added

The existing system is charging-focused only. The extension adds full
vehicle driving-energy management (vehicle dynamics → drivetrain →
regeneration → battery), as a separate, additive subsystem — not a
replacement of the charging BMS (task §39, verified: charging tests
untouched throughout).

## 3. Research foundation

Three citations, all independently verified real via live web search
(DOIs/publisher pages confirmed, not taken from the task prompt on
trust) — see `docs/research_grounding.md` and `docs/references.md`.
Used to justify method/design choices only, never to claim project
results.

## 4. Vehicle architecture

```
drive cycle -> vehicle forces -> wheel power -> drivetrain power
    -> PPO action -> desired battery power -> bidirectional safety
    layer -> feasible current -> ECM step -> reward -> observation
```
See `docs/driving_energy_management.md` for the full diagram and detail.

## 5. Vehicle equations

`F_accel = ma`, `F_roll = mgCrr`, `F_aero = 0.5·ρ·Cd·A·v²`,
`F_grade = mg·sin(θ)`, `F_tractive = ΣF`, `P_wheel = F_tractive·v`. See
`docs/vehicle_dynamics.md`.

## 6. Vehicle parameters

Fully sourced and classified — see `docs/vehicle_model_assumptions.md`.
Summary: `mass_kg` [datasheet], `frontal_area_m2` [derived],
`drag_coefficient` [assumption, flagged], `rolling_resistance_coefficient`
[literature], `wheel_radius_m` [assumption], `air_density_kg_m3`/`gravity_m_s2`
[literature]. Nothing presented as manufacturer data unless actually found
published.

## 7. Drive-cycle format

CSV (`time_s`, `speed_mps`, optional `acceleration_mps2`/`road_grade_deg`).
**No real drive-cycle dataset has been sourced yet** — only synthetic test
fixtures exist, explicitly labeled as such (task §10's "do not fabricate
real driving cycles" rule). See `data/drive_cycles/README.md`.

## 8. Drivetrain model

Reduced-order efficiency model, separate propulsion/regen efficiencies
(`configs/drivetrain.yaml`), motor-power and regen-power caps. See
`docs/vehicle_model_assumptions.md` for sourcing, `environment/drivetrain_model.py`.

## 9. Regenerative braking

Implemented in the drivetrain model (available regen power, capped,
lossy) and the new bidirectional safety layer (battery acceptance).
No-energy-creation explicitly verified across a range of braking
magnitudes (`tests/test_drivetrain.py::test_no_energy_creation_across_range`).
Unused regen is explicitly accounted as `friction_braking_w`, never
silently discarded.

## 10. Battery power coupling

`P = I/V` at the pre-step terminal voltage estimate, following the
project's documented sign convention (positive = charging) — verified,
not guessed (`audit/vehicle_integration_plan.md`). Discharge-direction
power converts to negative current; charge/regen-direction converts to
positive current, matching the existing convention exactly.

## 11. Safety integration

New, additive `safety_layer_bidirectional()` function
(`safety/safety_layer.py`) — delegates to the existing `safety_layer()`
unchanged for any non-negative request (verified byte-for-byte identical
in `tests/test_safety_bidirectional.py`), adds new discharge-side
temperature/SoC/undervoltage derating for negative requests. PPO's
action can never bypass this layer — there is no direct action-to-ECM
path in `ev_energy_env.py`.

## 12. Observation space

11-dim, fully documented and ordered (`environment/ev_energy_env.py`'s
`OBSERVATION_FIELDS`). No future drive-cycle values exposed. See
`docs/driving_energy_management.md`.

## 13. Action space

1-dim, `[-1,1]`, controls desired battery power (not raw current, motor
torque, or pedal position), asymmetric charge/discharge mapping. Sign
convention **verified against the existing project**, not copied from
the task prompt's illustrative example (which used the opposite sign).

## 14. Reward design

Five components, logged individually every step
(`tracking_error`, `energy_cost`, `regen_recovery`, `thermal_stress`,
`safety_penalty`). Weights are an **explicitly unvalidated first pass**
(`configs/energy_management.yaml`) — not tuned, per task §19's
log-then-normalize discipline, matching how the charging Stable V3
thermal reward was handled.

## 15. Baseline controller

`baselines/rule_based_ems.py` — direct power-following: full discharge
during propulsion, full regen-use during braking, else neutral.
Validated over a real 10-step episode rollout: SoC decreased during
propulsion steps, increased during braking/regen steps, exactly as
physically expected.

**First-class driving-EMS metrics** (per reviewer feedback — charging-time
metrics don't transfer meaningfully to a driving problem). New functions
in `utils/metrics.py` + `training/evaluate_drive_ems.py`, run on the
rule-based baseline over the synthetic fixture:

```
distance_km:              0.057
wh_per_km:                344.93
discharge_energy_wh:      29.72
regen_energy_wh:          10.06
net_energy_wh:            19.66
regen_recovery_fraction:  1.00   (all mechanically-available regen was captured)
min_soc:                  0.4993
max_temperature_c:        25.004
avg_temperature_c:        25.002
safety_interventions:     0
safety_intervention_rate: 0.0
```
These numbers are from the synthetic test fixture only (10-step
accelerate/cruise/brake profile) — not representative of a real driving
condition, same caveat as everywhere else in this report re: no real
drive-cycle data yet. The point of running this now is to confirm the
metrics themselves are correctly computed and will be meaningful once
real drive-cycle data exists, not to draw any conclusion about actual
energy efficiency from this run.

## 16. Tests

74 new tests across 7 new test files:
`test_vehicle_dynamics.py` (10), `test_drive_cycle.py` (16),
`test_drivetrain.py` (12), `test_safety_bidirectional.py` (11),
`test_ev_powertrain.py` (6), `test_ev_energy_env.py` (12),
`test_driving_ems_metrics.py` (7).

## 17. Exact test results

```
python -m compileall .   -> exit 0, no errors (every stage of this work)
pytest -q                -> 193 passed (119 original + 74 new), 0 failures
```
Re-run fresh for this report, not carried over from an earlier claim.

## 18. PPO smoke test

`training/train_drive_ems.py --smoke-test`:
```
env_init: True
obs_finite_on_reset: True
ppo_init: True
ppo_smoke_train: True
num_timesteps: 2048
checkpoint_save: True
checkpoint_load_predict_finite: True
full_rollout_finite: True
safety_layer_reachable: True
```
All checks passed. No NaN/Inf anywhere.

## 19. Multi-seed results

**Not run.** Deferred per the staged order (§30) — a meaningful
multi-seed diagnostic needs real drive-cycle data first (only synthetic
test fixtures currently exist).

## 20. Multi-drive-cycle results

**Not run**, for the same reason — no real urban/highway/mixed/aggressive
drive-cycle datasets have been sourced yet (task §10's fabrication
prohibition).

## 21. Assumptions

Full registry in `docs/vehicle_model_assumptions.md`. Notable ones:
`drag_coefficient=0.35` [assumption, not Tata-published],
`max_regen_power_w=25kW` [assumption, not Tata-published], all discharge
safety thresholds [assumption, mirror the shape of existing charging
rules], all reward weights [unvalidated placeholder].

## 22. Limitations (task §37 — must be disclosed, not hidden)

- Reduced-order vehicle model (no lateral dynamics, no tire slip, no
  suspension).
- Simplified drivetrain (single efficiency factor per direction, no
  detailed motor electromagnetics).
- Simplified regenerative braking (no motor-torque-curve-dependent
  recovery limit, just a flat power cap).
- **No real drive-cycle data** — only synthetic test fixtures, explicitly
  labeled as such.
- No hardware-in-the-loop or real-vehicle validation of any kind.
- Simplified battery degradation model (unchanged from the charging
  system — linear in `|I|·dt`, still cannot distinguish current-pacing
  strategies, still monitoring-only, not in any reward).
- `power_deficit_w` does not feed back into vehicle speed (documented
  simplification, task §20/§21).
- Reward weights are unvalidated (§14 above) — no training run should be
  treated as meaningful until they're diagnosed the way the charging
  thermal reward was.
- PPO hyperparameters in `configs/ppo_drive_ems.yaml` are a smoke-test
  starting point (copied from the charging PPO's known-stable
  configuration), not validated for this different environment's dynamics.

## 23. Research gaps

Per Source 3's own stated future directions (not claimed as done here):
physics-informed RL, degradation-aware RL, hybrid RL+optimization,
uncertainty-aware policies, explainable policies, hardware-in-the-loop
validation — all explicitly out of scope for this phase (task §38).

## 24. Next recommended phase

1. Source and document a real drive cycle (or several, for §20's
   multi-condition requirement) — the single largest gap blocking
   everything downstream (§19, §20, and any meaningful reward tuning).
2. Run the reward-component logging/normalization pass (§19's
   discipline) before trusting any weight in `energy_management.yaml`.
3. Only then: multi-seed short-diagnostic training, following the exact
   methodology already established for the charging PPO (seeds 7/21/42,
   short budgets first, training-curve stability checks before any
   longer run).

---

## Explicit status answers (task §40)

```
Charging BMS preserved?                    YES (119/119 original tests still passing)
Vehicle dynamics implemented?               YES
Drive-cycle interface implemented?          YES
Drivetrain implemented?                     YES
Regeneration implemented?                   YES
Bidirectional battery path validated?       YES (tests/test_ev_powertrain.py, real ECM)
Driving EMS environment implemented?        YES
New PPO created?                            YES (configs/ppo_drive_ems.yaml, separate)
Existing charging PPO modified?             NO
Safety layer modified?                      NO (new additive function only; safety_layer()
                                                 itself is byte-for-byte unchanged, verified)
Battery ECM modified?                       NO (one method exposed via a name, Stable V3 pass;
                                                 unchanged again in this pass)
Real-world validation performed?            NO
```
