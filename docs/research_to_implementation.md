# Research-to-Implementation Mapping

See `docs/research_grounding.md` for the citation verification notes and
the fuller assumptions/validation table. This document restates the
mapping in the exact 4-column format requested by task §36.

| Design feature | Literature motivation | Project implementation | Validation experiment |
|---|---|---|---|
| Vehicle dynamics | Source 1: dynamic EV operating conditions (accel, grade, load) | `environment/vehicle_dynamics.py`, reduced-order longitudinal force model | `tests/test_vehicle_dynamics.py`, 10 tests |
| Drive cycle | Source 1 & 4: variable driving conditions / generalization | `environment/drive_cycle.py`, time-series speed/accel/grade, no lookahead | `tests/test_drive_cycle.py`, 16 tests |
| Drivetrain / regeneration | Source 1: regenerative braking as a core dynamic condition | `environment/drivetrain_model.py`, capped/lossy propulsion + regen paths, no-energy-creation enforced | `tests/test_drivetrain.py`, 12 tests |
| Bidirectional battery path | Source 3: safety-critical deployment concern for RL near hard physical limits | `safety_layer_bidirectional()` (new, additive to `safety/safety_layer.py`) | `tests/test_safety_bidirectional.py` (11) + `tests/test_ev_powertrain.py` (6, real-ECM integration) |
| RL EMS (PPO) | Source 1 & 2: adaptive EMS vs. rule-based/MPC, reward-design sensitivity flagged as a known challenge | `environment/ev_energy_env.py` + `configs/ppo_drive_ems.yaml` (separate from the charging PPO) | PPO smoke test (`training/train_drive_ems.py --smoke-test`) — passed; multi-seed diagnostic not yet run |
| Rule-based baseline first | Source 2: rule-based control as the simplest EMS baseline, established before RL comparison | `baselines/rule_based_ems.py`, direct power-following controller | Validated over a real episode rollout (SoC direction correct in both propulsion and regen phases) |
| Safety layer as hard constraint | Source 3: RL should not be trusted to self-enforce physical/safety limits | Existing `safety/safety_layer.py` (charging, unmodified) + new `safety_layer_bidirectional()` (discharge/regen) — PPO's action never reaches the ECM unclamped | `tests/test_safety.py` (28, unchanged) + the two new safety test files above |
| Physics-based battery, not black-box | Source 3: physics-informed RL named as a promising direction, implying current black-box approaches are a limitation | Existing `environment/ecm_model.py` (1RC ECM), reused unmodified; confirmed already sign-agnostic for discharge by direct code audit | `tests/test_ecm.py` (unchanged) |
| Multi-condition / multi-seed evaluation | Source 1 & 4: generalization across driving conditions as an open challenge | Not yet run — requires real, sourced drive-cycle data (deliberately not fabricated, see `data/drive_cycles/README.md`) | Deferred |

Every literature citation above is verified real (DOI/publisher checked
via live web search, see `docs/research_grounding.md`) and is used only
to justify method/design choices, never to claim a specific numeric
result for this project.
