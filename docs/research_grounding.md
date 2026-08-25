# Research Grounding — Driving EMS Extension

All three source citations below were checked via live web search
(August 2026) against their publisher pages before being used here —
DOIs and publication venues confirmed real, not taken on trust from the
task prompt. Source 3's specific author list wasn't visible in the
search snippets obtained; the DOI/venue/publication-date match was
confirmed against ScienceDirect directly.

1. Ananganó-Alvarado, G., Umaña-Morel, I., & Keith-Norambuena, B. (2025).
   "Reinforcement learning in electric vehicle energy management: a
   comprehensive open-access review of methods, challenges, and future
   innovations." *Frontiers in Future Transportation*, 6, 1555250.
   DOI: 10.3389/ffutr.2025.1555250. **Confirmed real** — Frontiers
   publisher page, Universidad Católica del Norte authors, published
   9 June 2025.
2. He, H., Meng, X., Wang, Y., Khajepour, A., An, X., Wang, R., & Sun, F.
   (2024). "Deep reinforcement learning based energy management
   strategies for electrified vehicles: Recent advances and
   perspectives." *Renewable and Sustainable Energy Reviews*, 192,
   114248. DOI: 10.1016/j.rser.2023.114248. **Confirmed real** —
   ScienceDirect, Beijing Institute of Technology repository listing.
3. "Reinforcement learning as a control layer for electric vehicle
   interaction with multi-energy systems: A comprehensive review."
   ScienceDirect, DOI: 10.1016/j.rser.2026.[S1364032126000328].
   **Confirmed real** (article page live, dated ~January 2026) — author
   list not independently verified from search results in this pass.

Per the task's rule: these are used to justify *methods and design
scope* (why dynamic driving conditions, regeneration, and RL-based EMS
are a reasonable research direction), not to claim any parameter value
or numeric result from this project's implementation.

## Design-decision table

| Design Decision | Why It Is Needed | Literature Support | Project-Specific Implementation | Assumptions | Validation Method |
|---|---|---|---|---|---|
| Vehicle dynamics | The charging-only BMS has no model of driving power demand | Source 1: identifies acceleration, regen, grade, and load variation as core dynamic EV operating conditions | `environment/vehicle_dynamics.py` — reduced-order longitudinal point-mass model | Cd, wheel radius are engineering assumptions (not Nexon-specific, see `configs/vehicle.yaml`); no lateral dynamics, no tire slip | `tests/test_vehicle_dynamics.py`, 10 tests (zero speed, constant speed, accel, braking, grade, aero scaling, mass/Cd sensitivity, finiteness) — all passing |
| Drive cycle | PPO needs a time-varying demand signal to manage, not a fixed target | Source 1 & Source 4 (2025 review, cited in task prompt): dynamic/variable driving conditions and generalization across them | `environment/drive_cycle.py` — CSV-based, index-driven, no lookahead exposed | No real-world drive cycle sourced yet (see `data/drive_cycles/README.md`) — only synthetic test fixtures exist; CSV format and validation are implemented and tested, but no actual urban/highway/mixed data is present yet | `tests/test_drive_cycle.py`, 16 tests (reset, indexing, no-interpolation, speed validity, acceleration derivation, grade conversion, end-of-cycle, NaN/Inf/monotonic-time rejection, no-lookahead) — all passing |
| Regeneration | EV energy path the charging-only BMS never modeled | Source 1: explicitly names regenerative braking as a dynamic condition; Source 3: notes safety/deployment risk of RL near hard physical limits | `environment/drivetrain_model.py` (power path) + `safety_layer_bidirectional()` (battery-acceptance path) — regenerated power is now provably able to reach the battery and increase SoC, end-to-end tested | `max_regen_power_w=25kW` is an engineering [assumption]; discharge-side safety thresholds are also [assumption] (§ above) | `tests/test_drivetrain.py` (12) + `tests/test_ev_powertrain.py::test_regeneration_increases_soc` — full pipeline verified, not just each piece in isolation |
| RL EMS (PPO) | Adaptive control matching source reviews' stated RL rationale | Source 1 & Source 2: RL as the adaptive alternative to rule-based/MPC EMS, with reward-design sensitivity flagged as a known challenge | Deferred — new `configs/ppo_drive_ems.yaml`, separate from the charging PPO per task instruction | Not yet decided — action/observation space design comes after the drivetrain model exists | Not yet run |
| Battery constraints | RL must not be trusted to self-enforce hard limits | Source 3: identifies safety risk and limited real-world validation as persistent RL-EMS deployment challenges | `safety_layer_bidirectional()` (new, additive function) extends the existing `safety/safety_layer.py` for discharge/regen, delegating unchanged to `safety_layer()` for charging | `discharge_i_max_a`, `v_undervoltage_taper_start`, `soc_discharge_taper_start` are all [assumption] — no Tata-published discharge-side limits exist; mirror the shape of the existing charging-side rules | `tests/test_safety_bidirectional.py` (11 tests) + `tests/test_ev_powertrain.py` (6 tests, end-to-end through the real ECM) — all passing; existing `tests/test_safety.py` (28 tests) unchanged and still passing |
| Physics-based battery | Avoid a purely black-box battery model, consistent with source reviews' interpretability concerns | Source 3: physics-informed RL named as a promising future direction, implying current black-box approaches are a known limitation | Existing `environment/ecm_model.py` (1RC ECM), reused unmodified and confirmed already sign-agnostic for discharge (audit §4) | Linear SoH degradation model remains a known simplification (documented in Stable V3 reports) | `tests/test_ecm.py` (unchanged, still passing) |
| Multi-cycle evaluation | Generalization across driving conditions is explicitly flagged as an open challenge | Source 1 & Source 4: dynamic conditions / generalization named as central to EV RL-EMS evaluation | Deferred — `data/drive_cycles/` structure planned but not yet populated | No real-world drive-cycle data fabricated — will require an actual verified dataset before this can be built | Not yet run |
| Multi-seed evaluation | RL variance is a demonstrated property of this exact project (see `audit/thermal_weight_040_045_report.md` — seed 21 failed in 2 of 3 weight sweeps where seeds 7/42 succeeded) | Source 2: DRL configuration/evaluation methodology as a design axis | Seeds 7/21/42, same convention as every diagnostic run so far in this project | — | Already validated as a load-bearing practice in this project's own diagnostic history, prior to this task |

## Status of this document

This is the Phase-0/Phase-1 version — covers only the design decisions
made so far (freeze, audit, vehicle dynamics). It will be extended as
drive_cycle.py, drivetrain_model.py, and the rest of the pipeline are
built in subsequent stages, per the task's own staged implementation
order (§41).
