# R1/R2/R3 Demo Runbook — RL-BMS-Driving

> **For:** College/project evaluation (R1 = 3, R2 = 3, R3 = 1)  
> **Models used:** A1 50k charging (validated) · B3 100k driving (validated)  
> **Time budget:** ~10–15 minutes live demo

---

## Before You Start

```powershell
cd C:\Project\rl-bms-Driving
.\.venv\Scripts\Activate.ps1
```

---

## Section 1 — Architecture Overview (talk through, 3 min)

Draw or point to this diagram:

```
CHARGING TRACK (Track A)
════════════════════════════════════════════
  Battery ECM (Thevenin model)
    ↑ SoC, V_oc, V_rc, temperature
  ──→ Safety Layer  (derate by temp / SoC / voltage)
  ──→ PPO Charging Controller
        action: I_request ∈ [0, I_max]
  ──→ Reward: charge efficiency − thermal penalty − smoothness penalty
  ──→ Episode ends: SoC ≥ 0.80  OR  max steps

DRIVING TRACK (Track B)
════════════════════════════════════════════
  Drive Cycle (EPA UDDS / HWFET / US06 / WLTP)
    ↑ v_ref(t), a(t), grade(t)
  ──→ Vehicle Dynamics  (drag + grade + inertia → wheel power)
  ──→ PPO EMS Controller
        action: motor power split ∈ [−1, +1]
  ──→ Drivetrain  (propulsion η, regen η)
  ──→ Battery ECM  (SoC update, heat)
  ──→ Reward: regen recovery + efficiency − power deficit
```

Key points to state:
- Both tracks share the **same Thevenin ECM** and **same safety layer**
- Safety layer is a hard constraint — it cannot be bypassed by the agent
- PPO is trained with Stable-Baselines3; all hyperparameters are in `configs/`

---

## Section 2 — Run the Full Test Suite (2 min)

```powershell
python -m pytest tests/ -v
```

Expected output (live):
```
214 passed in ~9s
```

Say: *"214 unit tests covering the ECM model, safety layer, reward components,
drive cycle, vehicle dynamics, and the full environment. Zero failures."*

---

## Section 3 — Live Charging Evaluation (3 min)

Run the validated A1 50k seed 7 model:

```powershell
python -m training.evaluate `
    --model final_models\charging_A1_50k_seed7\trained_model `
    --run-name demo_charging_seed7
```

Point to the output:
- `mean_charging_time_s` ≈ **2114 s** (gate is 2199 s — 4% margin)
- `reached_target_all` = **True**
- `mean_applied_current_a` ≈ **154 A**
- No safety interventions

Results are written to `runs/demo_charging_seed7/evaluation/`.

---

## Section 4 — Live Driving Evaluation (3 min)

Run the validated B3 100k seed 7 model:

```powershell
python -m training.evaluate_drive_ems `
    --controller ppo `
    --model-path final_models\driving_B3_100k_seed7\ppo_driving_100000_steps `
    --all-cycles `
    --config-dir .\configs\final_driving
```

Point to the output table (these are **per-cycle** Wh/km — each row is one drive cycle):

| Cycle | Wh/km | Regen | Safety interventions |
| :--- | :---: | :---: | :---: |
| EPA UDDS | ~86.5 | ~100% | 0 |
| EPA HWFET | ~130.6 | ~100% | 0 |
| EPA US06 | ~179.3 | ~100% | 0 |
| WLTP Class 3b | ~117.8 | ~100% | 0 |

**Cross-cycle arithmetic mean ≈ 128.5 Wh/km** (sum of four Wh/km values ÷ 4).

Then show the rule-based baseline for comparison:

```powershell
python -m training.evaluate_drive_ems `
    --controller rule_based `
    --all-cycles `
    --config-dir .\configs\final_driving
```

Say: *"The PPO EMS matches or slightly exceeds the rule-based controller on
efficiency, with zero safety violations across all four standard drive cycles."*

---

## Section 5 — Honest Research Status (1 min)

> *"The short-horizon validated models work well. We also ran 1M-step long
> training on both tracks. The charging long-training collapsed — the agent
> lost the ability to reliably complete the charge cycle. This is a known
> failure mode of PPO under prolonged training without learning-rate annealing
> or checkpoint selection. We kept that result as a research finding rather
> than deleting it. The driving 1M run produced plausible numbers but was not
> formally gated. The validated A1 50k and B3 100k models are what we claim."*

Reference: `docs/research_findings.md`

---

## Files to Have Open

| File | Why |
| :--- | :--- |
| `environment/bms_env.py` | Show observation space, reward |
| `safety/safety_layer.py` | Show hard constraints |
| `configs/final_driving/` | Show hyperparameters |
| `final_models/README.md` | Show validated model summary |
| `docs/research_findings.md` | Show honest failure analysis |

---

## Key Numbers to Remember

| Metric | Value |
| :--- | :--- |
| Tests passing | **214 / 214** |
| Charging gate (max allowed) | **2199.3 s** |
| Charging A1 50k seed 7 | **2114.5 s** ✅ |
| Charging A1 50k seed 21 | **2095.0 s** ✅ |
| Charging A1 50k seed 42 | **2095.1 s** ✅ |
| Driving B3 100k cross-cycle mean Wh/km | **128.5–129.3 Wh/km** ✅ |
| Driving B3 100k WLTP-only Wh/km | **117.8–119.5 Wh/km** (per-cycle, lower because WLTP is longer) |
| Safety interventions | **0** across all cycles and all seeds |
| 1M charging model | ❌ Fails target reach — kept as research finding |
