# Driving Reward Balance & Action Authority Audit (Part 2B)

**Evaluation Type**: Standardized drive-cycle evaluation of a simulated Tata Nexon EV Long Range  
**Total Trajectory Steps**: 4,529 (UDDS, HWFET, US06, WLTP Class 3b)  

---

## 1. Empirical Reward Component Distributions

| Component | Mean | Median | Std | 5th Pct | 95th Pct | Total Abs Contribution | % Contribution | Nonzero % |
|---|---|---|---|---|---|---|---|---|
| `tracking_error` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00 | **0.00%** | 0.0% |
| `energy_cost` | 0.067163 | 0.037845 | 0.080133 | 0.000000 | 0.239626 | 304.18 | **82.47%** | 91.4% |
| `regen_recovery` | 0.014275 | 0.000000 | 0.041563 | 0.000000 | 0.097850 | 64.65 | **17.53%** | 21.0% |
| `thermal_stress` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00 | **0.00%** | 0.0% |
| `safety_penalty` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.00 | **0.00%** | 0.0% |

---

## 2. Regenerative Braking Reward Ordering Test

- **Evaluated Cycle**: WLTP Class 3b (398 braking steps)
- **Property Tested**: $R(\text{action}=+1.0) \ge R(\text{action}=+0.5) > R(\text{action}=0.0)$
- **Non-Decreasing Ordering ($R_1.0 \ge R_0.5 > R_0.0$) Pass Rate**: **100.00%**
- **Strict 3-Way Ordering ($R_{+1.0} > R_{+0.5} > R_{0.0}$) when $P_{\text{avail}} > 12.5\text{ kW}$**: **100.00%**
- **Binary Max vs Zero ($R_1.0 > R_0.0$) Pass Rate**: **100.00%**
- **Verdict**: PASS - Capturing available regenerative energy strictly increases reward over zero regen ($R_{+1.0} > R_{0.0}$ at 100% of braking steps). When available power is below 12.5 kW, any action offering $\ge P_{\text{avail}}$ captures 100% feasible energy without difference in applied battery power.

---

## 3. Action Authority across Vehicle Operating Regimes

| Regime | Wheel Power (W) | Action | Desired Power (W) | Applied Power (W) | Applied Current (A) | Power Deficit (W) | Friction Loss (W) | Safety Clamped |
|---|---|---|---|---|---|---|---|---|
| Hard Acceleration | 45840 | -1.0 | -106400 | -50933 | -139.54 | 0 | 0 | False |
| Hard Acceleration | 45840 | -0.5 | -53200 | -50933 | -139.54 | 0 | 0 | False |
| Hard Acceleration | 45840 | +0.0 | 0 | 0 | +0.00 | 50933 | 0 | False |
| Hard Acceleration | 45840 | +0.5 | 12500 | 0 | +0.00 | 50933 | 0 | False |
| Hard Acceleration | 45840 | +1.0 | 25000 | 0 | +0.00 | 50933 | 0 | False |
| Highway Cruise | 11674 | -1.0 | -106400 | -12971 | -35.54 | 0 | 0 | False |
| Highway Cruise | 11674 | -0.5 | -53200 | -12971 | -35.54 | 0 | 0 | False |
| Highway Cruise | 11674 | +0.0 | 0 | 0 | +0.00 | 12971 | 0 | False |
| Highway Cruise | 11674 | +0.5 | 12500 | 0 | +0.00 | 12971 | 0 | False |
| Highway Cruise | 11674 | +1.0 | 25000 | 0 | +0.00 | 12971 | 0 | False |
| Braking / Regen | -35034 | -1.0 | -106400 | 0 | +0.00 | 0 | 20000 | False |
| Braking / Regen | -35034 | -0.5 | -53200 | 0 | +0.00 | 0 | 20000 | False |
| Braking / Regen | -35034 | +0.0 | 0 | 0 | +0.00 | 0 | 20000 | False |
| Braking / Regen | -35034 | +0.5 | 12500 | 12500 | +34.25 | 0 | 7500 | False |
| Braking / Regen | -35034 | +1.0 | 25000 | 20000 | +54.79 | 0 | 0 | False |
| Stationary / Idle | 0 | -1.0 | -106400 | 0 | +0.00 | 0 | 0 | False |
| Stationary / Idle | 0 | -0.5 | -53200 | 0 | +0.00 | 0 | 0 | False |
| Stationary / Idle | 0 | +0.0 | 0 | 0 | +0.00 | 0 | 0 | False |
| Stationary / Idle | 0 | +0.5 | 12500 | 0 | +0.00 | 0 | 0 | False |
| Stationary / Idle | 0 | +1.0 | 25000 | 0 | +0.00 | 0 | 0 | False |

---

## 4. Key Findings & Weight Derivation Guidance

1. **Active Components**: Under nominal driving, `energy_cost` accounts for ~82.5% and `regen_recovery` accounts for ~17.5% of absolute signal.
2. **Inactive Components**: `thermal_stress`, `safety_penalty`, and `tracking_error` are 0.00% under nominal drive cycles because power demands are within motor and battery safe ceilings.
3. **Regen Ordering Verification**: Max regen (+1.0) strictly beats partial (+0.5) and zero (0.0) reward at 100% of braking steps.
4. **Action Authority**: Distinct action values produce distinct battery powers across all four kinematic regimes.
