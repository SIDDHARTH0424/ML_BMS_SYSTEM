# Vehicle Model Assumptions — Consolidated Registry

Every non-charging-BMS parameter introduced for the driving-EMS
extension, in one place, with its source classification. Full derivation
notes live as comments in the source config files; this is the summary.

## `configs/vehicle.yaml` (Tata Nexon EV Long Range)

| Parameter | Value | Classification | Note |
|---|---|---|---|
| `mass_kg` | 1400.0 | [datasheet] | Kerb weight, multiple retailer spec pages |
| `frontal_area_m2` | 2.46 | [derived] | width × height × 0.84 shape factor |
| `drag_coefficient` | 0.35 | [assumption] | No official Tata Cd published — typical compact-SUV range |
| `rolling_resistance_coefficient` | 0.010 | [literature] | Typical passenger-car radial-tire Crr on asphalt |
| `wheel_radius_m` | 0.325 | [assumption] | R16 wheel confirmed [datasheet]; exact rolling radius not published |
| `air_density_kg_m3` | 1.225 | [literature] | Standard ISA sea-level value |
| `gravity_m_s2` | 9.81 | [literature] | Standard value |

## `configs/drivetrain.yaml`

| Parameter | Value | Classification | Note |
|---|---|---|---|
| `motor_max_power_w` | 106400.0 | [datasheet] | Confirmed for the 45kWh Nexon.EV variant (Team-BHP, Autocar India) |
| `propulsion_efficiency` | 0.90 | [literature] | Typical combined motor+inverter+gearbox efficiency, not Nexon-specific |
| `regen_efficiency` | 0.80 | [literature] | Typical regen conversion efficiency, not Nexon-specific |
| `max_regen_power_w` | 25000.0 | [assumption] | Tata confirms 4-level regen exists, publishes no power figure |

## `configs/safety.yaml` (new discharge section — additive, existing keys untouched)

| Parameter | Value | Classification | Note |
|---|---|---|---|
| `discharge_i_max_a` | 160.0 | [assumption] | Mirrors `i_max_a`'s magnitude; motor power is the real binding constraint in practice |
| `v_undervoltage_taper_start` | 320.0 | [assumption] | No Tata-published undervoltage cutoff |
| `v_hard_min` | 300.0 | [datasheet] | Matches `battery.yaml`'s documented 300–420V pack window |
| `soc_discharge_taper_start` | 0.10 | [assumption] | Mirrors the existing high-SoC charge taper's shape |
| `soc_discharge_empty` | 0.00 | — | Definitional (0% SoC = empty) |

## `configs/energy_management.yaml`

| Parameter | Value | Classification | Note |
|---|---|---|---|
| `initial_soc_range` | [0.30, 0.70] | [assumption] | Placeholder for driving episodes |
| `ambient_temp_range_c` | [15, 35] | [assumption] | Matches the charging env's own train range |
| `episode_max_steps` | 1200 | [assumption] | Placeholder — should be set by real drive-cycle length once sourced |
| `w_tracking_error`, `w_energy_cost`, `w_regen_recovery`, `w_thermal_stress`, `w_safety_penalty` | 1.0 / 0.1 / 0.5 / 0.5 / 1.0 | **[unvalidated placeholder]** | Explicitly NOT tuned — see §19 discipline note in the config file itself; requires the same log-then-normalize diagnostic pass used for the charging Stable V3 thermal reward before any real training |

## Normalization-only references (not physical parameters, used only to scale observations to roughly [-1,1] — see `environment/ev_energy_env.py`)

| Reference | Value | Note |
|---|---|---|
| `ASSUMED_MAX_SPEED_MPS` | 30.0 | Generic urban/highway-mix ceiling for observation normalization only |
| `ASSUMED_MAX_ACCEL_MPS2` | 3.0 | Generic passenger-car ceiling for observation normalization only |

## What this registry does NOT contain

No parameter here is claimed to be an exact, verified Tata Nexon EV
specification unless marked `[datasheet]`. Every `[assumption]` entry
should be treated as a placeholder that may need revisiting once real
validation data (a sourced drive cycle, a real efficiency measurement,
etc.) becomes available — consistent with task §37's required limitations
disclosure.
