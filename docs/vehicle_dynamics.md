# Vehicle Dynamics — `environment/vehicle_dynamics.py`

Reduced-order longitudinal point-mass model. No lateral dynamics, no
suspension, no tire slip — deliberately, per task §7 ("intentionally
simpler than a full vehicle simulator").

## Equations (exactly as implemented, task §7.1)

```
F_accel = m * a
F_roll  = m * g * Crr                          (0 at v = 0)
F_aero  = 0.5 * rho * Cd * A * v^2
F_grade = m * g * sin(theta)
F_tractive = F_accel + F_roll + F_aero + F_grade
P_wheel = F_tractive * v                        (0 at v = 0)
```

## Sign convention

- `P_wheel > 0` → propulsion demand (vehicle needs power at the wheels)
- `P_wheel < 0` → braking/deceleration opportunity (available to the
  drivetrain/regen model — this module doesn't decide how much is used)
- `v = 0` → `P_wheel = 0` exactly (no aerodynamic or motion-dependent
  terms can act)
- `theta` (`road_grade_rad`) is signed, positive = uphill

## Assumption: `speed_mps >= 0` throughout

Rolling resistance and aerodynamic drag are written as magnitudes
opposing forward motion, not signed for reverse travel — consistent with
`environment/drive_cycle.py`'s own validation rule (`speed_mps >= 0`).
Negative speed is not a supported input to this module.

## Parameters

See `docs/vehicle_model_assumptions.md` for the full sourced/classified
parameter table (`configs/vehicle.yaml`).

## Validation

`tests/test_vehicle_dynamics.py`, 10 tests (task §22): zero speed,
constant speed, positive/negative acceleration, uphill/downhill, aero
scaling with speed, mass/Cd sensitivity, and finiteness across a grid of
realistic and edge-case inputs. All passing.
