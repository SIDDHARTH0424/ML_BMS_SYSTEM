# Drive Cycles

## Status: no real-world drive-cycle data included yet

Per the master task's explicit rule ("DO NOT fabricate real driving
cycles"), this directory does not contain a UDDS, WLTP, NEDC, or any
other named real-world drive cycle — none has been sourced and verified
yet. Adding one requires downloading an actual verified dataset (e.g.
from the EPA, WLTP facility, or a published, licensed source) and
documenting its license/usage terms here before use — not done in this
phase.

The only files present under `tests/fixtures/` are **synthetic test
fixtures**, explicitly named and documented as such, used only to
validate `environment/drive_cycle.py`'s parsing/validation logic
(`tests/test_drive_cycle.py`). They are not drive cycles in any
physical or standards sense — just short synthetic speed traces
(accelerate/cruise/brake) sized for fast unit tests.

## CSV format supported by `environment/drive_cycle.py`

Required columns:
```
time_s        - elapsed time, seconds, strictly increasing, starts at 0
speed_mps     - vehicle speed, m/s, must be >= 0
```
Optional columns:
```
acceleration_mps2  - if absent, derived as a_t = (v_t - v_{t-1}) / dt
                      (first sample defaults to 0.0, no prior sample to
                      derive from)
road_grade_deg     - degrees, positive = uphill; if absent, defaults to
                      0.0 (converted to radians internally to match
                      environment/vehicle_dynamics.py's convention)
```

## Sample time

Not fixed by the interface — `DriveCycle.dt_seconds` is inferred from the
first two rows' `time_s` difference and validated as constant and
positive. This project's own simulation timestep elsewhere is 1.0s
(`configs/battery.yaml` / `configs/simulation.yaml`); a real drive cycle
used with this project should use a matching or documented sample rate —
not enforced automatically, since `DriveCycle` does not resample or
interpolate (deliberately, see the module docstring — silent resampling
would change the physical meaning of recorded accelerations).

## Preprocessing

None performed by this module beyond validation (finite values, speed
sign, monotonic time) and the acceleration-derivation fallback described
above. Any smoothing, resampling, or grade-estimation done to a real
dataset before use must be documented here, per source, when that
dataset is actually added.

## License / usage notes

N/A until a real dataset is added — to be filled in with the specific
dataset's license terms at that time.
