"""
Drive-cycle interface: reads a time_s/speed_mps (+ optional
acceleration_mps2, road_grade_deg) CSV and exposes it as a simple
index-driven sequence for the (future) driving-EMS environment.

No interpolation is performed -- each step() call advances to the next
row in the file exactly as recorded. If a drive cycle's own sample
timestep differs from the project's dt_seconds (1.0s, configs/battery.yaml
/ configs/simulation.yaml), that is flagged as a mismatch (see
DriveCycle.dt_seconds and the validation below) rather than silently
resampled -- silent resampling would change the physical meaning of the
recorded accelerations without the caller knowing.

Per the master task spec: this module does NOT expose future drive-cycle
values to any caller -- current_speed()/current_acceleration()/
current_grade()/current_time() only ever return the CURRENT index's
values, never a lookahead. A preview/prediction feature, if ever added,
would be a new, explicit, separate method.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DriveCycleSample:
    time_s: float
    speed_mps: float
    acceleration_mps2: float
    road_grade_rad: float


class DriveCycleValidationError(ValueError):
    pass


class DriveCycle:
    """Index-driven drive-cycle reader. reset() to the start, step() to
    advance one sample, current_*() to read the sample at the current index."""

    REQUIRED_COLUMNS = ("time_s", "speed_mps")
    OPTIONAL_COLUMNS = ("acceleration_mps2", "road_grade_deg")

    def __init__(self, csv_path: str):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Drive cycle file not found: {csv_path}")
        self.csv_path = csv_path
        self._samples: List[DriveCycleSample] = self._load_and_validate(csv_path)
        self._idx = 0

    # ------------------------------------------------------------------ #
    # Loading and validation
    # ------------------------------------------------------------------ #
    def _load_and_validate(self, csv_path: str) -> List[DriveCycleSample]:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for col in self.REQUIRED_COLUMNS:
                if col not in fieldnames:
                    raise DriveCycleValidationError(
                        f"Drive cycle CSV missing required column '{col}': {csv_path}"
                    )
            has_accel_col = "acceleration_mps2" in fieldnames
            has_grade_col = "road_grade_deg" in fieldnames

            times: List[float] = []
            speeds: List[float] = []
            accels_raw: List[Optional[float]] = []
            grades_deg: List[float] = []

            for row_idx, row in enumerate(reader):
                try:
                    t = float(row["time_s"])
                    v = float(row["speed_mps"])
                except (TypeError, ValueError) as exc:
                    raise DriveCycleValidationError(
                        f"Row {row_idx}: non-numeric time_s/speed_mps in {csv_path}"
                    ) from exc

                if not math.isfinite(t) or not math.isfinite(v):
                    raise DriveCycleValidationError(
                        f"Row {row_idx}: non-finite time_s/speed_mps in {csv_path}"
                    )
                if v < 0.0:
                    raise DriveCycleValidationError(
                        f"Row {row_idx}: negative speed_mps ({v}) in {csv_path} -- "
                        f"speed_mps >= 0 is required (see module docstring)."
                    )

                a_raw = None
                if has_accel_col and row.get("acceleration_mps2", "") not in ("", None):
                    a_raw = float(row["acceleration_mps2"])
                    if not math.isfinite(a_raw):
                        raise DriveCycleValidationError(
                            f"Row {row_idx}: non-finite acceleration_mps2 in {csv_path}"
                        )

                g_deg = 0.0
                if has_grade_col and row.get("road_grade_deg", "") not in ("", None):
                    g_deg = float(row["road_grade_deg"])
                    if not math.isfinite(g_deg):
                        raise DriveCycleValidationError(
                            f"Row {row_idx}: non-finite road_grade_deg in {csv_path}"
                        )

                times.append(t)
                speeds.append(v)
                accels_raw.append(a_raw)
                grades_deg.append(g_deg)

        if len(times) < 2:
            raise DriveCycleValidationError(
                f"Drive cycle must have at least 2 rows to define a timestep: {csv_path}"
            )

        self.dt_seconds = times[1] - times[0]
        if self.dt_seconds <= 0.0:
            raise DriveCycleValidationError(
                f"Initial timestep must be positive (got {self.dt_seconds}) in {csv_path}"
            )

        # Time strictly increasing and uniform timestep spacing
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if dt <= 0.0:
                raise DriveCycleValidationError(
                    f"Row {i}: time_s must strictly increase (got dt={dt} at t={times[i]}) "
                    f"in {csv_path}"
                )
            if abs(dt - self.dt_seconds) > 1e-3:
                raise DriveCycleValidationError(
                    f"Row {i}: irregular timestep spacing detected (expected {self.dt_seconds:.4f}s, "
                    f"got {dt:.4f}s at t={times[i]}) in {csv_path}"
                )

        # Derive acceleration where absent: a_t = (v_t - v_{t-1}) / dt.
        # First sample's acceleration, if not provided, has no prior sample
        # to derive from -- defined as 0.0 (documented here, not silently
        # assumed elsewhere).
        accelerations: List[float] = []
        for i, a_raw in enumerate(accels_raw):
            if a_raw is not None:
                accelerations.append(a_raw)
            elif i == 0:
                accelerations.append(0.0)
            else:
                dt = times[i] - times[i - 1]
                accelerations.append((speeds[i] - speeds[i - 1]) / dt)

        samples = [
            DriveCycleSample(
                time_s=times[i],
                speed_mps=speeds[i],
                acceleration_mps2=accelerations[i],
                road_grade_rad=math.radians(grades_deg[i]),
            )
            for i in range(len(times))
        ]
        return samples

    # ------------------------------------------------------------------ #
    # Sequence interface
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._idx = 0

    def step(self) -> bool:
        """Advance to the next sample. Returns False if already at/past
        the last sample (caller should check is_done() and stop)."""
        if self.is_done():
            return False
        self._idx += 1
        return True

    def is_done(self) -> bool:
        return self._idx >= len(self._samples) - 1

    def current_speed(self) -> float:
        return self._samples[self._idx].speed_mps

    def current_acceleration(self) -> float:
        return self._samples[self._idx].acceleration_mps2

    def current_grade(self) -> float:
        """Road grade in radians (matches environment.vehicle_dynamics's
        expected unit -- see that module's docstring)."""
        return self._samples[self._idx].road_grade_rad

    def current_time(self) -> float:
        return self._samples[self._idx].time_s

    def total_duration_s(self) -> float:
        """Total elapsed time spanned by the loaded cycle (last sample's
        time_s). Used for trip-progress normalization by callers."""
        return self._samples[-1].time_s

    def __len__(self) -> int:
        return len(self._samples)
