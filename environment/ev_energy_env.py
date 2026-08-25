"""
Integrated driving energy-management environment (Phase 6). Does NOT
replace environment/battery_env.py -- a separate, standalone Gymnasium
environment for the driving EMS problem.

Per-step pipeline (task §16):
    drive cycle -> vehicle forces -> wheel power -> drivetrain power
        -> PPO action -> desired battery power -> bidirectional safety
        layer -> feasible current -> ECM step -> reward -> observation

Known, documented simplification (task §20/§21): this initial version
does NOT feed power_deficit back into vehicle speed -- the drive cycle's
prescribed speed/acceleration is treated as achieved regardless of
whether the battery could actually supply/absorb the implied power.
power_deficit is computed and reported (in info and as a reward
component) but does not alter vehicle_dynamics' next-step inputs. A
closed-loop driver/vehicle response to a power shortfall is out of scope
for this phase, per the task's explicit "do not build a sophisticated
driver-controller model yet" instruction.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.ecm_model import BatteryECM, BatteryState
from environment.vehicle_dynamics import VehicleDynamics
from environment.drive_cycle import DriveCycle
from environment.drivetrain_model import DrivetrainModel
from safety.safety_layer import safety_layer_bidirectional, state_based_discharge_multiplier


# Observation order, fixed and documented (task §17's "document the exact
# observation order"). Every entry here is a single float, normalized as
# noted, in exactly this sequence:
OBSERVATION_FIELDS = [
    "soc",                        # [0,1], already normalized
    "voltage_norm",                # (V - v_min) / (v_max - v_min), clipped [0,1]
    "temperature_norm",            # T / t_max_c, clipped [0,1]
    "prev_battery_power_norm",     # signed, prev applied battery power / max(discharge,charge) power
    "speed_norm",                  # v / assumed_max_speed_mps (30 m/s ~108 km/h, documented below)
    "acceleration_norm",           # a / assumed_max_accel_mps2 (3.0 m/s^2, documented below), clipped [-1,1]
    "grade_norm",                  # grade_rad / (pi/6) (30 degrees), clipped [-1,1]
    "wheel_power_norm",            # P_wheel / max_desired_discharge_power_w, clipped [-1,1]
    "available_regen_norm",        # available_regen_w / max_desired_charge_power_w, clipped [0,1]
    "ambient_temp_norm",           # ambient_c / t_max_c, clipped [0,1]
    "trip_progress",               # current_time / total_cycle_time, [0,1]
]

# Reasonable normalization references, NOT vehicle-specific data --
# documented explicitly as such, not presented as sourced parameters.
ASSUMED_MAX_SPEED_MPS = 30.0     # ~108 km/h, a generic urban/highway-mix ceiling for normalization only
ASSUMED_MAX_ACCEL_MPS2 = 3.0     # generic passenger-car ceiling for normalization only


class EVEnergyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        vehicle_config: Dict,
        drivetrain_config: Dict,
        battery_config: Dict,
        safety_config: Dict,
        energy_config: Dict,
        drive_cycle_path: str,
        mode: Optional[str] = None,
    ):
        super().__init__()
        self.vehicle_config = vehicle_config
        self.drivetrain_config = drivetrain_config
        self.battery_config = battery_config
        self.safety_config = safety_config
        self.energy_config = energy_config
        self.mode = mode or "train"

        self.vehicle = VehicleDynamics(vehicle_config)
        self.drivetrain = DrivetrainModel(drivetrain_config)
        self.ecm = BatteryECM(battery_config)
        self.drive_cycle_path = drive_cycle_path

        self.dt = float(energy_config["dt_seconds"])
        self.t_max_c = float(battery_config["t_max_c"])
        self.v_min = float(battery_config["v_min"])
        self.v_max = float(battery_config["v_max"])
        self.max_charge_power_w = float(energy_config["max_desired_charge_power_w"])
        self.max_discharge_power_w = float(energy_config["max_desired_discharge_power_w"])
        self.episode_max_steps = int(energy_config["episode_max_steps"])

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(OBSERVATION_FIELDS),), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._state: Optional[BatteryState] = None
        self._ambient_temp_c: float = 25.0
        self._prev_battery_power_w: float = 0.0
        self._step_count: int = 0
        self._drive_cycle: Optional[DriveCycle] = None

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        rng = self.np_random

        options = options or {}
        soc_lo, soc_hi = self.energy_config["initial_soc_range"]
        amb_lo, amb_hi = self.energy_config["ambient_temp_range_c"]
        initial_soc = float(options.get("initial_soc", rng.uniform(soc_lo, soc_hi)))
        self._ambient_temp_c = float(options.get("ambient_temp_c", rng.uniform(amb_lo, amb_hi)))

        self._state = BatteryState(soc=initial_soc, v_rc=0.0, temperature_c=self._ambient_temp_c)
        self._prev_battery_power_w = 0.0
        self._step_count = 0

        self._drive_cycle = DriveCycle(self.drive_cycle_path)
        self._drive_cycle.reset()

        obs = self._get_observation()
        info = {}
        return obs, info

    # ------------------------------------------------------------------ #
    def step(self, action):
        assert self._state is not None, "call reset() before step()"
        action_val = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))

        speed = self._drive_cycle.current_speed()
        accel = self._drive_cycle.current_acceleration()
        grade = self._drive_cycle.current_grade()
        forces = self.vehicle.compute(speed_mps=speed, acceleration_mps2=accel, road_grade_rad=grade)
        drivetrain_out = self.drivetrain.compute(p_wheel_w=forces.p_wheel)

        # action -> desired battery power (signed W). action_val > 0 ->
        # desired CHARGE power; action_val < 0 -> desired DISCHARGE power
        # (see configs/energy_management.yaml's documented convention).
        if action_val >= 0.0:
            desired_battery_power_w = action_val * self.max_charge_power_w
        else:
            desired_battery_power_w = action_val * self.max_discharge_power_w  # already negative

        power_deficit_w = 0.0
        friction_braking_w = 0.0  # regen not used by PPO -> assumed absorbed by friction brakes (loss)

        if forces.p_wheel >= 0.0:
            # Propulsion needed. Only a discharge-direction (negative)
            # action is physically meaningful here; a charge-direction
            # action has nothing to act on during propulsion (see module
            # docstring's documented simplification) and is treated as
            # "PPO chose not to supply power", i.e. full deficit.
            required_discharge_w = drivetrain_out.battery_power_w  # positive magnitude required
            requested_discharge_w = max(0.0, -desired_battery_power_w)  # magnitude PPO is offering
            supplied_discharge_w = min(requested_discharge_w, required_discharge_w)
            power_deficit_w = required_discharge_w - supplied_discharge_w
            feasible_desired_power_w = -supplied_discharge_w  # signed, discharge = negative
        else:
            # Braking / regen opportunity. Only a charge-direction
            # (positive) action is meaningful; PPO decides how much of
            # the available regen to actually use -- the rest is
            # explicitly accounted as friction-braking loss (task §12,
            # "Any unavailable regenerative energy must be explicitly
            # accounted for as braking loss").
            available_w = drivetrain_out.available_regenerative_power_w
            requested_charge_w = max(0.0, desired_battery_power_w)
            used_regen_w = min(requested_charge_w, available_w)
            friction_braking_w = available_w - used_regen_w
            feasible_desired_power_w = used_regen_w  # signed, charge = positive

        # Power -> current: I = P / V (task §14), using the pre-step
        # terminal voltage as the conversion estimate (consistent with
        # how the rest of this project estimates voltage before stepping
        # the ECM -- see battery_env.py's own estimated_voltage usage).
        v_est = self.ecm.terminal_voltage(self._state, 0.0)
        requested_current_a = feasible_desired_power_w / v_est if v_est > 0.0 else 0.0

        applied_current_a, safety_info = safety_layer_bidirectional(
            requested_current_a, self._state, self.safety_config, estimated_voltage=v_est
        )

        prev_state = self._state
        new_state = self.ecm.step(prev_state, applied_current_a, ambient_temp_c=self._ambient_temp_c)
        applied_power_w = applied_current_a * v_est

        reward, components = self._compute_reward(
            prev_state=prev_state, new_state=new_state, applied_current_a=applied_current_a,
            applied_power_w=applied_power_w, power_deficit_w=power_deficit_w,
            friction_braking_w=friction_braking_w, safety_info=safety_info,
        )

        self._state = new_state
        self._prev_battery_power_w = applied_power_w
        self._step_count += 1
        self._drive_cycle.step()

        terminated = False  # no terminal safety condition defined yet for this phase (documented limitation)
        truncated = (self._step_count >= self.episode_max_steps) or self._drive_cycle.is_done()

        obs = self._get_observation()
        info = {
            "applied_current_a": applied_current_a,
            "applied_power_w": applied_power_w,
            "power_deficit_w": power_deficit_w,
            "friction_braking_w": friction_braking_w,
            "p_wheel_w": forces.p_wheel,
            "safety_intervention": safety_info.as_dict(),
            "reward_components": components,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _compute_reward(self, prev_state, new_state, applied_current_a, applied_power_w,
                         power_deficit_w, friction_braking_w, safety_info):
        cfg = self.energy_config

        tracking_error = abs(power_deficit_w) / max(1.0, self.max_discharge_power_w)
        energy_cost = abs(applied_power_w) / max(1.0, self.max_discharge_power_w)
        regen_recovery = max(0.0, applied_power_w) / max(1.0, self.max_charge_power_w)

        t_ref = cfg["thermal_reference_temp_c"]
        t_scale = cfg["thermal_scale_c"]
        q_ref = cfg["thermal_q_reference_w"]
        thermal_excess = max(0.0, new_state.temperature_c - t_ref)
        q_gen = self.ecm.heat_generation_w(prev_state, applied_current_a)
        thermal_stress = ((thermal_excess / t_scale) ** 2) * (q_gen / q_ref)

        safety_penalty = safety_info.magnitude

        total = (
            -cfg["w_tracking_error"] * tracking_error
            - cfg["w_energy_cost"] * energy_cost
            + cfg["w_regen_recovery"] * regen_recovery
            - cfg["w_thermal_stress"] * thermal_stress
            - cfg["w_safety_penalty"] * safety_penalty
        )
        components = {
            "tracking_error": tracking_error,
            "energy_cost": energy_cost,
            "regen_recovery": regen_recovery,
            "thermal_stress": thermal_stress,
            "safety_penalty": safety_penalty,
        }
        return total, components

    # ------------------------------------------------------------------ #
    def _get_observation(self) -> np.ndarray:
        state = self._state
        v_t = self.ecm.terminal_voltage(state, 0.0)
        speed = self._drive_cycle.current_speed()
        accel = self._drive_cycle.current_acceleration()
        grade = self._drive_cycle.current_grade()
        forces = self.vehicle.compute(speed_mps=speed, acceleration_mps2=accel, road_grade_rad=grade)
        drivetrain_out = self.drivetrain.compute(p_wheel_w=forces.p_wheel)

        total_time = self._drive_cycle.total_duration_s()
        trip_progress = self._drive_cycle.current_time() / total_time if total_time > 0 else 0.0

        vals = [
            float(np.clip(state.soc, 0.0, 1.0)),
            float(np.clip((v_t - self.v_min) / max(1e-6, self.v_max - self.v_min), 0.0, 1.0)),
            float(np.clip(state.temperature_c / max(1e-6, self.t_max_c), 0.0, 1.0)),
            float(np.clip(self._prev_battery_power_w / max(self.max_charge_power_w, self.max_discharge_power_w), -1.0, 1.0)),
            float(np.clip(speed / ASSUMED_MAX_SPEED_MPS, 0.0, 1.0)),
            float(np.clip(accel / ASSUMED_MAX_ACCEL_MPS2, -1.0, 1.0)),
            float(np.clip(grade / (math.pi / 6.0), -1.0, 1.0)),
            float(np.clip(forces.p_wheel / max(1.0, self.max_discharge_power_w), -1.0, 1.0)),
            float(np.clip(drivetrain_out.available_regenerative_power_w / max(1.0, self.max_charge_power_w), 0.0, 1.0)),
            float(np.clip(self._ambient_temp_c / max(1e-6, self.t_max_c), 0.0, 1.0)),
            float(np.clip(trip_progress, 0.0, 1.0)),
        ]
        return np.array(vals, dtype=np.float32)
