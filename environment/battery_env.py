"""
Gymnasium environment wrapping the validated ECM battery model,
shared safety layer, and configurable reward function.

Pipeline per step:
    Action -> Controller Current -> Safety Layer -> Battery Model
        -> Observation -> Reward -> Termination
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.ecm_model import BatteryECM, BatteryState
from safety.safety_layer import safety_layer, state_based_current_multiplier


class BatteryChargingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        battery_config: Dict,
        safety_config: Dict,
        reward_config: Dict,
        simulation_config: Dict,
        mode: Optional[str] = None,
        enforce_safety: bool = True,
    ):
        super().__init__()
        self.battery_config = battery_config
        self.safety_config = safety_config
        self.reward_config = reward_config
        self.simulation_config = simulation_config
        # Ablation switch: when False, the safety layer's clamped current is
        # NOT applied to the battery — only the physical current bounds
        # [0, i_max] are enforced. Safety-layer intervention info is still
        # computed every step (for monitoring/comparison), and the episode
        # still hard-terminates on overvoltage/overtemperature via
        # _check_termination, so this is "monitoring mode", not an
        # unconstrained/unsafe simulation.
        self.enforce_safety = enforce_safety

        self.ecm = BatteryECM(battery_config)
        self.dt = float(simulation_config.get("dt_seconds", self.ecm.dt))
        self.max_episode_steps = int(simulation_config["max_episode_steps"])
        self.target_soc = float(simulation_config["target_soc"])
        self.mode = mode or simulation_config.get("mode", "train")

        self.i_max = float(battery_config["i_max_a"])
        self.v_max = float(battery_config["v_max"])
        self.t_max = float(battery_config["t_max_c"])

        # Observation: [SoC, terminal_voltage, temperature, prev_current, ambient_temp,
        #               state_based_safe_fraction] — normalised. The 6th dimension is v2:
        # exposes the safety layer's STATE-BASED (SoC + temperature only,
        # NOT voltage) ceiling multiplier directly, so the policy has
        # explicit access to an approximation of "how much can I safely
        # request right now" instead of having to infer it from raw state.
        # Named honestly: this is NOT the full safety ceiling (which also
        # incorporates a voltage taper) — voltage is excluded because its
        # multiplier depends on the terminal voltage AT the requested
        # current, a circular dependency. In this system voltage tapering
        # heavily overlaps the SoC taper zone near full charge, so SoC alone
        # is a reasonable but not exact proxy. See
        # safety.safety_layer.state_based_current_multiplier docstring.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        # Action: normalised charging current in [0, 1] -> current = action * Imax
        # Symmetric, normalized action space, per SB3's own guidance: an
        # unbounded Gaussian policy sampling into an ASYMMETRIC Box([0,1])
        # tends to clip hard at the 0 boundary whenever the mean drifts
        # negative, with zero gradient through the clip — the policy can get
        # permanently stuck outputting exactly 0 current. [-1,1] symmetric
        # avoids that failure mode; remapped to [0, i_max] in step().
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Normalisation reference ranges
        self._v_min = float(battery_config["v_min"])
        self._v_range = max(1e-6, self.v_max - self._v_min)
        self._t_min_ref = 0.0
        self._t_range = max(1e-6, self.t_max - self._t_min_ref)

        self._state: Optional[BatteryState] = None
        self._ambient_temp_c: float = 25.0
        self._prev_current_a: float = 0.0
        self._is_first_step: bool = True
        self._step_count: int = 0
        self._eval_scenario_idx: int = 0

        self._np_random = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        initial_soc, ambient_temp = self._sample_initial_conditions(options)

        self._state = self.ecm.reset_state(initial_soc=initial_soc, ambient_temp_c=ambient_temp)
        self._ambient_temp_c = ambient_temp
        self._prev_current_a = 0.0
        self._is_first_step = True
        self._step_count = 0

        obs = self._get_observation()
        info = {"initial_soc": initial_soc, "ambient_temp_c": ambient_temp}
        return obs, info

    def step(self, action):
        # Capture both unclipped raw action and clipped action
        raw_action = float(np.asarray(action).flatten()[0])
        clipped_action = float(np.clip(raw_action, -1.0, 1.0))
        action_val = (clipped_action + 1.0) / 2.0
        requested_current = action_val * self.i_max

        # Evaluate the voltage estimate at i_max (the worst case), NOT at
        # requested_current. Using the actual request creates a circular
        # dependency: higher requests -> higher estimated voltage -> lower
        # voltage-taper multiplier -> lower ceiling, and since the ceiling
        # would then be a DEcreasing function of the request once it binds,
        # applied = min(requested, ceiling(requested)) can genuinely violate
        # monotonicity (verified directly: constructing a state with an
        # artificially high Vrc reproduces real, if small, non-monotonic
        # steps). This system's actual reachable states never trigger it in
        # practice (voltage tapering is always shadowed by SoC tapering
        # before it can bind alone — max reachable Vrc is ~2V, far short of
        # what's needed), but that's an accident of today's parameter
        # values, not a guarantee. Evaluating at i_max makes the estimate —
        # and therefore the ceiling — purely a function of state, matching
        # the same design already used in state_based_current_multiplier.
        estimated_voltage = self.ecm.terminal_voltage(self._state, self.i_max)
        clamped_current, safety_info = safety_layer(
            requested_current, self._state, self.safety_config, estimated_voltage=estimated_voltage
        )

        if self.enforce_safety:
            applied_current = clamped_current
        else:
            # Ablation / monitoring mode: bypass the safety layer's clamping,
            # but still respect hard physical bounds (can't charge negative,
            # can't exceed the absolute current ceiling) and still record
            # what the safety layer *would* have done via safety_info.
            applied_current = max(0.0, min(requested_current, self.i_max))

        new_state = self.ecm.step(self._state, applied_current, self._ambient_temp_c)
        terminal_voltage = self.ecm.terminal_voltage(new_state, applied_current)

        reward, reward_components = self._compute_reward(
            prev_state=self._state,
            requested_current=requested_current,
            new_state=new_state,
            applied_current=applied_current,
            safety_info=safety_info,
            terminal_voltage=terminal_voltage,
        )

        self._prev_current_a = applied_current
        self._is_first_step = False
        self._state = new_state
        self._step_count += 1

        terminated, term_reason = self._check_termination(terminal_voltage)
        truncated = self._step_count >= self.max_episode_steps

        # v3 fix: terminal bonuses/penalties were applied to the scalar
        # reward but never logged into reward_components, so Stage 2's
        # reward_components.csv couldn't reconstruct the true total reward
        # at terminal steps — always zero-summed short of the real total on
        # the step an episode actually ended.
        reward_components["target_reached_bonus"] = 0.0
        reward_components["terminal_shortfall_penalty"] = 0.0
        reward_components["overvoltage_penalty"] = 0.0
        reward_components["overtemperature_penalty"] = 0.0

        if terminated and term_reason == "target_soc_reached":
            bonus = self.reward_config["target_reached_bonus"]
            reward += bonus
            reward_components["target_reached_bonus"] = bonus
        elif terminated and term_reason == "overvoltage":
            penalty = self.reward_config["overvoltage_penalty"]
            reward -= penalty
            reward_components["overvoltage_penalty"] = penalty
        elif terminated and term_reason == "overtemperature":
            penalty = self.reward_config["overtemperature_penalty"]
            reward -= penalty
            reward_components["overtemperature_penalty"] = penalty
        elif truncated and not terminated:
            # v3: episode ran out of time without reaching target_soc — e.g.
            # a policy that charges partway then stalls (Run 008's 700k
            # checkpoint: ~0.9246 final SoC, sitting idle until truncation).
            # Explicitly penalize the shortfall so "close but truncated" is
            # never reward-neutral relative to "reached target".
            shortfall = max(0.0, self.target_soc - self._state.soc)
            penalty = self.reward_config["terminal_shortfall_penalty_weight"] * shortfall
            reward -= penalty
            reward_components["terminal_shortfall_penalty"] = penalty

        obs = self._get_observation()
        target_reached = bool(terminated and term_reason == "target_soc_reached")
        q_gen = self.ecm.heat_generation_w(self._state, applied_current)
        info = {
            "raw_action": raw_action,
            "clipped_action": clipped_action,
            "requested_current": requested_current,
            "safe_current_ceiling": safety_info.safe_current_ceiling,
            "applied_current": applied_current,
            "applied_current_a": applied_current,
            "safety_intervention": safety_info.as_dict(),
            "q_gen": q_gen,
            "temperature": self._state.temperature_c,
            "ambient_temp_c": self._ambient_temp_c,
            "terminal_voltage": terminal_voltage,
            "termination_reason": term_reason if (terminated or truncated) else None,
            "reward_components": reward_components,
            "target_reached": target_reached,
            "final_soc_if_ended": self._state.soc if (terminated or truncated) else None,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _sample_initial_conditions(self, options: Optional[Dict]) -> Tuple[float, float]:
        if options and "initial_soc" in options and "ambient_temp_c" in options:
            return float(options["initial_soc"]), float(options["ambient_temp_c"])

        if self.mode == "eval":
            eval_cfg = self.simulation_config["eval"]
            soc_grid = eval_cfg["initial_soc_grid"]
            temp_grid = eval_cfg["ambient_temp_grid_c"]
            n = len(soc_grid) * len(temp_grid)
            idx = self._eval_scenario_idx % n
            self._eval_scenario_idx += 1
            soc_idx = idx // len(temp_grid)
            temp_idx = idx % len(temp_grid)
            return float(soc_grid[soc_idx]), float(temp_grid[temp_idx])

        train_cfg = self.simulation_config["train"]
        soc_lo, soc_hi = train_cfg["initial_soc_range"]
        if "mixed_ambient_sampler" in train_cfg:
            mix = train_cfg["mixed_ambient_sampler"]
            p_stress = float(mix.get("p_stress", 0.25))
            if float(self._np_random.uniform(0.0, 1.0)) < p_stress:
                temp_lo, temp_hi = mix["stress_range_c"]
            else:
                temp_lo, temp_hi = mix["normal_range_c"]
        else:
            temp_lo, temp_hi = train_cfg["ambient_temp_range_c"]
        soc = float(self._np_random.uniform(soc_lo, soc_hi))
        temp = float(self._np_random.uniform(temp_lo, temp_hi))
        return soc, temp

    def _get_observation(self) -> np.ndarray:
        s = self._state
        v = self.ecm.terminal_voltage(s, self._prev_current_a)

        soc_n = float(np.clip(s.soc, 0.0, 1.0))
        v_n = float(np.clip((v - self._v_min) / self._v_range, 0.0, 1.0))
        t_n = float(np.clip((s.temperature_c - self._t_min_ref) / self._t_range, 0.0, 1.0))
        i_n = float(np.clip(self._prev_current_a / self.i_max, 0.0, 1.0))
        t_amb_n = float(np.clip((self._ambient_temp_c - self._t_min_ref) / self._t_range, 0.0, 1.0))
        state_based_safe_frac = float(np.clip(state_based_current_multiplier(s, self.safety_config), 0.0, 1.0))

        return np.array([soc_n, v_n, t_n, i_n, t_amb_n, state_based_safe_frac], dtype=np.float32)

    def _compute_reward(self, prev_state, requested_current, new_state, applied_current, safety_info, terminal_voltage):
        w = self.reward_config["weights"]

        delta_soc = new_state.soc - prev_state.soc
        progress = w["charging_progress"] * delta_soc

        temp_start = self.reward_config["temperature_penalty_start_c"]
        temp_excess = max(0.0, new_state.temperature_c - temp_start)
        temp_penalty = w["temperature_penalty"] * temp_excess

        # Stable V3: state-aware thermal reward (default OFF via
        # reward.yaml thermal_enabled — baseline reward is unchanged when
        # disabled). Zero/negligible while temperature is comfortably below
        # thermal_reference_temp_c, increasing quadratically toward the
        # existing hard thermal cutoff — see reward.yaml for the reference
        # derivation. Uses the ECM's own heat_generation_w (no duplicate
        # formula) evaluated at this step's prev_state.v_rc / applied_current,
        # the same instantaneous basis _derivatives uses internally.
        thermal_reward = 0.0
        if self.reward_config.get("thermal_enabled", False):
            t_ref = self.reward_config["thermal_reference_temp_c"]
            t_scale = self.reward_config["thermal_scale_c"]
            q_ref = self.reward_config["thermal_q_reference_w"]
            thermal_weight = self.reward_config["thermal_weight"]
            thermal_excess = max(0.0, new_state.temperature_c - t_ref)
            normalized_excess = thermal_excess / t_scale
            q_gen = self.ecm.heat_generation_w(prev_state, applied_current)
            normalized_q_gen = q_gen / q_ref
            thermal_reward = thermal_weight * (normalized_excess ** 2) * normalized_q_gen

        safety_penalty = w["safety_penalty"] * safety_info.magnitude

        # v2: absolute over-request penalty. safety_penalty above uses a
        # FRACTIONAL magnitude (1 - applied/requested), which is a weak
        # signal exactly when it matters most — e.g. requesting 160A when
        # only 80A is allowed gives magnitude=0.5, a moderate penalty for
        # wasting a full 80A of request. This term penalizes the wasted
        # current directly, in Amps (normalized by i_max), so a large
        # absolute waste always costs regardless of the ratio. This is the
        # fix for the v1 finding (docs/results_and_discussion.md Section 4)
        # that the trained policy saturated to always requesting i_max,
        # since ~91% of an episode the safety layer doesn't engage at all
        # and over-requesting was effectively free there.
        overrequest_a = max(0.0, requested_current - applied_current)
        overrequest_penalty = w["overrequest_penalty"] * (overrequest_a / self.i_max)

        if self._is_first_step:
            # No real "previous action" exists yet — comparing against the
            # reset default of 0 would penalize every episode's first action
            # purely as an artifact, not as a genuine smoothness violation.
            smoothness_penalty = 0.0
        else:
            current_delta = abs(applied_current - self._prev_current_a) / self.i_max
            smoothness_penalty = w["smoothness_penalty"] * current_delta

        # v3: constant per-step cost, applied unconditionally — see
        # reward.yaml derivation. Without this, the reward had no explicit
        # incentive for finishing faster; two episodes reaching the same
        # final SoC via different step counts were reward-equivalent.
        time_penalty = w["time_penalty"]

        total = progress - temp_penalty - safety_penalty - overrequest_penalty - smoothness_penalty - time_penalty - thermal_reward
        components = {
            "progress": progress,
            "temp_penalty": temp_penalty,
            "safety_penalty": safety_penalty,
            "overrequest_penalty": overrequest_penalty,
            "smoothness_penalty": smoothness_penalty,
            "time_penalty": time_penalty,
            "thermal_reward": thermal_reward,
        }
        return total, components

    def _check_termination(self, terminal_voltage: float) -> Tuple[bool, Optional[str]]:
        if self._state.soc >= self.target_soc:
            return True, "target_soc_reached"
        if terminal_voltage >= self.v_max:
            return True, "overvoltage"
        if self._state.temperature_c >= self.t_max:
            return True, "overtemperature"
        return False, None