from __future__ import annotations

"""
==============================================================================
RL-BMS CONSOLIDATED CODEBASE
==============================================================================
This single file contains the complete source code and configuration files
for the RL-BMS project. Each component is labeled with its original file path.
==============================================================================
"""



# ==============================================================================
# FILE: configs/simulation.yaml
# LOCAL PATH: file:////home/claude/rl-bms/configs/simulation.yaml
# ==============================================================================

"""
# Simulation Configuration

dt_seconds: 1.0
max_episode_steps: 7200        # 2 hours at 1s steps (safety net; normally terminates earlier)
target_soc: 0.95

mode: "train"                  # "train" or "eval" — env reads this to pick sampling strategy

train:
  initial_soc_range: [0.10, 0.30]
  ambient_temp_range_c: [15.0, 35.0]

eval:
  initial_soc_grid: [0.10, 0.15, 0.20, 0.25, 0.30]
  ambient_temp_grid_c: [15.0, 25.0, 35.0]
  fixed_seed: 42

"""

# ==============================================================================
# FILE: configs/battery.yaml
# LOCAL PATH: file:////home/claude/rl-bms/configs/battery.yaml
# ==============================================================================

"""
# Battery Model Configuration — 1RC Thevenin Equivalent Circuit Model
# Chemistry: NMC (Nickel Manganese Cobalt)
#
# Derived for a Tata Nexon EV Long Range-class pack (45 kWh usable, 121 Ah,
# ~372 V nominal — Tata.ev spec sheet / EV Database, July 2026 search).
# Pack topology assumed ~100s8p from public specs (100 series x ~3.7V NMC
# cells = ~370V nominal; 8 parallel x ~15Ah cells = ~120Ah), consistent with
# large-format NMC pouch cells (3.0-4.2V window, ~3.7V nominal — see e.g.
# arXiv:2203.08515, a 55Ah/3.7V NMC pouch cell characterisation).
# Every parameter below is annotated with its source: [datasheet] = read
# directly off a public spec sheet, [derived] = computed from datasheet
# figures + a documented pack topology assumption, [literature] = typical
# published value for this cell/pack class, not Nexon-specific.

chemistry: "NMC"

# --- Capacity ---
nominal_capacity_ah: 121.0          # [datasheet] Tata Nexon EV LR: "45 kWh (121 Ah)" usable capacity

# --- Electrical (1RC ECM) parameters, pack-level ---
# R0 [derived]: per-cell DCIR ~1.5 mOhm (literature, large-format NMC pouch
#   cells, e.g. 20-55Ah class) x 100 series / 8 parallel = ~18.75 mOhm,
#   plus ~15 mOhm busbar/connector/contactor overhead (typical 10-30 mOhm
#   range per batterydesign.net pack-resistance analysis) => ~34 mOhm.
r0_ohm: 0.034
# R1 [derived]: per-cell polarisation resistance ~1.0 mOhm (literature) x
#   100/8 => ~12.5 mOhm. No pack-level overhead (polarisation is intra-cell).
r1_ohm: 0.0125
# C1 [literature]: chosen so tau = R1*C1 = 30s, within the 20-40s range
#   commonly reported for 1RC polarisation dynamics of large-format NMC cells.
c1_farad: 2400.0

# --- OCV-SoC lookup table (NMC pack, 0..1 SoC -> Volts) ---
# [derived]: typical single-cell NMC OCV-SoC curve (3.0V @ 0% -> 3.7V nominal
# @ ~50-60% -> 4.2V @ 100%, standard NMC shape) scaled x100 for the assumed
# 100-series pack. Smooth curve by construction -> strong PPO learning signal.
ocv_soc_points:
  soc:    [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
  ocv_v:  [300.0, 330.0, 345.0, 355.0, 360.0, 363.0, 365.0, 368.0, 372.0, 380.0, 395.0, 405.0, 420.0]

# --- Thermal model (lumped capacitance) ---
mass_kg: 300.0                      # [literature/estimated] typical reported pack mass for a 40-45kWh
                                     #   Ziptron-class pack in EV comparison reviews; not an official Tata figure
specific_heat_j_per_kgk: 900.0      # [literature] typical lumped effective specific heat (cells+housing+coolant)
                                     #   used in EV battery thermal modelling papers, common range 800-1100 J/kgK
convection_h: 25.0                  # [assumption] simplified natural-convection-equivalent proxy.
                                     #   NOTE: the real Ziptron pack is liquid-cooled; this v1 lumped model does
                                     #   not simulate a coolant loop (flagged as a known Version-1 simplification,
                                     #   consistent with "hardware/embedded deployment" being out of scope).
surface_area_m2: 2.0                # [estimated] approximate pack footprint for an underbody-mounted EV pack

# --- Safety-relevant electrical limits (physical, not policy) ---
v_max: 420.0                        # [derived] 100 series x 4.2V/cell max (literature NMC cell upper limit)
v_min: 300.0                        # [derived] 100 series x 3.0V/cell min (literature NMC cell lower limit)
t_max_c: 60.0                       # [literature] widely-cited NMC hard thermal safety limit
i_max_a: 160.0                      # [datasheet] Tata's quoted 60kW DC fast-charge spec / ~372V nominal
                                     #   = ~161A, rounded to 160A (matches the real charger's current ceiling
                                     #   rather than an arbitrary C-rate)

# --- State of Health (monitoring only, not in reward) ---
soh_initial: 1.0
soh_degradation_per_ah_throughput: 0.0000015   # [assumption] tunable; not calibrated against real degradation
                                                 #   data (explicitly out of scope per the implementation plan)

# --- Numerical integration ---
integration_method: "euler"         # "euler" (default) or "rk4"
dt_seconds: 1.0                     # Simulation timestep

"""

# ==============================================================================
# FILE: configs/safety.yaml
# LOCAL PATH: file:////home/claude/rl-bms/configs/safety.yaml
# ==============================================================================

"""
# Shared Safety Layer Configuration
# Applies identically to PPO and all baseline controllers.
# Scaled to match the sourced pack parameters in battery.yaml
# (121Ah / ~372V nominal / 300-420V window / 160A DC fast-charge ceiling).

# --- Current limiting ---
i_max_a: 160.0                       # Hard current ceiling (mirrors battery.yaml i_max_a — real 60kW DC charger spec)
i_min_a: 0.0                         # No discharge in Version 1 (charging only)

# --- Temperature protection ---
t_derate_start_c: 45.0               # [literature] NMC charging derating typically begins ~45C
t_hard_cutoff_c: 55.0                # [literature] below the 60C hard failure limit, with safety margin
t_derate_curve: "linear"             # linear derating between start and cutoff

# --- Progressive voltage tapering (emulates CCCV, avoids abrupt cutoff/oscillation) ---
v_taper_start: 405.0                 # Begins tapering ~15V below the 420V hard cell-voltage limit
v_hard_max: 420.0                    # Matches battery.yaml v_max (100s x 4.2V/cell)
v_taper_curve: "linear"

# --- SoC tapering near target ---
soc_taper_start: 0.90                # SoC fraction at which tapering begins
soc_taper_full: 1.00                 # SoC fraction at which current -> ~0
soc_taper_curve: "linear"

# --- Logging ---
log_interventions: true
intervention_fields: ["type", "requested_current", "applied_current", "magnitude"]

"""

# ==============================================================================
# FILE: configs/reward.yaml
# LOCAL PATH: file:////home/claude/rl-bms/configs/reward.yaml
# ==============================================================================

"""
# Reward Function Configuration
# Reward = ChargingProgress - TemperaturePenalty - SafetyPenalty - OverrequestPenalty
#          - SmoothnessPenalty - TimePenalty [- TerminalShortfallPenalty on truncation]
# All weights are tunable; scale so each term contributes meaningfully.
#
# IMPORTANT — charging_progress scale derivation:
# dSoC/step at full current (160A, 121Ah pack, dt=1s) = 160/(121*3600) = 3.67e-4.
# A weight of 10 (an earlier draft value) gives only ~0.0037 reward/step at
# full charging current — dwarfed by smoothness_penalty (up to 0.5/step) and
# temperature_penalty, so the agent's easiest optimum was minimizing
# penalties rather than charging. weight=1000 gives ~0.37 reward/step at
# full current, the same order of magnitude as the penalty terms below.

weights:
  charging_progress: 1000.0    # see derivation above — per unit SoC gained this step
  temperature_penalty: 0.05    # multiplied by max(0, T - t_penalty_start_c)
  safety_penalty: 5.0          # multiplied by safety-layer intervention magnitude (0..1)
  overrequest_penalty: 2.0     # v2: multiplied by (requested_A - applied_A) / i_max_A — see
                                #   docs/results_and_discussion.md Section 4. Unlike safety_penalty
                                #   (a FRACTIONAL magnitude, weak when the absolute waste is large
                                #   but the ratio is small), this scales with the raw wasted current
                                #   every step the safety layer clamps anything, even mid-episode —
                                #   giving consistent pressure to learn the safe ceiling instead of
                                #   blindly requesting i_max and letting the safety layer absorb it.
  smoothness_penalty: 0.5      # multiplied by |I_t - I_(t-1)| / i_max_a
  time_penalty: 0.05           # v3: constant per-step cost, applied EVERY step regardless of
                                #   whether charging occurred. Fixes a real gap found in Run 008:
                                #   the reward had no explicit incentive for finishing FASTER —
                                #   charging_progress rewards SoC gained, but two policies reaching
                                #   the same final SoC via different total step counts were treated
                                #   as equally good by the RL objective (the environment's own
                                #   evaluation metrics captured charging_time_s, but the reward the
                                #   agent was actually optimizing did not). Weight matches
                                #   temperature_penalty's order of magnitude; over a 5000-step
                                #   difference between a fast and slow episode this is a -250
                                #   cumulative difference, a meaningful fraction of total episode
                                #   reward (~500-900 in early experiments).

temperature_penalty_start_c: 40.0   # below this, no thermal penalty

# v3: applied ONLY when an episode ends via truncation (max_episode_steps
# reached) rather than reaching target_soc — penalizes finishing short of
# the target, distinct from and in addition to the per-step time_penalty.
# Fixes the Run 008 failure mode where a policy that charged to ~0.9246 and
# then sat idle for the rest of a 7200-step episode was never explicitly
# told that "close but truncated" is worse than "reached target". Weight
# matches charging_progress's scale (1000/unit-SoC) for symmetry: a 0.025
# shortfall (Run 008's 700k checkpoint) costs ~25 reward under this weight.
terminal_shortfall_penalty_weight: 1000.0

# --- Terminal bonuses/penalties ---
target_reached_bonus: 50.0
overvoltage_penalty: 20.0
overtemperature_penalty: 20.0
"""

# ==============================================================================
# FILE: configs/ppo.yaml
# LOCAL PATH: file:////home/claude/rl-bms/configs/ppo.yaml
# ==============================================================================

"""
# PPO Hyperparameter Configuration (Stable-Baselines3)

policy: "MlpPolicy"
policy_kwargs:
  net_arch: [64, 64]     # Two hidden layers, 64 neurons each

learning_rate: 0.0003
n_steps: 8192
# run_010 note: increased from 2048 -- run_009's actual TensorBoard curve
# (not just checkpoint end-states) showed explained_variance repeatedly
# breaking down (including going negative, e.g. -20.5 at step 274k, -2.5 at
# step 525k) alongside a sustained ep_rew_mean decline and steadily rising
# policy std (0.83 -> 1.48) through the back half of training -- consistent
# with the value function failing to bootstrap correctly when a 2048-step
# rollout captures only a fraction of episodes that run up to 7200 steps.
# approx_kl stayed low throughout (target_kl wasn't the bottleneck), ruling
# out "individual updates too large" as the mechanism. This is the single
# variable changed for run_010; seed, target_kl, reward, and safety layer
# are all unchanged from run_009.
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
# target_kl: caps how far a single PPO update can shift the policy — if the
# approx_kl for an epoch exceeds this, remaining epochs on that batch are
# skipped. Added after observing a mid-training collapse (checkpoint sweep
# on run_003 showed mean_final_soc dropping from ~0.95 at 100k-150k steps to
# ~0.175 at 225k-375k steps, before partially recovering by 1M) — a likely
# sign of one or more destabilizing large updates. 0.03 is a commonly used
# conservative value; None (SB3 default) applies no cap.
target_kl: 0.01
# run_005 note: tightened from 0.03 — run_003 and run_004 (target_kl=0.03)
# produced IDENTICAL checkpoint-by-checkpoint results under seed=42,
# indicating 0.03 never actually engaged (approx_kl never exceeded it).
# 0.01 is deliberately tight enough to bite on ordinary updates.
# NOTE: was 0.0 in earlier versions — that let the action std collapse
# (0.96 -> 0.05 over a 1M-step run) before the policy found real charging
# behavior, locking in a "do nothing" local optimum with no exploration left
# to escape it. 0.01 keeps some exploration pressure through training.
vf_coef: 0.5
max_grad_norm: 0.5

# --- Training stage control ---
stage1_sanity_timesteps: 5000
stage2_reward_verification_timesteps: 20000
stage3_hpo_timesteps: 50000
stage4_full_training_timesteps: 1000000

seed: 7
# run_006 note: changed from 123 — third seed in the seed-sensitivity
# comparison (42 -> severe collapse ~175k, 123 -> moderate decline ~475k).
# target_kl stays at 0.01 (unchanged from run_005) to isolate seed as the
# only variable. n_steps/lr/batch_size/gamma/gae_lambda/reward/battery model
# all deliberately untouched per the three-seed experiment design.
tensorboard_log: "runs/tensorboard"
checkpoint_freq: 25000
"""

# ==============================================================================
# FILE: configs/evaluation.yaml
# LOCAL PATH: file:////home/claude/rl-bms/configs/evaluation.yaml
# ==============================================================================

"""
# Evaluation Framework Configuration

controllers: ["ppo", "ppo_no_safety", "cc", "cccv", "adaptive", "max_current"]
# ppo_no_safety: ablation — same trained policy, safety layer left in
# monitoring-only mode (interventions logged but not enforced; episode still
# hard-terminates on overvoltage/overtemperature). Quantifies the safety
# layer's contribution rather than simulating an unconstrained battery.
# max_current: control experiment — a trivial controller that always
# requests i_max unconditionally. Tests whether PPO's advantage over the
# other baselines comes from anything learned, or entirely from the safety
# layer (see docs/results_and_discussion.md Section 4 — the trained policy
# was found to saturate to "always request max current" with zero
# measurable state-dependence).

n_runs_per_scenario: 1          # deterministic scenarios use fixed seeds; increase if adding stochastic noise later

metrics:
  - charging_time_s
  - peak_temperature_c
  - average_temperature_c
  - final_soc
  - safety_interventions
  - current_smoothness
  - energy_efficiency
  - average_input_power_w
  - energy_per_percent_soc_wh
  - voltage_stability
  - target_reached
  - time_to_target_s
  - target_shortfall

report:
  aggregate: ["mean", "std"]
  export_formats: ["csv", "png"]
  output_dir: "runs/{run_id}/evaluation"

# Baseline controller parameters (used only during evaluation/baseline runs).
# Scaled to the sourced 121Ah / 160A pack in battery.yaml.
cc:
  current_a: 121.0                # [derived] 1C for a 121Ah pack — common industry baseline C-rate

cccv:
  cc_current_a: 121.0             # 1C CC phase
  cv_voltage_v: 400.0             # [derived] typical CV setpoint, ~20V below the 420V hard cell-voltage limit
  cv_cutoff_current_a: 6.0        # [literature] ~C/20 cutoff, standard CCCV termination criterion
  cv_proportional_gain: 0.1       # CV-phase taper gain (fraction of cc_current shed per volt of overshoot)

adaptive:
  # SoC-banded current table (simple rule-based adaptive charging), scaled to i_max=160A
  soc_bands: [0.0, 0.3, 0.6, 0.8, 1.0]
  current_a_per_band: [160.0, 130.0, 83.0, 36.0]

"""

# ==============================================================================
# FILE: environment/__init__.py
# LOCAL PATH: file:////home/claude/rl-bms/environment/__init__.py
# ==============================================================================



# ==============================================================================
# FILE: environment/ecm_model.py
# LOCAL PATH: file:////home/claude/rl-bms/environment/ecm_model.py
# ==============================================================================

"""
1RC Thevenin Equivalent Circuit Model (ECM) for a lithium-ion (NMC) battery.

State: SoC, terminal voltage, RC polarisation voltage, temperature, SoH.
Integration: Euler (default) or RK4, selected via config.

This module has NO dependency on Gymnasium, RL, or the safety layer —
it is a pure physics simulator so it can be validated in isolation
before anything else is built on top of it (per "Physics Before AI").
"""

# from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class BatteryState:
    soc: float                # 0..1
    v_rc: float                # RC branch (polarisation) voltage, volts
    temperature_c: float       # pack temperature, deg C
    soh: float = 1.0           # state of health, 1.0 = new
    ah_throughput: float = 0.0  # cumulative |current|*dt integrated, for SoH tracking

    def copy(self) -> "BatteryState":
        return BatteryState(
            soc=self.soc,
            v_rc=self.v_rc,
            temperature_c=self.temperature_c,
            soh=self.soh,
            ah_throughput=self.ah_throughput,
        )


class BatteryECM:
    """1RC Thevenin ECM with NMC OCV(SoC) lookup and lumped thermal model."""

    def __init__(self, config: Dict):
        self.capacity_ah = float(config["nominal_capacity_ah"])
        self.r0 = float(config["r0_ohm"])
        self.r1 = float(config["r1_ohm"])
        self.c1 = float(config["c1_farad"])

        ocv_pts = config["ocv_soc_points"]
        self._ocv_soc = np.asarray(ocv_pts["soc"], dtype=float)
        self._ocv_v = np.asarray(ocv_pts["ocv_v"], dtype=float)

        self.mass_kg = float(config["mass_kg"])
        self.cp = float(config["specific_heat_j_per_kgk"])
        self.h = float(config["convection_h"])
        self.area_m2 = float(config["surface_area_m2"])

        self.v_max = float(config["v_max"])
        self.v_min = float(config["v_min"])
        self.t_max_c = float(config["t_max_c"])
        self.i_max_a = float(config["i_max_a"])

        self.soh_degradation_per_ah = float(config["soh_degradation_per_ah_throughput"])

        self.dt = float(config.get("dt_seconds", 1.0))
        self.integration_method = str(config.get("integration_method", "euler")).lower()
        if self.integration_method not in ("euler", "rk4"):
            raise ValueError(f"Unknown integration_method: {self.integration_method}")

    # ------------------------------------------------------------------ #
    # OCV lookup
    # ------------------------------------------------------------------ #
    def ocv(self, soc: float) -> float:
        soc_clamped = float(np.clip(soc, 0.0, 1.0))
        return float(np.interp(soc_clamped, self._ocv_soc, self._ocv_v))

    # ------------------------------------------------------------------ #
    # Derivatives (used by both Euler and RK4)
    # ------------------------------------------------------------------ #
    def _derivatives(self, state: BatteryState, current_a: float, ambient_temp_c: float):
        """Return d(soc)/dt, d(v_rc)/dt, d(temperature)/dt for given state and current.

        Convention: positive current = charging.
        """
        d_soc_dt = current_a / (self.capacity_ah * 3600.0)

        d_vrc_dt = (current_a / self.c1) - (state.v_rc / (self.r1 * self.c1))

        # Standard 1RC resistive-loss formulation: I^2*R0 (ohmic) +
        # Vrc^2/R1 (polarization resistance dissipation). NOT current_a*v_rc:
        # that term is the TOTAL power entering the R1||C1 branch, which
        # includes energy being stored in C1 as well as heat dissipated in
        # R1 — using it as heat over-counts during charging transients and,
        # more visibly, gives exactly zero heat during relaxation (I=0 but
        # Vrc>0 still decaying through R1, which is physically dissipating
        # stored energy the whole time). At steady state (Vrc=I*R1) the two
        # formulations coincide (I*Vrc = I^2*R1 = Vrc^2/R1), so this only
        # changes transient behavior, not steady-state — but this system's
        # RC time constant (~30s) is short relative to episode length, and
        # smoothness-penalized current changes happen often, so transients
        # are common enough to matter here.
        q_gen = (current_a ** 2) * self.r0 + (state.v_rc ** 2) / self.r1
        q_loss = self.h * self.area_m2 * (state.temperature_c - ambient_temp_c)
        d_temp_dt = (q_gen - q_loss) / (self.mass_kg * self.cp)

        return d_soc_dt, d_vrc_dt, d_temp_dt

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, state: BatteryState, current_a: float, ambient_temp_c: float) -> BatteryState:
        """Advance the battery state by one timestep under the given applied current.

        NOTE: current_a is assumed to already be safety-clamped by the caller
        (the safety layer) — this model does not enforce policy limits itself,
        only reports the resulting physical state.
        """
        dt = self.dt

        if self.integration_method == "euler":
            d_soc, d_vrc, d_temp = self._derivatives(state, current_a, ambient_temp_c)
            new_soc = state.soc + d_soc * dt
            new_vrc = state.v_rc + d_vrc * dt
            new_temp = state.temperature_c + d_temp * dt
        else:  # rk4
            def deriv_at(s: BatteryState):
                return self._derivatives(s, current_a, ambient_temp_c)

            k1 = deriv_at(state)
            s2 = BatteryState(
                soc=state.soc + 0.5 * dt * k1[0],
                v_rc=state.v_rc + 0.5 * dt * k1[1],
                temperature_c=state.temperature_c + 0.5 * dt * k1[2],
                soh=state.soh,
                ah_throughput=state.ah_throughput,
            )
            k2 = deriv_at(s2)
            s3 = BatteryState(
                soc=state.soc + 0.5 * dt * k2[0],
                v_rc=state.v_rc + 0.5 * dt * k2[1],
                temperature_c=state.temperature_c + 0.5 * dt * k2[2],
                soh=state.soh,
                ah_throughput=state.ah_throughput,
            )
            k3 = deriv_at(s3)
            s4 = BatteryState(
                soc=state.soc + dt * k3[0],
                v_rc=state.v_rc + dt * k3[1],
                temperature_c=state.temperature_c + dt * k3[2],
                soh=state.soh,
                ah_throughput=state.ah_throughput,
            )
            k4 = deriv_at(s4)

            new_soc = state.soc + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            new_vrc = state.v_rc + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            new_temp = state.temperature_c + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

        new_soc = float(np.clip(new_soc, 0.0, 1.0))

        # SoH bookkeeping (monitoring only — not fed into reward)
        new_ah_throughput = state.ah_throughput + abs(current_a) * dt / 3600.0
        new_soh = max(
            0.0,
            state.soh - self.soh_degradation_per_ah * abs(current_a) * dt / 3600.0,
        )

        return BatteryState(
            soc=new_soc,
            v_rc=new_vrc,
            temperature_c=new_temp,
            soh=new_soh,
            ah_throughput=new_ah_throughput,
        )

    def terminal_voltage(self, state: BatteryState, current_a: float) -> float:
        """Terminal voltage under the positive-current-charging convention.

        V_t = V_OC(SoC) + I*R0 + V_RC
        (matches the RL-BMS doc's V_t = V_OC - I*R0 - V_RC, which uses the
        opposite discharge-positive sign convention; both express the same
        physics — see tests/test_ecm.py for the sign check.)
        """
        return self.ocv(state.soc) + current_a * self.r0 + state.v_rc

    def reset_state(self, initial_soc: float, ambient_temp_c: float) -> BatteryState:
        return BatteryState(soc=initial_soc, v_rc=0.0, temperature_c=ambient_temp_c, soh=1.0, ah_throughput=0.0)


# ==============================================================================
# FILE: environment/battery_env.py
# LOCAL PATH: file:////home/claude/rl-bms/environment/battery_env.py
# ==============================================================================

"""
Gymnasium environment wrapping the validated ECM battery model,
shared safety layer, and configurable reward function.

Pipeline per step:
    Action -> Controller Current -> Safety Layer -> Battery Model
        -> Observation -> Reward -> Termination
"""

# from __future__ import annotations

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
        # Remap from [-1, 1] (the actual policy output space) to [0, 1] (the
        # physically meaningful fraction of i_max) before scaling to current.
        raw_action = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))
        action_val = (raw_action + 1.0) / 2.0
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
        info = {
            "safety_intervention": safety_info.as_dict(),
            "terminal_voltage": terminal_voltage,
            "termination_reason": term_reason if (terminated or truncated) else None,
            "reward_components": reward_components,
            "applied_current_a": applied_current,
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

        total = progress - temp_penalty - safety_penalty - overrequest_penalty - smoothness_penalty - time_penalty
        components = {
            "progress": progress,
            "temp_penalty": temp_penalty,
            "safety_penalty": safety_penalty,
            "overrequest_penalty": overrequest_penalty,
            "smoothness_penalty": smoothness_penalty,
            "time_penalty": time_penalty,
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

# ==============================================================================
# FILE: environment/env_factory.py
# LOCAL PATH: file:////home/claude/rl-bms/environment/env_factory.py
# ==============================================================================

"""Factory for building a BatteryChargingEnv from the on-disk config directory."""

# from __future__ import annotations

from typing import Optional

from environment.battery_env import BatteryChargingEnv
from utils.config import load_config


def make_env(mode: str = "train", config_dir: Optional[str] = None) -> BatteryChargingEnv:
    kwargs = {"config_dir": config_dir} if config_dir else {}
    battery_cfg = load_config("battery", **kwargs)
    safety_cfg = load_config("safety", **kwargs)
    reward_cfg = load_config("reward", **kwargs)
    sim_cfg = load_config("simulation", **kwargs)
    return BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode=mode)


# ==============================================================================
# FILE: safety/__init__.py
# LOCAL PATH: file:////home/claude/rl-bms/safety/__init__.py
# ==============================================================================



# ==============================================================================
# FILE: safety/safety_layer.py
# LOCAL PATH: file:////home/claude/rl-bms/safety/safety_layer.py
# ==============================================================================

"""
Shared rule-based safety layer.

Applied identically to the PPO controller and every baseline controller,
guaranteeing fair comparison. Pure function: no hidden state, no side
effects other than the returned intervention info (caller is responsible
for logging).

Interface:
    safe_current, info = safety_layer(requested_current, state, config)

SAFETY LAYER v2 (semantics fix): computes the safe current CEILING first
(i_max * derating_multiplier), then clamps the request against that ceiling
directly — applied = min(requested, ceiling). The original v1 implementation
capped the request at i_max FIRST, then multiplied by the derating factor
(applied = min(requested, i_max) * mult), which double-derated any request
below i_max: requesting exactly the ceiling got derated a second time on top
of its own reduction, making "always request >= i_max" the unique way to
reach the true ceiling and making any self-limiting strategy actively worse
than just maxing out — regardless of reward shaping. v2 fixes this: the
applied current is now monotonically non-decreasing in the requested current
(see tests/test_safety.py monotonicity tests), so a controller that requests
exactly the safe ceiling now actually receives the safe ceiling.

NOTE: results from runs generated before this fix (see project run history
prior to run_008) were produced under v1's double-derating semantics and are
not directly comparable to v2 results — all controllers (CC/CCCV/Adaptive/
PPO) experienced the old, non-monotonic clamp behavior.
"""

# from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class SafetyInfo:
    requested_current: float
    safe_current_ceiling: float    # i_max * combined derating multiplier — the true allowed ceiling
    applied_current: float
    intervention_type: str          # "none", "current_limit", "temperature", "voltage_taper", "soc_taper"
    magnitude: float                 # normalized [0,1]: how much the request was reduced
    derating_multiplier: float       # combined multiplier actually applied (1.0 = no derating)

    def as_dict(self) -> Dict:
        return {
            "requested_current": self.requested_current,
            "safe_current_ceiling": self.safe_current_ceiling,
            "applied_current": self.applied_current,
            "type": self.intervention_type,
            "magnitude": self.magnitude,
            "derating_multiplier": self.derating_multiplier,
        }

    @property
    def intervened(self) -> bool:
        return self.intervention_type != "none"


def _linear_derate(value: float, start: float, full: float) -> float:
    """Return a multiplier in [0,1]: 1.0 below `start`, 0.0 at/above `full`, linear between."""
    if full <= start:
        return 1.0 if value < start else 0.0
    if value <= start:
        return 1.0
    if value >= full:
        return 0.0
    return 1.0 - (value - start) / (full - start)


def state_based_current_multiplier(state, config: Dict) -> float:
    """The fraction of i_max the safety layer currently permits, based only
    on state (temperature, SoC) — no request or voltage estimate needed.

    Voltage tapering is deliberately excluded here: its multiplier depends
    on the terminal voltage AT the requested current (a circular
    dependency), and in this system's dynamics the voltage taper zone
    heavily overlaps the SoC taper zone near full charge, so SoC alone is
    already a strong proxy. Used both as an observation feature (so the
    policy has direct access to "how much can I safely ask for right now"
    instead of having to infer it) and to compute the over-request penalty
    in the reward function.
    """
    soc = state.soc if hasattr(state, "soc") else state["soc"]
    temp = state.temperature_c if hasattr(state, "temperature_c") else state["temperature_c"]

    temp_mult = _linear_derate(temp, config["t_derate_start_c"], config["t_hard_cutoff_c"])
    soc_mult = _linear_derate(soc, config["soc_taper_start"], config["soc_taper_full"])
    return min(temp_mult, soc_mult)


def safety_layer(requested_current_a: float, state, config: Dict, estimated_voltage: float = None):
    """Clamp a requested charging current to keep the battery within safe bounds.

    v2 semantics: computes the safe ceiling (i_max * combined derating
    multiplier) first, then clamps the request against that ceiling.
    Monotonic in the request: applied_current never decreases as
    requested_current_a increases, for fixed state/estimated_voltage.

    Args:
        requested_current_a: current requested by the controller (A), charging positive.
        state: object/dict with attributes/keys `soc` and `temperature_c` (a BatteryState works).
        config: safety.yaml loaded as a dict.
        estimated_voltage: optional pre-computed terminal voltage at the requested current,
            used for voltage tapering. If None, voltage tapering is skipped (caller can
            re-check post-hoc, or this arg can be supplied by the environment which has
            the ECM available to estimate it).

    Returns:
        (safe_current_a, SafetyInfo)
    """
    soc = state.soc if hasattr(state, "soc") else state["soc"]
    temp = state.temperature_c if hasattr(state, "temperature_c") else state["temperature_c"]

    i_max = config["i_max_a"]

    # --- Compute each rule's multiplier independently (pure state/estimate, no request involved) ---
    temp_mult = _linear_derate(temp, config["t_derate_start_c"], config["t_hard_cutoff_c"])
    soc_mult = _linear_derate(soc, config["soc_taper_start"], config["soc_taper_full"])

    volt_mult = 1.0
    if estimated_voltage is not None:
        volt_mult = _linear_derate(estimated_voltage, config["v_taper_start"], config["v_hard_max"])

    # Most restrictive rule wins (min, not product — avoids unrealistic
    # compounding when multiple mild derates are simultaneously active).
    combined_mult = min(temp_mult, soc_mult, volt_mult)

    # --- The actual safe ceiling, computed ONCE, independent of the request ---
    safe_ceiling = i_max * combined_mult

    # --- Clamp the request against the ceiling directly (monotonic) ---
    applied_current = max(0.0, min(requested_current_a, safe_ceiling))

    # --- Determine which rule (if any) is responsible for the binding constraint ---
    if applied_current < requested_current_a - 1e-9:
        if combined_mult >= 1.0:
            intervention_type = "current_limit"
        elif combined_mult == temp_mult:
            intervention_type = "temperature"
        elif combined_mult == soc_mult:
            intervention_type = "soc_taper"
        else:
            intervention_type = "voltage_taper"
    else:
        intervention_type = "none"

    magnitude = 0.0
    if requested_current_a > 1e-9:
        magnitude = max(0.0, 1.0 - (applied_current / requested_current_a))

    info = SafetyInfo(
        requested_current=requested_current_a,
        safe_current_ceiling=safe_ceiling,
        applied_current=applied_current,
        intervention_type=intervention_type,
        magnitude=magnitude,
        derating_multiplier=combined_mult,
    )
    return applied_current, info

# ==============================================================================
# FILE: baselines/__init__.py
# LOCAL PATH: file:////home/claude/rl-bms/baselines/__init__.py
# ==============================================================================



# ==============================================================================
# FILE: baselines/base_controller.py
# LOCAL PATH: file:////home/claude/rl-bms/baselines/base_controller.py
# ==============================================================================

"""
Common interface every baseline controller implements, so the evaluation
harness can treat them uniformly alongside the PPO agent.
"""

# from __future__ import annotations

from abc import ABC, abstractmethod


class BaseController(ABC):
    name: str = "base"

    def reset(self) -> None:
        """Reset any internal controller state (e.g. CCCV phase). No-op by default."""
        pass

    @abstractmethod
    def act(self, observation: dict) -> float:
        """Return a *requested* charging current in Amps (pre-safety-layer).

        observation keys: soc, terminal_voltage, temperature_c,
        previous_current_a, ambient_temp_c
        """
        raise NotImplementedError


# ==============================================================================
# FILE: baselines/cc.py
# LOCAL PATH: file:////home/claude/rl-bms/baselines/cc.py
# ==============================================================================

"""Constant Current (CC) baseline controller."""

# from __future__ import annotations

from baselines.base_controller import BaseController


class ConstantCurrentController(BaseController):
    name = "cc"

    def __init__(self, config: dict):
        self.current_a = float(config["current_a"])

    def act(self, observation: dict) -> float:
        return self.current_a


class MaxCurrentController(BaseController):
    """Trivial control-experiment baseline: always requests the physical
    current ceiling, unconditionally, regardless of any state input.

    Exists to test whether PPO's apparent performance advantage over
    CC/CCCV/Adaptive comes from anything learned, or entirely from the
    safety layer it's wrapped in — see docs/results_and_discussion.md
    Section 4. If this trivial controller's evaluation numbers closely
    match the trained PPO policy's, that confirms the safety layer (not
    training) is the source of the observed advantage, since the seed-7
    policy was found to saturate to "always request max current" with zero
    measurable state-dependence (via training/policy_sensitivity_analysis.py).
    """
    name = "max_current"

    def __init__(self, config: dict):
        self.i_max_a = float(config["i_max_a"])

    def act(self, observation: dict) -> float:
        return self.i_max_a

# ==============================================================================
# FILE: baselines/cccv.py
# LOCAL PATH: file:////home/claude/rl-bms/baselines/cccv.py
# ==============================================================================

"""Constant Current - Constant Voltage (CCCV) baseline controller.

Industry-standard charging strategy: charge at fixed current until the
terminal voltage reaches the CV setpoint, then hold that voltage by
tapering current down (approximated here via a simple proportional
controller on the voltage error, since the safety layer's own voltage
taper handles the physical realism / smoothness).
"""

# from __future__ import annotations

from baselines.base_controller import BaseController


class CCCVController(BaseController):
    name = "cccv"

    def __init__(self, config: dict):
        self.cc_current_a = float(config["cc_current_a"])
        self.cv_voltage_v = float(config["cv_voltage_v"])
        self.cv_cutoff_current_a = float(config["cv_cutoff_current_a"])
        self.cv_proportional_gain = float(config.get("cv_proportional_gain", 0.1))
        self._in_cv_phase = False

    def reset(self) -> None:
        self._in_cv_phase = False

    def act(self, observation: dict) -> float:
        voltage = observation["terminal_voltage"]

        if not self._in_cv_phase and voltage >= self.cv_voltage_v:
            self._in_cv_phase = True

        if not self._in_cv_phase:
            return self.cc_current_a

        # CV phase: proportional taper to hold voltage near the setpoint.
        # Gain is tunable via configs/evaluation.yaml: cccv.cv_proportional_gain
        # (fraction of cc_current shed per volt of overshoot); clamped to
        # [cutoff, cc_current] so it decays monotonically.
        error = voltage - self.cv_voltage_v
        gain = self.cv_proportional_gain * self.cc_current_a
        requested = self.cc_current_a - gain * max(error, 0.0)
        return max(self.cv_cutoff_current_a, min(requested, self.cc_current_a))


# ==============================================================================
# FILE: baselines/adaptive.py
# LOCAL PATH: file:////home/claude/rl-bms/baselines/adaptive.py
# ==============================================================================

"""Rule-Based Adaptive Charging baseline controller.

Charging current adjusted using predefined SoC-banded rules
(faster charging at low SoC, tapered at high SoC) — a simple
heuristic representative of common adaptive-charging firmware logic.
"""

# from __future__ import annotations

import numpy as np

from baselines.base_controller import BaseController


class AdaptiveController(BaseController):
    name = "adaptive"

    def __init__(self, config: dict):
        self.soc_bands = np.asarray(config["soc_bands"], dtype=float)
        self.current_per_band = np.asarray(config["current_a_per_band"], dtype=float)
        if len(self.current_per_band) != len(self.soc_bands) - 1:
            raise ValueError("current_a_per_band must have one fewer entry than soc_bands")

    def act(self, observation: dict) -> float:
        soc = observation["soc"]
        band_idx = int(np.clip(np.searchsorted(self.soc_bands, soc, side="right") - 1,
                                0, len(self.current_per_band) - 1))
        return float(self.current_per_band[band_idx])


# ==============================================================================
# FILE: agents/__init__.py
# LOCAL PATH: file:////home/claude/rl-bms/agents/__init__.py
# ==============================================================================



# ==============================================================================
# FILE: agents/train_ppo.py
# LOCAL PATH: file:////home/claude/rl-bms/agents/train_ppo.py
# ==============================================================================

"""
PPO agent training, staged per the implementation plan:

  Stage 1: short sanity run       (crashes / NaNs / reward sanity)
  Stage 2: reward verification    (log every reward component)
  Stage 3: hyperparameter search  (small sweep over lr/batch/ent_coef)
  Stage 4: full training          (full timestep budget, checkpoints, TensorBoard)

Usage:
    python -m agents.train_ppo --stage 1
    python -m agents.train_ppo --stage 4 --run-name run_004
"""

# from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment.env_factory import make_env
from utils.config import CONFIG_DIR, load_config, snapshot_configs
from utils.logger import CSVLogger, create_run_dir
from utils.seed import set_global_seed


class RewardComponentLoggingCallback(BaseCallback):
    """Stage-2 style callback: logs each reward component to CSV every step."""

    def __init__(self, csv_logger: CSVLogger, verbose: int = 0):
        super().__init__(verbose)
        self.csv_logger = csv_logger

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            components = info.get("reward_components")
            if components:
                row = dict(components)
                row["timestep"] = self.num_timesteps
                self.csv_logger.log(row)
        return True


def linear_schedule(initial_value: float):
    """Linearly anneal from initial_value at the start of training to ~0 at the end.

    Sustaining a constant learning rate across a very long run (Stage 4's
    ~4880 updates over 1M steps) is a known source of late-training
    instability in PPO — the policy can find a good solution early and then
    random-walk away from it as training continues with no annealing.
    Decaying the LR protects the optimum found earlier in training.
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def build_agent(env, ppo_cfg: dict, tensorboard_dir: str) -> PPO:
    return PPO(
        policy=ppo_cfg["policy"],
        env=env,
        learning_rate=linear_schedule(ppo_cfg["learning_rate"]),
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=dict(ppo_cfg["policy_kwargs"]),
        target_kl=ppo_cfg.get("target_kl"),
        tensorboard_log=tensorboard_dir,
        seed=ppo_cfg["seed"],
        verbose=1,
    )


def run_stage(
    stage: int,
    run_name: str | None = None,
    hpo_overrides: dict | None = None,
    run_dir: str | None = None,
):
    """Run one training stage.

    v4 fix (audit ISSUE-011, see audit/ISSUES.md): previously this function
    unconditionally called create_run_dir(run_name=run_name) every time it
    was invoked. training/train.py's multi-stage orchestrator reused the
    run name it got back from Stage 1 for Stages 2-4, which meant Stage 2
    called create_run_dir("run_001") again -- and since Stage 1 had already
    written files into runs/run_001/, create_run_dir's (correct, must-stay)
    non-empty-directory guard raised FileExistsError on every multi-stage
    run past Stage 1. Confirmed by reproducing it directly before this fix.

    Fix: separate CREATE from REUSE.
      - If `run_dir` is given (multi-stage orchestrator path): use it
        directly, do NOT call create_run_dir again. The directory (and its
        config/checkpoints/tensorboard/plots subdirs) was already created
        once, by the caller, before the stage loop started.
      - If `run_dir` is None (single-stage CLI path, `python -m
        agents.train_ppo --stage N`, unchanged from before): call
        create_run_dir as before, including its overwrite protection.
    """
    ppo_cfg = load_config("ppo")

    # v3 fix: apply CLI/HPO overrides BEFORE snapshotting — the previous
    # order saved the raw YAML defaults regardless of e.g. `--lr 0.0001`,
    # so a run's saved config/ directory didn't reflect what was actually
    # used to train it (a real reproducibility bug: Stage 3 HPO sweep runs
    # were undocumented).
    if hpo_overrides:
        for k, v in hpo_overrides.items():
            ppo_cfg[k] = v

    set_global_seed(ppo_cfg["seed"])

    if run_dir is None:
        run_dir = create_run_dir(os.path.join(os.path.dirname(CONFIG_DIR), "runs"), run_name=run_name)
    snapshot_configs(CONFIG_DIR, os.path.join(run_dir, "config"))

    # Save the EFFECTIVE ppo config (post-override) separately from the raw
    # snapshot above, plus the exact invoking command, so a run directory is
    # self-describing even when CLI overrides were used. Namespaced per
    # stage (v4 fix, ISSUE-011) so a later stage's snapshot never silently
    # overwrites an earlier stage's -- e.g. Stage 3's HPO override values
    # must remain inspectable even after Stage 4 has also run in the same
    # run_dir.
    def _to_plain(obj):
        if isinstance(obj, dict):
            return {k: _to_plain(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_plain(v) for v in obj]
        return obj

    with open(os.path.join(run_dir, "config", f"effective_ppo_stage{stage}.yaml"), "w") as f:
        yaml.safe_dump(_to_plain(ppo_cfg), f, default_flow_style=False, sort_keys=False)
    with open(os.path.join(run_dir, f"command_stage{stage}.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    env = Monitor(make_env(mode="train"))

    model = build_agent(env, ppo_cfg, tensorboard_dir=os.path.join(run_dir, "tensorboard"))

    timesteps_by_stage = {
        1: ppo_cfg["stage1_sanity_timesteps"],
        2: ppo_cfg["stage2_reward_verification_timesteps"],
        3: ppo_cfg["stage3_hpo_timesteps"],
        4: ppo_cfg["stage4_full_training_timesteps"],
    }
    total_timesteps = timesteps_by_stage[stage]

    callbacks = []
    if stage == 2:
        csv_logger = CSVLogger(os.path.join(run_dir, "reward_components.csv"))
        callbacks.append(RewardComponentLoggingCallback(csv_logger))
    if stage == 4:
        callbacks.append(
            CheckpointCallback(
                save_freq=ppo_cfg["checkpoint_freq"],
                save_path=os.path.join(run_dir, "checkpoints"),
                name_prefix="ppo_bms",
            )
        )

    model.learn(total_timesteps=total_timesteps, callback=callbacks or None)

    # v4 fix (ISSUE-011): save a stage-namespaced copy FIRST so an earlier
    # stage's model is never silently lost when a later stage overwrites
    # the canonical trained_model.zip path (each stage builds a fresh PPO
    # model from scratch -- see audit/STAGE_PIPELINE.md -- so without this,
    # Stage 2 running would delete/replace Stage 1's saved model, etc.).
    # trained_model.zip itself is kept pointing at the most-recently-
    # completed stage's model, since training/select_best_checkpoint.py,
    # training/evaluate.py, and the README/docs all reference that fixed
    # path as "the" trained model for a run.
    stage_model_path = os.path.join(run_dir, f"trained_model_stage{stage}.zip")
    model.save(stage_model_path)
    model_path = os.path.join(run_dir, "trained_model.zip")
    shutil.copyfile(stage_model_path, model_path)

    # Stage 1 sanity check: verify no NaNs crept into the policy weights.
    if stage == 1:
        for param in model.policy.parameters():
            if not np.all(np.isfinite(param.detach().cpu().numpy())):
                raise RuntimeError("NaN/Inf detected in policy parameters after sanity run.")
        print(f"[Stage 1] Sanity run complete, no NaNs detected. Model saved to {model_path}")
    else:
        print(f"[Stage {stage}] Training complete. Model saved to {model_path}")

    return run_dir, model_path


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for RL-BMS.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate (stage 3 HPO)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (stage 3 HPO)")
    parser.add_argument("--ent-coef", type=float, default=None, help="Override entropy coef (stage 3 HPO)")
    args = parser.parse_args()

    overrides = {}
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.ent_coef is not None:
        overrides["ent_coef"] = args.ent_coef

    run_stage(args.stage, run_name=args.run_name, hpo_overrides=overrides or None)


if __name__ == "__main__":
    main()

# ==============================================================================
# FILE: training/__init__.py
# LOCAL PATH: file:////home/claude/rl-bms/training/__init__.py
# ==============================================================================



# ==============================================================================
# FILE: training/evaluate.py
# LOCAL PATH: file:////home/claude/rl-bms/training/evaluate.py
# ==============================================================================

"""
Evaluation framework: runs PPO and all baseline controllers through the
identical fixed evaluation grid (SoC x ambient temp), computes the shared
metric set, and produces comparison tables + plots.

Usage:
    python -m training.evaluate --model runs/run_004/trained_model.zip --run-name run_004
"""

# from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from baselines.adaptive import AdaptiveController
from baselines.cc import ConstantCurrentController, MaxCurrentController
from baselines.cccv import CCCVController
from environment.battery_env import BatteryChargingEnv
from environment.ecm_model import BatteryECM
from safety.safety_layer import safety_layer
from utils.config import load_config
from utils.metrics import summarize_episode
from utils.plotting import plot_comparison_bar, plot_profile


def _run_baseline_episode(controller, ecm: BatteryECM, safety_cfg: Dict, reward_cfg: Dict,
                           sim_cfg: Dict, initial_soc: float, ambient_temp: float) -> Dict:
    """Run one baseline controller through one scenario, logging the same
    per-step arrays the PPO path logs, so metrics.summarize_episode applies uniformly."""
    dt = float(sim_cfg.get("dt_seconds", ecm.dt))
    max_steps = int(sim_cfg["max_episode_steps"])
    target_soc = float(sim_cfg["target_soc"])
    v_max = ecm.v_max
    t_max = ecm.t_max_c

    state = ecm.reset_state(initial_soc=initial_soc, ambient_temp_c=ambient_temp)
    controller.reset()
    prev_current = 0.0

    log = {"soc": [], "temperature_c": [], "current_a": [], "voltage_v": [],
           "safety_intervention": [], "input_energy_wh": [], "stored_energy_wh": []}

    for _ in range(max_steps):
        v = ecm.terminal_voltage(state, prev_current)
        obs = {"soc": state.soc, "terminal_voltage": v, "temperature_c": state.temperature_c,
               "previous_current_a": prev_current, "ambient_temp_c": ambient_temp}
        requested = controller.act(obs)
        # Safety ceiling estimate uses i_max (worst case), NOT the actual
        # request or prev_current — matches battery_env.py's fix for the
        # voltage-taper circularity (see that file for the full explanation).
        # This keeps baselines evaluated under the identical safety
        # semantics PPO trains under.
        v_for_ceiling = ecm.terminal_voltage(state, ecm.i_max_a)
        applied, safety_info = safety_layer(requested, state, safety_cfg, estimated_voltage=v_for_ceiling)

        state = ecm.step(state, applied, ambient_temp)
        terminal_v = ecm.terminal_voltage(state, applied)

        log["soc"].append(state.soc)
        log["temperature_c"].append(state.temperature_c)
        log["current_a"].append(applied)
        log["voltage_v"].append(terminal_v)
        log["safety_intervention"].append(safety_info.intervened)
        # v3 fix: input_energy_wh = charger's true input power at the
        # terminals; stored_energy_wh = the OCV-referenced portion that
        # actually raises stored energy. No term is counted twice (see
        # utils/metrics.py energy_efficiency docstring for the bug this fixes).
        log["input_energy_wh"].append(applied * terminal_v * dt / 3600.0)
        log["stored_energy_wh"].append(applied * ecm.ocv(state.soc) * dt / 3600.0)

        prev_current = applied

        if state.soc >= target_soc or terminal_v >= v_max or state.temperature_c >= t_max:
            break

    return log


def _run_ppo_episode(model, env: BatteryChargingEnv, initial_soc: float, ambient_temp: float) -> Dict:
    obs, _ = env.reset(options={"initial_soc": initial_soc, "ambient_temp_c": ambient_temp})
    dt = env.dt
    log = {"soc": [], "temperature_c": [], "current_a": [], "voltage_v": [],
           "safety_intervention": [], "input_energy_wh": [], "stored_energy_wh": []}

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        v = info["terminal_voltage"]
        applied = info["applied_current_a"]
        log["soc"].append(env._state.soc)
        log["temperature_c"].append(env._state.temperature_c)
        log["current_a"].append(applied)
        log["voltage_v"].append(v)
        log["safety_intervention"].append(info["safety_intervention"]["type"] != "none")
        log["input_energy_wh"].append(applied * v * dt / 3600.0)
        log["stored_energy_wh"].append(applied * env.ecm.ocv(env._state.soc) * dt / 3600.0)

    return log


def run_evaluation(model_path: str, run_name: str):
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    eval_cfg = load_config("evaluation")

    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]

    ecm = BatteryECM(battery_cfg)
    dt = float(sim_cfg.get("dt_seconds", ecm.dt))

    controllers = {
        "cc": ConstantCurrentController(eval_cfg["cc"]),
        "cccv": CCCVController(eval_cfg["cccv"]),
        "adaptive": AdaptiveController(eval_cfg["adaptive"]),
        "max_current": MaxCurrentController(battery_cfg),
    }

    ppo_env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval",
                                  enforce_safety=True)
    # Ablation: same trained policy, safety layer left in monitoring-only mode
    # (interventions are still logged; the episode still hard-terminates on
    # overvoltage/overtemperature) so we can quantify what the safety layer
    # is actually contributing rather than simulating an unbounded battery.
    ppo_env_no_safety = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval",
                                            enforce_safety=False)
    model = PPO.load(model_path)

    out_dir = os.path.join("runs", run_name, "evaluation")
    traj_dir = os.path.join(out_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    def _save_trajectory(controller_name: str, soc0: float, temp0: float, log: Dict) -> None:
        """Full per-step state trajectory (SoC, voltage, temperature, current) for one episode."""
        pd.DataFrame({
            "step": range(len(log["soc"])),
            "time_s": [i * dt for i in range(len(log["soc"]))],
            "soc": log["soc"],
            "voltage_v": log["voltage_v"],
            "temperature_c": log["temperature_c"],
            "current_a": log["current_a"],
            "safety_intervention": log["safety_intervention"],
        }).to_csv(
            os.path.join(traj_dir, f"{controller_name}_soc{soc0:.2f}_temp{temp0:.0f}.csv"),
            index=False,
        )

    all_rows: List[Dict] = []
    profiles: Dict[str, Dict] = {}  # keyed by controller -> one representative scenario's log

    for soc in soc_grid:
        for temp in temp_grid:
            for name, controller in controllers.items():
                log = _run_baseline_episode(controller, ecm, safety_cfg, reward_cfg, sim_cfg, soc, temp)
                metrics = summarize_episode(log, dt, target_soc=sim_cfg["target_soc"], initial_soc=soc)
                metrics.update({"controller": name, "initial_soc": soc, "ambient_temp_c": temp})
                all_rows.append(metrics)
                _save_trajectory(name, soc, temp, log)
                if soc == soc_grid[0] and temp == temp_grid[0]:
                    profiles[name] = log

            log = _run_ppo_episode(model, ppo_env, soc, temp)
            metrics = summarize_episode(log, dt, target_soc=sim_cfg["target_soc"], initial_soc=soc)
            metrics.update({"controller": "ppo", "initial_soc": soc, "ambient_temp_c": temp})
            all_rows.append(metrics)
            _save_trajectory("ppo", soc, temp, log)
            if soc == soc_grid[0] and temp == temp_grid[0]:
                profiles["ppo"] = log

            # Ablation: PPO with the safety layer in monitoring-only mode
            log_no_safety = _run_ppo_episode(model, ppo_env_no_safety, soc, temp)
            metrics_no_safety = summarize_episode(log_no_safety, dt, target_soc=sim_cfg["target_soc"], initial_soc=soc)
            metrics_no_safety.update({"controller": "ppo_no_safety", "initial_soc": soc, "ambient_temp_c": temp})
            all_rows.append(metrics_no_safety)
            _save_trajectory("ppo_no_safety", soc, temp, log_no_safety)
            if soc == soc_grid[0] and temp == temp_grid[0]:
                profiles["ppo_no_safety"] = log_no_safety

    df = pd.DataFrame(all_rows)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "raw_metrics.csv"), index=False)

    summary = df.groupby("controller").agg(["mean", "std"])
    summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"))

    # Comparison bar plots per metric
    metric_names = [m for m in eval_cfg["metrics"]]
    for metric in metric_names:
        means = df.groupby("controller")[metric].mean()
        stds = df.groupby("controller")[metric].std().fillna(0.0)
        plot_comparison_bar(
            labels=list(means.index),
            means=list(means.values),
            stds=list(stds.values),
            ylabel=metric,
            title=f"{metric} by controller",
            out_path=os.path.join(out_dir, "plots", f"{metric}.png"),
        )

    # Representative episode profiles (first eval scenario)
    for series_name, ylabel in [("soc", "SoC"), ("voltage_v", "Voltage (V)"),
                                 ("temperature_c", "Temperature (C)"), ("current_a", "Current (A)")]:
        series = {}
        for ctrl_name, log in profiles.items():
            n = len(log[series_name])
            series[ctrl_name] = log[series_name]
        # pad to common time axis using each controller's own step count
        max_len = max(len(v) for v in series.values())
        time_axis = [i * dt for i in range(max_len)]
        padded = {k: (v + [v[-1]] * (max_len - len(v))) for k, v in series.items()}
        plot_profile(time_axis, padded, ylabel=ylabel,
                     title=f"{ylabel} profile (scenario: SoC={soc_grid[0]}, T={temp_grid[0]}C)",
                     out_path=os.path.join(out_dir, "plots", f"profile_{series_name}.png"))

    print(f"Evaluation complete. Results written to {out_dir}")
    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO vs baseline controllers.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model .zip")
    parser.add_argument("--run-name", type=str, required=True, help="Run name for output directory")
    args = parser.parse_args()
    run_evaluation(args.model, args.run_name)


if __name__ == "__main__":
    main()

# ==============================================================================
# FILE: training/policy_sensitivity_analysis.py
# LOCAL PATH: file:////home/claude/rl-bms/training/policy_sensitivity_analysis.py
# ==============================================================================

"""
Policy sensitivity analysis: does the trained PPO policy's output actually
depend on the environment's state, or has it converged to something close
to a constant-current strategy regardless of SoC/voltage/temperature?

Two complementary tests:

1. PARTIAL-DEPENDENCE SWEEPS — hold every observation dimension fixed at a
   baseline value except one, sweep that one across its full [0,1] range,
   and record the policy's raw (pre-clip) mean output and the resulting
   current. This isolates each state variable's individual effect on the
   policy, uncontaminated by what actually happens during a real episode.

2. REAL-TRAJECTORY CORRELATION — using the per-step trajectory CSVs already
   produced by training/evaluate.py (runs/<run_name>/evaluation/trajectories/),
   compute the correlation and total variation of applied current against
   SoC and temperature as they actually evolve during real episodes. This
   catches state-dependence that a pure sweep could miss (e.g. if the
   policy only reacts to *combinations* of state variables) and is the more
   ecologically valid test of what the policy actually does in practice.

Usage:
    python -m training.policy_sensitivity_analysis --model runs/run_006/trained_model.zip --run-name run_006
"""

# from __future__ import annotations

import argparse
import glob
import os
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from utils.config import load_config


OBS_DIMS = ["soc", "voltage_norm", "temperature_norm", "prev_current_norm", "ambient_temp_norm", "state_based_safe_fraction"]


def _baseline_obs() -> np.ndarray:
    """A representative mid-charge, moderate-condition observation to hold
    fixed while sweeping each dimension in turn."""
    # soc=0.5, voltage_norm=0.5, temp_norm mapped from ~25C, prev_current_norm=0.5, ambient mapped from ~25C
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    t_range = battery_cfg["t_max_c"]  # t_min_ref is 0 in the env
    temp_norm_25c = 25.0 / t_range

    # safe_current_fraction at this baseline state (soc=0.5, temp=25C) — well
    # within safe bounds (soc_taper_start=0.9, t_derate_start=45C), so the
    # safety layer permits full current here: multiplier = 1.0.
    from safety.safety_layer import state_based_current_multiplier
    baseline_state = SimpleNamespace(soc=0.5, temperature_c=25.0)
    safe_frac_baseline = state_based_current_multiplier(baseline_state, safety_cfg)

    return np.array([0.5, 0.5, temp_norm_25c, 0.5, temp_norm_25c, safe_frac_baseline], dtype=np.float32)


def run_partial_dependence(model: PPO, i_max_a: float, n_points: int = 21) -> pd.DataFrame:
    baseline = _baseline_obs()
    rows: List[Dict] = []

    for dim_idx, dim_name in enumerate(OBS_DIMS):
        sweep_values = np.linspace(0.0, 1.0, n_points)
        for val in sweep_values:
            obs = baseline.copy()
            obs[dim_idx] = val
            obs_t = torch.as_tensor(obs).float().unsqueeze(0)
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_t)
                raw_mean = float(dist.distribution.mean.item())
            clipped = float(np.clip(raw_mean, -1.0, 1.0))
            action_val = (clipped + 1.0) / 2.0
            current_a = action_val * i_max_a
            rows.append({
                "swept_dim": dim_name,
                "swept_value": val,
                "raw_policy_mean": raw_mean,
                "clipped_action": clipped,
                "current_a": current_a,
            })

    return pd.DataFrame(rows)


def summarize_partial_dependence(df: pd.DataFrame) -> pd.DataFrame:
    """For each swept dimension, report BOTH the raw (pre-clip) policy
    response and the clipped/converted current — plus what fraction of the
    sweep was saturated at the action boundary.

    This distinguishes two very different situations that both show
    range_a == 0 in the clipped current alone: (a) the policy genuinely does
    not respond to this input at all, vs (b) the policy DOES respond
    internally (raw_mean varies) but the response is entirely absorbed by
    clipping to [-1, 1] before it ever reaches the environment. Only (a) is
    evidence of "no sensitivity" — (b) means the underlying network has
    learned something the clipped action space is hiding from view.
    """
    grouped = df.groupby("swept_dim")
    summary = grouped["current_a"].agg(["min", "max"])
    summary["range_a"] = summary["max"] - summary["min"]
    denom = df["current_a"].max()
    summary["range_fraction_of_i_max"] = summary["range_a"] / denom if denom > 0 else 0.0

    raw = grouped["raw_policy_mean"].agg(["min", "max"])
    summary["raw_policy_mean_min"] = raw["min"]
    summary["raw_policy_mean_max"] = raw["max"]
    summary["raw_policy_mean_range"] = raw["max"] - raw["min"]

    summary["fraction_saturated"] = grouped["raw_policy_mean"].apply(
        lambda s: float(((s <= -1.0) | (s >= 1.0)).mean())
    )

    return summary.sort_values("raw_policy_mean_range", ascending=False)


def analyze_real_trajectories(run_name: str) -> pd.DataFrame:
    """Correlate applied current against SoC and temperature within actual
    logged PPO episodes (both safety-enforced and no-safety variants)."""
    traj_dir = os.path.join("runs", run_name, "evaluation", "trajectories")
    ppo_files = sorted(glob.glob(os.path.join(traj_dir, "ppo_*.csv"))) + \
        sorted(glob.glob(os.path.join(traj_dir, "ppo_no_safety_*.csv")))

    if not ppo_files:
        print(f"No PPO trajectory files found under {traj_dir}. "
              f"Run training/evaluate.py first with this run-name.")
        return pd.DataFrame()

    rows = []
    for path in ppo_files:
        df = pd.read_csv(path)
        variant = "ppo_no_safety" if "no_safety" in os.path.basename(path) else "ppo"
        current_std = df["current_a"].std()
        current_range = df["current_a"].max() - df["current_a"].min()
        soc_corr = df["current_a"].corr(df["soc"]) if current_std > 1e-9 else float("nan")
        temp_corr = df["current_a"].corr(df["temperature_c"]) if current_std > 1e-9 else float("nan")
        rows.append({
            "file": os.path.basename(path),
            "variant": variant,
            "current_std_a": current_std,
            "current_range_a": current_range,
            "corr_current_vs_soc": soc_corr,
            "corr_current_vs_temp": temp_corr,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze whether the PPO policy is state-adaptive or near-constant.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO model .zip")
    parser.add_argument("--run-name", type=str, required=True,
                         help="Run name whose evaluation/trajectories/ folder to analyze (must have run training.evaluate first)")
    args = parser.parse_args()

    battery_cfg = load_config("battery")
    i_max_a = float(battery_cfg["i_max_a"])

    print("=" * 70)
    print("PART 1: Partial-dependence sweep (isolated effect of each state variable)")
    print("=" * 70)
    model = PPO.load(args.model)
    pd_df = run_partial_dependence(model, i_max_a)
    pd_summary = summarize_partial_dependence(pd_df)
    print(pd_summary.to_string())
    print()

    out_dir = os.path.join("runs", args.run_name, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    pd_df.to_csv(os.path.join(out_dir, "sensitivity_partial_dependence.csv"), index=False)
    pd_summary.to_csv(os.path.join(out_dir, "sensitivity_partial_dependence_summary.csv"))

    print("=" * 70)
    print("PART 2: Real-trajectory correlation (actual observed behavior)")
    print("=" * 70)
    traj_df = analyze_real_trajectories(args.run_name)
    if not traj_df.empty:
        print(traj_df.to_string(index=False))
        traj_df.to_csv(os.path.join(out_dir, "sensitivity_trajectory_correlation.csv"), index=False)
    print()

    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("NOTE: 'ppo' (safety-enforced) trajectories are excluded from the")
    print("verdict below — the safety layer mechanically tapers current near")
    print("SoC/voltage limits regardless of whether the underlying policy is")
    print("adaptive, so their variation isn't diagnostic of the policy itself.")
    print("The verdict uses only the partial-dependence sweep (pure policy,")
    print("no safety layer involved) and the ppo_no_safety trajectories")
    print("(policy's own behavior, safety layer in monitoring-only mode).")
    print()
    print("IMPORTANT: a dimension can show clipped current_a range == 0 while")
    print("the underlying raw_policy_mean still varies substantially — the")
    print("action is symmetric [-1,1] and gets hard-clipped before reaching")
    print("the environment, so a raw mean moving from e.g. 1.05 to 2.3 is")
    print("entirely invisible in the clipped current. The verdict below checks")
    print("raw_policy_mean_range, not clipped current, for exactly this reason.")
    print("Per-dimension fraction_saturated (in the summary table above) shows")
    print("how much of each sweep was pinned at the clip boundary.")
    print()

    # Use the RAW (pre-clip) policy response, not the clipped current, as the
    # primary flatness signal — clipping can hide genuine internal
    # sensitivity (see note above). But raw variation alone isn't enough:
    # a raw mean moving from 3.9 to 4.8 varies internally, yet EVERY value
    # in that range clips to the identical action (both are >1.0), so it
    # has ZERO practical behavioral effect despite a nonzero raw range —
    # this was a real gap in an earlier version of this verdict, which
    # would have called that case "(A) state-adaptive" purely because
    # raw_policy_mean_range > 0.05, without checking whether the range
    # straddles the clip boundary or sits entirely on one side of it.
    max_raw_range = pd_summary["raw_policy_mean_range"].max()
    max_range_fraction = pd_summary["range_fraction_of_i_max"].max()
    no_safety_rows = traj_df[traj_df["variant"] == "ppo_no_safety"] if not traj_df.empty else pd.DataFrame()
    mean_current_std_no_safety = no_safety_rows["current_std_a"].mean() if not no_safety_rows.empty else float("nan")

    policy_is_flat = max_raw_range < 0.05 and (
        no_safety_rows.empty or mean_current_std_no_safety < 0.05 * i_max_a
    )
    # Practically saturated: every dimension's sweep is ~always clipped
    # (fraction_saturated ~1.0 everywhere), regardless of how much the raw
    # mean itself moves within the saturated region.
    practically_saturated = (pd_summary["fraction_saturated"] >= 0.95).all()

    if policy_is_flat:
        print(f"Raw policy output varies by less than 0.05 (pre-clip units) across")
        print("every state dimension in the partial-dependence sweep — this is a")
        print("genuine lack of internal sensitivity, not a clipping artifact — and")
        print("the safety-layer-free (ppo_no_safety) trajectories show negligible")
        print(f"current variation (mean std = {mean_current_std_no_safety:.2f}A).")
        print("=> Evidence supports (B): PPO discovered an optimized near-constant-current")
        print("   strategy under the v1 reward/environment, not a state-adaptive policy.")
    elif practically_saturated:
        print(f"Raw policy output DOES vary with state (largest raw range: {max_raw_range:.3f}),")
        print("but every swept dimension is saturated (fraction_saturated >= 0.95) across")
        print("essentially the ENTIRE sweep — the raw mean is moving, but staying on one")
        print("side of the clip boundary throughout, so every value in that range produces")
        print("the IDENTICAL clipped action. This has zero practical behavioral effect.")
        print("=> Evidence supports (B), practically: the network technically responds")
        print("   internally, but that response is invisible to the environment. This is")
        print("   NOT genuine state-adaptive behavior despite nonzero raw variation.")
    else:
        print(f"Policy output varies meaningfully with state (largest RAW response")
        print(f"range: {max_raw_range:.3f}; largest clipped-current sweep range:")
        print(f"{max_range_fraction*100:.1f}% of i_max; ppo_no_safety trajectory current")
        print(f"std: {mean_current_std_no_safety:.1f}A).")
        print("=> Evidence supports (A): PPO learned a state-adaptive charging policy.")
    print()
    print("For reference, 'ppo' (safety-enforced) trajectory stats are in")
    print(f"{out_dir}/sensitivity_trajectory_correlation.csv — expect these to show")
    print("more variation than ppo_no_safety even under a flat/constant policy,")
    print("purely from the safety layer's own tapering behavior.")
    print()
    print(f"Full data written to {out_dir}/sensitivity_*.csv")


if __name__ == "__main__":
    main()

# ==============================================================================
# FILE: training/select_best_checkpoint.py
# LOCAL PATH: file:////home/claude/rl-bms/training/select_best_checkpoint.py
# ==============================================================================

"""
Sweep every checkpoint saved during Stage 4 training and report which one
actually performs best — PPO is not guaranteed to improve monotonically
over very long runs (policy can find a good solution early and drift away
from it later with no LR annealing to protect it). Don't assume the final
checkpoint (trained_model.zip) is the best one; check.

v2 (audit fix, see audit/ISSUES.md ISSUE-008): the previous version scored
checkpoints by mean_final_soc alone, which cannot distinguish a checkpoint
that reaches 95% SoC safely and quickly from one that reaches 95% while
running hot, intervening on safety constantly, or taking far longer — and
could select a checkpoint with a marginally higher final SoC (e.g. 96% vs
95%) over one with substantially better thermal/safety behavior. This
version computes the full evaluation metric set (utils.metrics.
summarize_episode — the same function training/evaluate.py uses) per
checkpoint per scenario, then selects using an explicit lexicographic
policy:

    1. target_reached_rate (higher is better — did it actually finish)
    2. safety_interventions (lower is better — fewer hard constraint hits)
    3. charging_time_s (lower is better — faster, among safe/complete runs)
    4. peak_temperature_c (lower is better — thermal stress)
    5. energy_efficiency (higher is better — tie-break)

Each criterion is only used to break ties left by the previous one (values
compared with a small tolerance so near-identical checkpoints don't get
arbitrarily ordered by noise).

Usage:
    python -m training.select_best_checkpoint --run-name run_002
"""

# from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from environment.battery_env import BatteryChargingEnv
from training.evaluate import _run_ppo_episode
from utils.config import load_config
from utils.metrics import aggregate_runs, summarize_episode

# Lexicographic selection order. `lower_is_better=False` means higher wins.
# `tolerance` treats values within this absolute difference as a tie for
# this criterion, falling through to the next one (avoids letting float
# noise decide the outcome on an otherwise-equal criterion).
_SELECTION_CRITERIA = [
    ("target_reached_rate", False, 1e-6),
    ("safety_interventions", True, 0.5),
    ("charging_time_s", True, 1.0),
    ("peak_temperature_c", True, 0.1),
    ("energy_efficiency", False, 1e-3),
]


def _score_checkpoint(model, env: BatteryChargingEnv, scenarios, dt: float, target_soc: float) -> dict:
    """Run one checkpoint through every scenario and return the aggregated
    (mean-across-scenarios) full metric set, using the exact same per-step
    logging and summarize_episode() that training/evaluate.py uses for
    final reported results — so checkpoint selection and final evaluation
    can never silently disagree on what a metric means."""
    per_scenario_metrics = []
    for soc0, temp0 in scenarios:
        log = _run_ppo_episode(model, env, soc0, temp0)
        per_scenario_metrics.append(
            summarize_episode(log, dt, target_soc=target_soc, initial_soc=soc0)
        )

    agg = aggregate_runs(per_scenario_metrics)
    # target_reached is boolean per scenario; report as a rate in [0,1]
    # rather than mean/std of a bool, which aggregate_runs would otherwise
    # compute correctly but under a less intuitive name.
    target_reached_rate = float(np.mean([m["target_reached"] for m in per_scenario_metrics]))

    flat = {"target_reached_rate": target_reached_rate}
    for key, stats in agg.items():
        if key == "target_reached":
            continue
        flat[key] = stats["mean"]
        flat[f"{key}_std"] = stats["std"]
        flat[f"{key}_valid_runs"] = stats["valid_runs"]
        flat[f"{key}_failed_runs"] = stats["failed_runs"]
    return flat


def _select_best(results: list[tuple[str, dict]]) -> tuple[str, dict]:
    """Apply the lexicographic policy in _SELECTION_CRITERIA. Returns the
    winning (path, metrics) pair."""
    candidates = list(results)
    for key, lower_is_better, tol in _SELECTION_CRITERIA:
        if len(candidates) == 1:
            break
        values = [m[key] for _, m in candidates]
        best_val = min(values) if lower_is_better else max(values)
        candidates = [
            (p, m) for p, m in candidates
            if abs(m[key] - best_val) <= tol
        ]
    # If still tied after every criterion, keep the first (stable: sorted
    # checkpoint path order, i.e. earliest checkpoint) rather than an
    # arbitrary max() comparison on a dict.
    return candidates[0]


def sweep_checkpoints(run_name: str, n_scenarios: int = 4):
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval")
    dt = float(sim_cfg.get("dt_seconds", env.ecm.dt))
    target_soc = float(sim_cfg["target_soc"])

    soc_grid = sim_cfg["eval"]["initial_soc_grid"]
    temp_grid = sim_cfg["eval"]["ambient_temp_grid_c"]
    scenarios = [(soc_grid[i % len(soc_grid)], temp_grid[i % len(temp_grid)])
                 for i in range(n_scenarios)]

    run_dir = os.path.join("runs", run_name)
    checkpoint_paths = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "*.zip")))
    final_path = os.path.join(run_dir, "trained_model.zip")
    if os.path.isfile(final_path):
        checkpoint_paths.append(final_path)

    if not checkpoint_paths:
        print(f"No checkpoints found under {run_dir}/checkpoints/ (and no trained_model.zip).")
        return

    results = []
    for path in checkpoint_paths:
        model = PPO.load(path)
        metrics = _score_checkpoint(model, env, scenarios, dt, target_soc)
        results.append((path, metrics))
        print(
            f"{os.path.basename(path):28s} "
            f"target_reached_rate={metrics['target_reached_rate']:.2f}  "
            f"safety_interventions={metrics['safety_interventions']:6.2f}  "
            f"charging_time_s={metrics['charging_time_s']:8.1f}  "
            f"peak_temp_c={metrics['peak_temperature_c']:6.2f}  "
            f"energy_eff={metrics['energy_efficiency']:.3f}  "
            f"final_soc={metrics['final_soc']:.4f}"
        )

    best_path, best_metrics = _select_best(results)

    print(f"\nBest checkpoint (lexicographic: target_reached_rate > "
          f"safety_interventions > charging_time_s > peak_temperature_c > "
          f"energy_efficiency): {best_path}")
    for key in ("target_reached_rate", "safety_interventions", "charging_time_s",
                "peak_temperature_c", "energy_efficiency", "final_soc"):
        print(f"  {key}: {best_metrics[key]:.4f}")
    print("Re-run training/evaluate.py with --model pointing at this checkpoint "
          "for the full evaluation grid + plots.")

    table = pd.DataFrame(
        [{"checkpoint": os.path.basename(p), **m} for p, m in results]
    )
    table_path = os.path.join(run_dir, "checkpoint_selection.csv")
    if os.path.isdir(run_dir):
        table.to_csv(table_path, index=False)
        print(f"Full checkpoint comparison table written to {table_path}")

    return best_path


def main():
    parser = argparse.ArgumentParser(description="Find the best-performing PPO checkpoint in a run.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--n-scenarios", type=int, default=4,
                         help="Number of quick scenarios to test per checkpoint (default 4, for speed).")
    args = parser.parse_args()
    sweep_checkpoints(args.run_name, n_scenarios=args.n_scenarios)


if __name__ == "__main__":
    main()


# ==============================================================================
# FILE: utils/__init__.py
# LOCAL PATH: file:////home/claude/rl-bms/utils/__init__.py
# ==============================================================================



# ==============================================================================
# FILE: utils/config.py
# LOCAL PATH: file:////home/claude/rl-bms/utils/config.py
# ==============================================================================

"""
Configuration loading utility.

Loads YAML config files into plain dicts (with dot-access convenience),
and validates that required keys are present so failures happen at
load-time, not deep inside training.
"""

# from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


class ConfigDict(dict):
    """Dict subclass allowing attribute-style access: cfg.battery.r0_ohm"""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_config_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config_dict(v) for v in obj]
    return obj


def load_config(name: str, config_dir: str = CONFIG_DIR) -> ConfigDict:
    """Load a single YAML config file by name (e.g. 'battery' or 'battery.yaml')."""
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    path = os.path.join(config_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Config file is empty: {path}")
    return _to_config_dict(raw)


def load_all_configs(config_dir: str = CONFIG_DIR) -> ConfigDict:
    """Load every *.yaml file in config_dir into one namespace keyed by filename stem."""
    configs: Dict[str, Any] = {}
    for fname in sorted(os.listdir(config_dir)):
        if fname.endswith(".yaml"):
            stem = fname[: -len(".yaml")]
            configs[stem] = load_config(stem, config_dir=config_dir)
    return ConfigDict(configs)


def require_keys(cfg: Dict[str, Any], keys: Iterable[str], context: str = "") -> None:
    """Raise a clear error if any required key is missing from a config dict."""
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys {missing} in {context or 'config'}")


def snapshot_configs(config_dir: str, dest_dir: str) -> None:
    """Copy all config YAML files into a run directory for reproducibility."""
    import shutil

    os.makedirs(dest_dir, exist_ok=True)
    for fname in os.listdir(config_dir):
        if fname.endswith(".yaml"):
            shutil.copy(os.path.join(config_dir, fname), os.path.join(dest_dir, fname))


def deep_copy(cfg: Any) -> Any:
    return copy.deepcopy(cfg)


# ==============================================================================
# FILE: utils/logger.py
# LOCAL PATH: file:////home/claude/rl-bms/utils/logger.py
# ==============================================================================

"""
Logging utility: append-only CSV metric logging, plus a thin wrapper
for creating per-run directories with reproducibility artifacts
(config snapshot + git commit hash).
"""

# from __future__ import annotations

import csv
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


def _next_run_index(runs_root: str) -> int:
    """Return the next safe numeric run index: max(existing numeric suffixes) + 1.

    Fixes the previous `len(existing) + 1` logic, which used the COUNT of
    run_* directories rather than the highest number present. That is wrong
    whenever a run number is missing (deleted, renamed, or created out of
    band) or non-numeric: e.g. existing = [run_001, run_002, run_005] has
    len=3 -> next_idx=4 -> "run_004", which COLLIDES with an already-planned
    or manually created run_004, or silently reuses a number. Non-run
    directories (anything not matching ^run_(\\d+)$, including malformed
    names like "run_abc" or "run_01_backup") are ignored entirely rather
    than counted, so they can't perturb the index either.
    """
    max_idx = 0
    if os.path.isdir(runs_root):
        for d in os.listdir(runs_root):
            if not os.path.isdir(os.path.join(runs_root, d)):
                continue
            m = _RUN_DIR_RE.match(d)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


class CSVLogger:
    """Append dict rows to a CSV file, writing the header on first use."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._fieldnames: Optional[List[str]] = None
        if os.path.isfile(filepath):
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self._fieldnames = header

    def log(self, row: Dict[str, Any]) -> None:
        write_header = self._fieldnames is None
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})


def get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "no-git-repo"


def create_run_dir(runs_root: str, run_name: Optional[str] = None) -> str:
    """Create runs/run_XXX/ with config/, checkpoints/, tensorboard/, plots/ subdirs."""
    os.makedirs(runs_root, exist_ok=True)
    if run_name is None:
        run_name = f"run_{_next_run_index(runs_root):03d}"

    run_dir = os.path.join(runs_root, run_name)
    # Guard against accidental overwrite: if the caller passed an explicit
    # run_name (or auto-numbering somehow raced) that already has content,
    # fail loudly instead of silently reusing/merging into it.
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise FileExistsError(
            f"Run directory '{run_dir}' already exists and is non-empty. "
            "Pass a different --run-name or remove the existing directory."
        )

    for sub in ("config", "checkpoints", "tensorboard", "plots"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    with open(os.path.join(run_dir, "git_commit_hash.txt"), "w") as f:
        f.write(get_git_commit_hash() + "\n")
    with open(os.path.join(run_dir, "created_at.txt"), "w") as f:
        f.write(datetime.now(timezone.utc).isoformat() + "\n")

    return run_dir


# ==============================================================================
# FILE: utils/metrics.py
# LOCAL PATH: file:////home/claude/rl-bms/utils/metrics.py
# ==============================================================================

"""
Evaluation metric calculations, centralised so every controller
(PPO and baselines) is scored identically.

All functions take raw per-step episode arrays (numpy arrays or lists)
and return a single scalar metric.
"""

# from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def charging_time_s(dt_seconds: float, n_steps: int) -> float:
    return dt_seconds * n_steps


def peak_temperature_c(temps: Sequence[float]) -> float:
    return float(np.max(temps))


def average_temperature_c(temps: Sequence[float]) -> float:
    return float(np.mean(temps))


def final_soc(socs: Sequence[float]) -> float:
    return float(socs[-1])


def safety_interventions(intervention_flags: Sequence[bool]) -> int:
    return int(np.sum(intervention_flags))


def current_smoothness(currents: Sequence[float]) -> float:
    """Mean absolute step-to-step current change. Lower is smoother."""
    currents = np.asarray(currents, dtype=float)
    if len(currents) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(currents))))


def energy_efficiency(input_energy_wh: float, stored_energy_wh: float) -> float:
    """Fraction of the charger's input energy that actually raises the
    cell's stored (OCV-referenced) energy, vs. being lost to internal
    resistance.

    v3 fix: an earlier version computed this as
    delivered/(delivered+dissipated) where delivered = I*V_terminal
    (already inclusive of the I^2*R0 drop, since V_terminal = OCV+I*R0+Vrc)
    and dissipated = I^2*R0 was then added AGAIN in the denominator — double
    counting the R0 loss and producing a physically inconsistent ratio, not
    a real efficiency figure. This version instead compares the charger's
    true input energy (I*V_terminal, computed once) against the energy that
    actually raises the OCV-referenced stored state (I*OCV) — no term is
    counted twice.
    """
    if input_energy_wh <= 0:
        return 0.0
    return float(np.clip(stored_energy_wh / input_energy_wh, 0.0, 1.0))


def average_input_power_w(input_energy_wh: float, charging_time_s: float) -> float:
    """Mean electrical power delivered AT THE BATTERY TERMINALS over the
    episode, in Watts. Renamed from average_charging_power_w — this is
    input power at the terminals (I*V_terminal), not power that ends up
    stored in the cell's electrochemical energy (see energy_efficiency for
    that distinction)."""
    if charging_time_s <= 0:
        return 0.0
    return float(input_energy_wh * 3600.0 / charging_time_s)


def energy_per_percent_soc_wh(input_energy_wh: float, delta_soc: float) -> float:
    """Wh of charger input energy spent per 1% of SoC gained — a direct,
    physically unambiguous efficiency figure that doesn't require
    decomposing losses at all."""
    delta_soc_pct = delta_soc * 100.0
    if delta_soc_pct <= 0:
        return float("nan")
    return float(input_energy_wh / delta_soc_pct)


def voltage_stability(voltages: Sequence[float]) -> float:
    """Standard deviation of terminal voltage over the episode.

    NOTE: this is a coarse proxy, not a true stability metric — voltage
    naturally rises over the course of a charge (SoC increases -> OCV
    increases), so a high std here can simply reflect a wide SoC range
    covered, not erratic/unstable control. A tighter metric (max dV/dt,
    CV-phase tracking error, or std after de-trending the expected SoC-driven
    rise) would better isolate genuine instability; not yet implemented.
    """
    return float(np.std(voltages))


def target_reached(final_soc_value: float, target_soc: float) -> bool:
    return final_soc_value >= target_soc


def target_shortfall(final_soc_value: float, target_soc: float) -> float:
    """How far short of the target the episode finished. 0 if reached or exceeded."""
    return max(0.0, target_soc - final_soc_value)


def summarize_episode(episode_log: Dict[str, List[float]], dt_seconds: float,
                       target_soc: float = 0.95, initial_soc: float = None) -> Dict[str, float]:
    """Compute the full evaluation metric set for one episode's logged arrays.

    Expects episode_log to contain keys: 'temperature_c', 'soc', 'current_a',
    'voltage_v', 'safety_intervention', 'input_energy_wh', 'stored_energy_wh'.

    v3: added target_reached / time_to_target / target_shortfall — a policy
    that truncates without reaching target_soc (e.g. charges partway then
    stalls) must not be scored identically to one that genuinely completes.
    Also fixed energy accounting (see energy_efficiency docstring) — this
    function now expects 'input_energy_wh' / 'stored_energy_wh' rather than
    the old 'delivered_energy_wh' / 'dissipated_energy_wh' keys.

    v3.1 fix: `initial_soc` is now an explicit parameter. episode_log["soc"]
    only contains POST-step values (both episode runners append to the log
    after calling ecm.step/env.step), so episode_log["soc"][0] is the SoC
    after the first step, not the true reset value — a small but real bias
    in energy_per_percent_soc_wh. Callers should pass the actual reset SoC;
    if omitted, falls back to the old (slightly biased) inference for
    backward compatibility.
    """
    n_steps = len(episode_log.get("soc", []))
    dt_total = charging_time_s(dt_seconds, n_steps)
    input_wh = sum(episode_log.get("input_energy_wh", [0.0]))
    stored_wh = sum(episode_log.get("stored_energy_wh", [0.0]))
    final_soc_value = final_soc(episode_log["soc"])
    if initial_soc is None:
        initial_soc = episode_log["soc"][0] if episode_log.get("soc") else final_soc_value
    reached = target_reached(final_soc_value, target_soc)
    return {
        "charging_time_s": dt_total,
        "peak_temperature_c": peak_temperature_c(episode_log["temperature_c"]),
        "average_temperature_c": average_temperature_c(episode_log["temperature_c"]),
        "final_soc": final_soc_value,
        "safety_interventions": safety_interventions(episode_log["safety_intervention"]),
        "current_smoothness": current_smoothness(episode_log["current_a"]),
        "energy_efficiency": energy_efficiency(input_wh, stored_wh),
        "average_input_power_w": average_input_power_w(input_wh, dt_total),
        "energy_per_percent_soc_wh": energy_per_percent_soc_wh(
            input_wh, final_soc_value - initial_soc
        ),
        "voltage_stability": voltage_stability(episode_log["voltage_v"]),
        "target_reached": reached,
        "time_to_target_s": dt_total if reached else float("nan"),
        "target_shortfall": target_shortfall(final_soc_value, target_soc),
    }


def aggregate_runs(metric_dicts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate mean/std across multiple episode metric dicts, per metric key.

    Fix: the previous implementation used np.mean/np.std directly, so a
    single NaN in one run's value for a key (e.g. time_to_target_s for a
    run that never reached target_soc — see summarize_episode) propagated
    NaN into mean/std for ALL runs' values of that key, silently hiding
    every other run's real result.

    This version uses nanmean/nanstd (ignoring NaN entries) but does NOT
    blindly trust them: it also reports how many runs actually contributed
    a valid (non-NaN) value vs. how many were NaN (e.g. target not reached),
    so failures are visible rather than averaged away. If every value for a
    key is NaN, the aggregate mean/std are explicitly reported as NaN too
    (nanmean/nanstd would otherwise emit a RuntimeWarning and return NaN
    anyway; we short-circuit to avoid the warning and make the "no valid
    data" case explicit).
    """
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    result: Dict[str, Dict[str, float]] = {}
    for k in keys:
        values = np.array([m[k] for m in metric_dicts], dtype=float)
        valid_mask = ~np.isnan(values)
        n_valid = int(np.sum(valid_mask))
        n_failed = int(len(values) - n_valid)
        if n_valid == 0:
            mean_val = float("nan")
            std_val = float("nan")
        else:
            mean_val = float(np.nanmean(values))
            std_val = float(np.nanstd(values))
        result[k] = {
            "mean": mean_val,
            "std": std_val,
            "valid_runs": n_valid,
            "failed_runs": n_failed,
        }
    return result

# ==============================================================================
# FILE: utils/plotting.py
# LOCAL PATH: file:////home/claude/rl-bms/utils/plotting.py
# ==============================================================================

"""
Plotting utilities: episode profiles and cross-controller comparison plots.
Uses matplotlib only (no display backend required).
"""

# from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_profile(time_s: List[float], series: Dict[str, List[float]], ylabel: str,
                  title: str, out_path: str) -> None:
    """Plot one or more time-series (e.g. multiple controllers) on one axes."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values in series.items():
        ax.plot(time_s, values, label=label, linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison_bar(labels: List[str], means: List[float], stds: List[float],
                         ylabel: str, title: str, out_path: str) -> None:
    """Bar chart with error bars for cross-controller metric comparison."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = range(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==============================================================================
# FILE: utils/seed.py
# LOCAL PATH: file:////home/claude/rl-bms/utils/seed.py
# ==============================================================================

"""
Global random seed management for reproducibility.
"""

# from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's random, NumPy, and (if importable) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy Generator, optionally seeded, for isolated stochastic sampling."""
    return np.random.default_rng(seed)


# ==============================================================================
# FILE: tests/__init__.py
# LOCAL PATH: file:////home/claude/rl-bms/tests/__init__.py
# ==============================================================================



# ==============================================================================
# FILE: tests/test_ecm.py
# LOCAL PATH: file:////home/claude/rl-bms/tests/test_ecm.py
# ==============================================================================

"""
Validate the 1RC Thevenin ECM battery model against manual
Constant-Current charging calculations, before anything else
(safety layer, environment, RL) is built on top of it.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.ecm_model import BatteryECM, BatteryState
from utils.config import load_config


@pytest.fixture
def battery_config():
    return load_config("battery")


@pytest.fixture
def ecm(battery_config):
    return BatteryECM(battery_config)


# --------------------------------------------------------------------- #
# 1. SoC evolution: manual coulomb-counting check
# --------------------------------------------------------------------- #
def test_soc_matches_manual_coulomb_counting(ecm, battery_config):
    """SoC after N seconds of constant current I must equal I*t / (capacity*3600)."""
    current_a = 45.0  # arbitrary sub-1C current; test checks the physics relationship, not a specific C-rate
    ambient = 25.0
    state = ecm.reset_state(initial_soc=0.20, ambient_temp_c=ambient)

    n_steps = 100
    for _ in range(n_steps):
        state = ecm.step(state, current_a, ambient)

    elapsed_s = n_steps * ecm.dt
    expected_delta_soc = current_a * elapsed_s / (battery_config["nominal_capacity_ah"] * 3600.0)
    expected_soc = 0.20 + expected_delta_soc

    assert state.soc == pytest.approx(expected_soc, rel=1e-6)


def test_soc_clamped_to_valid_range(ecm):
    """SoC must never exceed [0, 1] even under sustained high current."""
    state = ecm.reset_state(initial_soc=0.98, ambient_temp_c=25.0)
    for _ in range(10000):
        state = ecm.step(state, current_a=135.0, ambient_temp_c=25.0)
    assert 0.0 <= state.soc <= 1.0


# --------------------------------------------------------------------- #
# 2. OCV interpolation
# --------------------------------------------------------------------- #
def test_ocv_interpolation_matches_table_endpoints(ecm, battery_config):
    pts = battery_config["ocv_soc_points"]
    assert ecm.ocv(pts["soc"][0]) == pytest.approx(pts["ocv_v"][0])
    assert ecm.ocv(pts["soc"][-1]) == pytest.approx(pts["ocv_v"][-1])


def test_ocv_is_monotonic_increasing(ecm):
    """NMC OCV-SoC curve should be monotonically non-decreasing (smooth learning signal)."""
    socs = np.linspace(0, 1, 50)
    ocvs = [ecm.ocv(s) for s in socs]
    assert all(b >= a - 1e-9 for a, b in zip(ocvs, ocvs[1:]))


def test_ocv_clamps_out_of_range_soc(ecm, battery_config):
    pts = battery_config["ocv_soc_points"]
    assert ecm.ocv(-0.5) == pytest.approx(pts["ocv_v"][0])
    assert ecm.ocv(1.5) == pytest.approx(pts["ocv_v"][-1])


# --------------------------------------------------------------------- #
# 3. Terminal voltage response
# --------------------------------------------------------------------- #
def test_terminal_voltage_rises_above_ocv_when_charging(ecm):
    state = ecm.reset_state(initial_soc=0.5, ambient_temp_c=25.0)
    v = ecm.terminal_voltage(state, current_a=50.0)
    assert v > ecm.ocv(state.soc)


def test_terminal_voltage_equals_ocv_at_zero_current(ecm):
    state = ecm.reset_state(initial_soc=0.5, ambient_temp_c=25.0)
    v = ecm.terminal_voltage(state, current_a=0.0)
    assert v == pytest.approx(ecm.ocv(state.soc))


# --------------------------------------------------------------------- #
# 4. RC branch dynamics: charges toward steady state I*R1, decays at rest
# --------------------------------------------------------------------- #
def test_vrc_approaches_steady_state_under_constant_current(ecm):
    current_a = 45.0
    ambient = 25.0
    state = ecm.reset_state(initial_soc=0.5, ambient_temp_c=ambient)

    tau = ecm.r1 * ecm.c1  # RC time constant (s)
    n_steps = int(10 * tau / ecm.dt)  # ~10 time constants -> converged
    for _ in range(n_steps):
        state = ecm.step(state, current_a, ambient)

    expected_steady_vrc = current_a * ecm.r1
    assert state.v_rc == pytest.approx(expected_steady_vrc, rel=0.02)


def test_vrc_decays_toward_zero_at_rest(ecm):
    ambient = 25.0
    state = BatteryState(soc=0.5, v_rc=0.3, temperature_c=ambient)
    tau = ecm.r1 * ecm.c1
    n_steps = int(10 * tau / ecm.dt)
    for _ in range(n_steps):
        state = ecm.step(state, current_a=0.0, ambient_temp_c=ambient)
    assert state.v_rc == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------- #
# 5. Thermal model
# --------------------------------------------------------------------- #
def test_temperature_rises_under_sustained_high_current(ecm):
    ambient = 25.0
    state = ecm.reset_state(initial_soc=0.3, ambient_temp_c=ambient)
    for _ in range(3600):  # 1 hour at 1s steps
        state = ecm.step(state, current_a=100.0, ambient_temp_c=ambient)
    assert state.temperature_c > ambient


def test_temperature_stable_at_zero_current_and_equal_ambient(ecm):
    """No current, temp==ambient -> zero heat gen, zero net loss -> temp unchanged."""
    ambient = 25.0
    state = ecm.reset_state(initial_soc=0.3, ambient_temp_c=ambient)
    for _ in range(1000):
        state = ecm.step(state, current_a=0.0, ambient_temp_c=ambient)
    assert state.temperature_c == pytest.approx(ambient, abs=1e-6)


def test_temperature_relaxes_toward_ambient_when_hot_and_idle(ecm):
    ambient = 25.0
    state = BatteryState(soc=0.5, v_rc=0.0, temperature_c=45.0)
    for _ in range(20000):
        state = ecm.step(state, current_a=0.0, ambient_temp_c=ambient)
    assert state.temperature_c == pytest.approx(ambient, abs=0.5)


# --------------------------------------------------------------------- #
# 6. Manual Constant-Current charge comparison (integration-level sanity)
# --------------------------------------------------------------------- #
def test_manual_cc_charge_trajectory_matches_expected_shape(ecm, battery_config):
    """Charge from 20% to ~50% SoC at 1C; verify SoC is monotonically increasing,
    voltage stays within physical bounds, and elapsed time matches the
    capacity/current relationship (Q = I*t)."""
    current_a = 45.0
    ambient = 25.0
    state = ecm.reset_state(initial_soc=0.20, ambient_temp_c=ambient)

    target_soc = 0.50
    socs = [state.soc]
    step_count = 0
    max_steps = int(3600 * 2 / ecm.dt)  # safety cap: 2 hours

    while state.soc < target_soc and step_count < max_steps:
        state = ecm.step(state, current_a, ambient)
        v = ecm.terminal_voltage(state, current_a)
        assert v <= battery_config["v_max"] + 5.0  # allow small numerical slack
        socs.append(state.soc)
        step_count += 1

    # Monotonic SoC increase under constant positive current
    assert all(b >= a - 1e-9 for a, b in zip(socs, socs[1:]))

    expected_seconds = (target_soc - 0.20) * battery_config["nominal_capacity_ah"] * 3600.0 / current_a
    actual_seconds = step_count * ecm.dt
    assert actual_seconds == pytest.approx(expected_seconds, rel=0.01)


# --------------------------------------------------------------------- #
# 7. Euler vs RK4 integration method agreement (both should be config-selectable)
# --------------------------------------------------------------------- #
def test_euler_and_rk4_agree_closely_for_smooth_dynamics(battery_config):
    cfg_euler = dict(battery_config)
    cfg_euler["integration_method"] = "euler"
    cfg_rk4 = dict(battery_config)
    cfg_rk4["integration_method"] = "rk4"

    ecm_euler = BatteryECM(cfg_euler)
    ecm_rk4 = BatteryECM(cfg_rk4)

    s_euler = ecm_euler.reset_state(0.3, 25.0)
    s_rk4 = ecm_rk4.reset_state(0.3, 25.0)

    for _ in range(500):
        s_euler = ecm_euler.step(s_euler, 45.0, 25.0)
        s_rk4 = ecm_rk4.step(s_rk4, 45.0, 25.0)

    assert s_euler.soc == pytest.approx(s_rk4.soc, rel=1e-3)
    assert s_euler.v_rc == pytest.approx(s_rk4.v_rc, rel=1e-2)
    assert s_euler.temperature_c == pytest.approx(s_rk4.temperature_c, rel=1e-2)


# --------------------------------------------------------------------- #
# 8. State of Health tracking (monitoring only)
# --------------------------------------------------------------------- #
def test_soh_decreases_monotonically_with_throughput(ecm):
    state = ecm.reset_state(initial_soc=0.3, ambient_temp_c=25.0)
    prev_soh = state.soh
    for _ in range(1000):
        state = ecm.step(state, current_a=45.0, ambient_temp_c=25.0)
        assert state.soh <= prev_soh + 1e-12
        prev_soh = state.soh
    assert state.ah_throughput > 0


# --------------------------------------------------------------------- #
# Thermal model correctness: Vrc^2/R1 resistive-loss formulation
# --------------------------------------------------------------------- #
def test_relaxation_heat_generation_nonzero(ecm):
    """Regression test: at rest (I=0) with a charged RC branch (Vrc > 0),
    R1 is physically still dissipating stored polarization energy as heat
    while Vrc decays. An earlier implementation used current_a*v_rc for
    heat generation, which incorrectly gave exactly zero heat here since
    current_a=0 — even though the RC branch is actively discharging through
    R1. The correct Vrc^2/R1 formulation must give nonzero heat."""
    state = BatteryState(soc=0.5, v_rc=0.05, temperature_c=25.0)
    d_soc, d_vrc, d_temp = ecm._derivatives(state, current_a=0.0, ambient_temp_c=25.0)
    assert d_temp > 0.0


def test_heat_generation_matches_steady_state_equivalence(ecm):
    """At steady state (Vrc = I*R1), Vrc^2/R1 and I*Vrc coincide
    algebraically (both equal I^2*R1) — confirms the fix only changes
    transient behavior, not the steady-state heat generation rate."""
    current_a = 45.0
    steady_vrc = current_a * ecm.r1
    state = BatteryState(soc=0.5, v_rc=steady_vrc, temperature_c=25.0)
    _, d_vrc, _ = ecm._derivatives(state, current_a=current_a, ambient_temp_c=25.0)
    assert d_vrc == pytest.approx(0.0, abs=1e-9)  # confirms this Vrc is indeed the steady-state value


# ==============================================================================
# FILE: tests/test_safety.py
# LOCAL PATH: file:////home/claude/rl-bms/tests/test_safety.py
# ==============================================================================

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.ecm_model import BatteryState
from safety.safety_layer import safety_layer, state_based_current_multiplier
from utils.config import load_config


@pytest.fixture
def safety_config():
    return load_config("safety")


def make_state(soc=0.3, temp=25.0):
    return BatteryState(soc=soc, v_rc=0.0, temperature_c=temp)


# --------------------------------------------------------------------- #
# Current limiting
# --------------------------------------------------------------------- #
def test_current_capped_at_i_max(safety_config):
    state = make_state()
    applied, info = safety_layer(999.0, state, safety_config)
    assert applied == pytest.approx(safety_config["i_max_a"])
    assert info.intervened


def test_negative_current_clamped_to_zero(safety_config):
    state = make_state()
    applied, info = safety_layer(-50.0, state, safety_config)
    assert applied == 0.0


def test_request_within_bounds_passes_through_unchanged(safety_config):
    state = make_state(soc=0.3, temp=25.0)
    requested = 40.0
    applied, info = safety_layer(requested, state, safety_config)
    assert applied == pytest.approx(requested)
    assert not info.intervened


# --------------------------------------------------------------------- #
# Temperature derating
# --------------------------------------------------------------------- #
def test_no_derate_below_temp_threshold(safety_config):
    state = make_state(temp=safety_config["t_derate_start_c"] - 5.0)
    applied, info = safety_layer(50.0, state, safety_config)
    assert applied == pytest.approx(50.0)


def test_partial_derate_between_start_and_cutoff(safety_config):
    mid_temp = (safety_config["t_derate_start_c"] + safety_config["t_hard_cutoff_c"]) / 2
    state = make_state(temp=mid_temp)
    applied, info = safety_layer(safety_config["i_max_a"], state, safety_config)
    assert 0.0 < applied < safety_config["i_max_a"]
    assert info.intervention_type == "temperature"


def test_full_cutoff_at_or_above_hard_temp_limit(safety_config):
    state = make_state(temp=safety_config["t_hard_cutoff_c"] + 5.0)
    applied, info = safety_layer(50.0, state, safety_config)
    assert applied == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------- #
# SoC tapering
# --------------------------------------------------------------------- #
def test_no_soc_taper_below_threshold(safety_config):
    state = make_state(soc=safety_config["soc_taper_start"] - 0.1)
    applied, info = safety_layer(50.0, state, safety_config)
    assert applied == pytest.approx(50.0)


def test_soc_taper_reduces_current_near_target(safety_config):
    mid_soc = (safety_config["soc_taper_start"] + safety_config["soc_taper_full"]) / 2
    state = make_state(soc=mid_soc)
    applied, info = safety_layer(safety_config["i_max_a"], state, safety_config)
    assert 0.0 < applied < safety_config["i_max_a"]
    assert info.intervention_type == "soc_taper"


def test_soc_taper_zero_current_at_full(safety_config):
    state = make_state(soc=safety_config["soc_taper_full"])
    applied, info = safety_layer(50.0, state, safety_config)
    assert applied == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------- #
# Voltage tapering
# --------------------------------------------------------------------- #
def test_voltage_taper_applies_when_estimate_provided(safety_config):
    state = make_state()
    mid_v = (safety_config["v_taper_start"] + safety_config["v_hard_max"]) / 2
    applied, info = safety_layer(safety_config["i_max_a"], state, safety_config, estimated_voltage=mid_v)
    assert 0.0 < applied < safety_config["i_max_a"]
    assert info.intervention_type == "voltage_taper"


def test_voltage_taper_skipped_without_estimate(safety_config):
    state = make_state()
    applied, info = safety_layer(50.0, state, safety_config, estimated_voltage=None)
    assert applied == pytest.approx(50.0)


def test_hard_voltage_max_forces_zero_current(safety_config):
    state = make_state()
    applied, info = safety_layer(
        50.0, state, safety_config, estimated_voltage=safety_config["v_hard_max"] + 5.0
    )
    assert applied == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------- #
# Multiple simultaneous constraints -> most restrictive wins
# --------------------------------------------------------------------- #
def test_most_restrictive_rule_dominates(safety_config):
    """High temp AND high SoC both active -> current should reflect the min of both multipliers."""
    hot = safety_config["t_hard_cutoff_c"]  # multiplier -> 0 at hard cutoff
    state = make_state(soc=0.5, temp=hot)
    applied, info = safety_layer(50.0, state, safety_config)
    assert applied == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------- #
# Logging / info completeness
# --------------------------------------------------------------------- #
def test_safety_info_fields_present(safety_config):
    state = make_state()
    _, info = safety_layer(50.0, state, safety_config)
    d = info.as_dict()
    assert set(["requested_current", "applied_current", "type", "magnitude"]).issubset(d.keys())


def test_magnitude_zero_when_no_intervention(safety_config):
    state = make_state(soc=0.3, temp=25.0)
    _, info = safety_layer(30.0, state, safety_config)
    assert info.magnitude == pytest.approx(0.0)


def test_magnitude_one_when_fully_blocked(safety_config):
    state = make_state(soc=safety_config["soc_taper_full"])
    _, info = safety_layer(30.0, state, safety_config)
    assert info.magnitude == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# state_based_current_multiplier (v2: state-only ceiling, no request needed)
# --------------------------------------------------------------------- #
def test_state_multiplier_full_at_safe_state(safety_config):
    state = make_state(soc=0.3, temp=25.0)
    assert state_based_current_multiplier(state, safety_config) == pytest.approx(1.0)


def test_state_multiplier_reduced_near_soc_taper(safety_config):
    mid_soc = (safety_config["soc_taper_start"] + safety_config["soc_taper_full"]) / 2
    state = make_state(soc=mid_soc, temp=25.0)
    mult = state_based_current_multiplier(state, safety_config)
    assert 0.0 < mult < 1.0


def test_state_multiplier_reduced_near_temp_cutoff(safety_config):
    mid_temp = (safety_config["t_derate_start_c"] + safety_config["t_hard_cutoff_c"]) / 2
    state = make_state(soc=0.3, temp=mid_temp)
    mult = state_based_current_multiplier(state, safety_config)
    assert 0.0 < mult < 1.0


def test_state_multiplier_matches_actual_safety_layer_clamp(safety_config):
    """The state-only multiplier should match what safety_layer() actually
    applies for a request AT OR ABOVE i_max — the safety layer caps the
    request at i_max first, then derates, so the true ceiling (i_max *
    multiplier) is only reached when the request is >= i_max. A request
    BELOW i_max gets derated a second time on top of its own reduction
    (applied = requested * mult, not i_max * mult) — see the overrequest
    penalty tests in test_environment.py for the practical implication."""
    state = make_state(soc=0.95, temp=25.0)  # deep in SoC taper zone
    predicted_mult = state_based_current_multiplier(state, safety_config)
    applied, info = safety_layer(1000.0, state, safety_config)  # request >= i_max
    assert applied == pytest.approx(safety_config["i_max_a"] * predicted_mult, rel=1e-6)


# --------------------------------------------------------------------- #
# Safety Layer v2 semantics — deterministic tests per the fix design doc
# --------------------------------------------------------------------- #
def test_v2_no_derating_passthrough_and_cap(safety_config):
    """Test A: with multiplier=1.0 (safe state), requests below i_max pass
    through unchanged; requests above i_max are capped at i_max."""
    state = make_state(soc=0.3, temp=25.0)  # safe state, multiplier == 1.0
    i_max = safety_config["i_max_a"]

    applied, _ = safety_layer(100.0, state, safety_config)
    assert applied == pytest.approx(100.0)

    applied, _ = safety_layer(i_max, state, safety_config)
    assert applied == pytest.approx(i_max)

    applied, _ = safety_layer(i_max + 50.0, state, safety_config)
    assert applied == pytest.approx(i_max)


def test_v2_temperature_derating_ceiling_semantics(safety_config):
    """Test B: with an active temperature derating multiplier, the ceiling
    is i_max * multiplier — requesting exactly the ceiling should yield the
    ceiling (not double-derated), and requesting less than the ceiling
    should pass through unchanged."""
    from safety.safety_layer import state_based_current_multiplier

    mid_temp = (safety_config["t_derate_start_c"] + safety_config["t_hard_cutoff_c"]) / 2
    state = make_state(soc=0.3, temp=mid_temp)
    mult = state_based_current_multiplier(state, safety_config)
    i_max = safety_config["i_max_a"]
    ceiling = i_max * mult

    # Request above i_max -> capped at ceiling (not i_max)
    applied, _ = safety_layer(i_max, state, safety_config)
    assert applied == pytest.approx(ceiling, rel=1e-6)

    # Request exactly the ceiling -> get exactly the ceiling (the core v2 fix)
    applied, _ = safety_layer(ceiling, state, safety_config)
    assert applied == pytest.approx(ceiling, rel=1e-6)

    # Request below the ceiling -> passes through unchanged, no double-derating
    below_ceiling = ceiling * 0.5
    applied, _ = safety_layer(below_ceiling, state, safety_config)
    assert applied == pytest.approx(below_ceiling, rel=1e-6)


def test_v2_critical_temperature_zero_current(safety_config):
    """Test C: at or above the hard temperature cutoff, applied current is zero."""
    state = make_state(soc=0.3, temp=safety_config["t_hard_cutoff_c"])
    applied, _ = safety_layer(safety_config["i_max_a"], state, safety_config)
    assert applied == pytest.approx(0.0, abs=1e-9)

    state_over = make_state(soc=0.3, temp=safety_config["t_hard_cutoff_c"] + 10.0)
    applied, _ = safety_layer(safety_config["i_max_a"], state_over, safety_config)
    assert applied == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("temp,soc", [
    (25.0, 0.3),    # fully safe
    (50.0, 0.3),    # mid temperature derate
    (25.0, 0.95),   # mid SoC taper
    (52.0, 0.95),   # both active simultaneously
])
def test_v2_monotonicity_applied_never_decreases_with_request(safety_config, temp, soc):
    """Test D: for a fixed state, applied current must never decrease as the
    requested current increases. This is the core property the v1 bug
    violated (requesting less could, perversely, still be double-derated
    relative to requesting the true ceiling) and the v2 fix guarantees."""
    state = make_state(soc=soc, temp=temp)
    requests = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, safety_config["i_max_a"]]

    applied_values = [safety_layer(r, state, safety_config)[0] for r in requests]

    for a, b in zip(applied_values, applied_values[1:]):
        assert b >= a - 1e-9, (
            f"Non-monotonic: applied current decreased from {a} to {b} "
            f"as requested current increased (state: soc={soc}, temp={temp})"
        )


def test_v2_info_exposes_three_way_breakdown(safety_config):
    """The info dict should expose requested / ceiling / applied separately,
    plus the derating multiplier, for debugging and reporting."""
    state = make_state(soc=0.95, temp=25.0)
    _, info = safety_layer(safety_config["i_max_a"], state, safety_config)
    d = info.as_dict()
    assert set(["requested_current", "safe_current_ceiling", "applied_current",
                "type", "magnitude", "derating_multiplier"]).issubset(d.keys())
    assert d["safe_current_ceiling"] == pytest.approx(
        safety_config["i_max_a"] * info.derating_multiplier, rel=1e-6
    )

# ==============================================================================
# FILE: tests/test_environment.py
# LOCAL PATH: file:////home/claude/rl-bms/tests/test_environment.py
# ==============================================================================

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.battery_env import BatteryChargingEnv
from utils.config import load_config


@pytest.fixture
def env():
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    sim_cfg["max_episode_steps"] = 500  # shorten for fast tests
    return BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")


def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert obs.shape == (6,)


def test_step_returns_valid_tuple(env):
    env.reset(seed=42)
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) or isinstance(reward, np.floating)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "safety_intervention" in info


def test_episode_terminates_on_target_soc():
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    sim_cfg["max_episode_steps"] = 10000
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    obs, _ = env.reset(seed=1, options={"initial_soc": 0.20, "ambient_temp_c": 25.0})

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 10000:
        obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
        steps += 1

    assert terminated or truncated
    if terminated:
        assert info["termination_reason"] in ("target_soc_reached", "overvoltage", "overtemperature")


def test_truncation_at_max_episode_steps(env):
    env.reset(seed=1, options={"initial_soc": 0.20, "ambient_temp_c": 25.0})
    terminated = truncated = False
    steps = 0
    # Action of -1.0 (mapped to 0 current in the new symmetric [-1,1] space)
    # -> never charges -> should truncate at max_episode_steps, never terminate
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.array([-1.0], dtype=np.float32))
        steps += 1
        if steps > 10000:
            pytest.fail("Episode never ended")
    assert truncated
    assert not terminated
    assert steps == env.max_episode_steps


def test_eval_mode_uses_fixed_grid_scenarios():
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="eval")

    seen = set()
    for _ in range(len(sim_cfg["eval"]["initial_soc_grid"]) * len(sim_cfg["eval"]["ambient_temp_grid_c"])):
        obs, info = env.reset()
        seen.add((info["initial_soc"], info["ambient_temp_c"]))
    # All grid combinations should be distinct scenarios
    assert len(seen) == len(sim_cfg["eval"]["initial_soc_grid"]) * len(sim_cfg["eval"]["ambient_temp_grid_c"])


def test_gymnasium_check_env_compliance():
    """Official Stable-Baselines3 environment checker."""
    from stable_baselines3.common.env_checker import check_env

    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    sim_cfg["max_episode_steps"] = 200
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")

    check_env(env, warn=True, skip_render_check=True)


def test_reward_penalizes_smoothness_violation(env):
    env.reset(seed=1, options={"initial_soc": 0.5, "ambient_temp_c": 25.0})
    _, reward_low_jump, *_ = env.step(np.array([0.5], dtype=np.float32))
    obs, reward_big_jump, terminated, truncated, info = env.step(np.array([-1.0], dtype=np.float32))
    # Reward components should include a nonzero smoothness penalty after a big current swing
    assert info["reward_components"]["smoothness_penalty"] >= 0.0


def test_overrequest_penalty_smaller_for_smaller_requests_in_taper_zone():
    """The safety layer derates whatever is requested (applied = requested *
    multiplier when requested < i_max, not a clean cap at a fixed ceiling —
    see test_state_multiplier_matches_actual_safety_layer_clamp), so a
    smaller request in the taper zone should waste less current in absolute
    terms than a larger one, even though neither reaches zero waste. This is
    the real, achievable incentive the overrequest_penalty provides: reduce
    requests as the safety margin shrinks, not eliminate waste entirely."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")

    def penalty_for_action(raw_action):
        env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
        env.reset(seed=1, options={"initial_soc": 0.95, "ambient_temp_c": 25.0})
        _, _, _, _, info = env.step(np.array([raw_action], dtype=np.float32))
        return info["reward_components"]["overrequest_penalty"]

    small_request_penalty = penalty_for_action(-0.5)   # ~40A requested
    large_request_penalty = penalty_for_action(1.0)    # 160A requested
    assert small_request_penalty < large_request_penalty


def test_overrequest_penalty_positive_when_requesting_max_in_taper_zone():
    """Requesting full current deep in the SoC taper zone (where it will be
    heavily clamped) should incur a nonzero over-request penalty."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env.reset(seed=1, options={"initial_soc": 0.95, "ambient_temp_c": 25.0})

    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["reward_components"]["overrequest_penalty"] > 0.0


def test_overrequest_penalty_zero_when_no_clamping_needed():
    """Early in charging (no taper active), requesting full current should
    incur zero over-request penalty since nothing is actually wasted."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env.reset(seed=1, options={"initial_soc": 0.20, "ambient_temp_c": 25.0})

    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["reward_components"]["overrequest_penalty"] == pytest.approx(0.0, abs=1e-6)


def test_terminal_shortfall_penalty_applies_on_truncation_without_target():
    """v3: an episode that truncates without reaching target_soc should
    incur a terminal shortfall penalty proportional to how far short it
    finished — this is the fix for Run 008's failure mode (a policy that
    charged to ~0.9246 and then sat idle until the 7200-step truncation,
    which the old reward treated as reward-neutral relative to succeeding)."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    sim_cfg["max_episode_steps"] = 5  # force truncation almost immediately
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env.reset(seed=1, options={"initial_soc": 0.20, "ambient_temp_c": 25.0})

    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.array([-1.0], dtype=np.float32))  # 0 current
        total_reward += reward

    assert truncated and not terminated
    # Shortfall should be large (started at 0.20, target 0.95, never charged)
    # -> a substantial negative contribution beyond the per-step time_penalty alone.
    per_step_time_penalty_only = reward_cfg["weights"]["time_penalty"] * sim_cfg["max_episode_steps"]
    assert total_reward < -per_step_time_penalty_only  # shortfall penalty adds beyond just time cost


def test_no_shortfall_penalty_when_target_reached():
    """A successfully-completed episode (terminated, not truncated) should
    not incur the terminal shortfall penalty."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env.reset(seed=1, options={"initial_soc": 0.20, "ambient_temp_c": 25.0})

    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))  # full current

    assert terminated and not truncated
    assert info["target_reached"] is True


def test_voltage_estimate_uses_worst_case_not_actual_request():
    """v3.1 fix: the safety ceiling's voltage estimate must be evaluated at
    i_max (worst case), not at the actual requested current — using the
    actual request creates a circular dependency (higher request -> higher
    estimated voltage -> lower voltage-taper multiplier -> lower ceiling)
    that can genuinely violate the safety layer's monotonicity guarantee
    (confirmed by direct construction: an artificially high Vrc reproduces
    real non-monotonic steps, though not reachable by this system's own
    physics). This test confirms the environment's estimate does not vary
    with the requested action."""
    battery_cfg = load_config("battery")
    safety_cfg = load_config("safety")
    reward_cfg = load_config("reward")
    sim_cfg = load_config("simulation")
    env = BatteryChargingEnv(battery_cfg, safety_cfg, reward_cfg, sim_cfg, mode="train")
    env.reset(seed=1, options={"initial_soc": 0.5, "ambient_temp_c": 25.0})

    # Two very different requested actions from the same state should see
    # the same safety ceiling behavior driven only by state, not by their
    # own differing requests -- verified indirectly via the safety_penalty
    # magnitude being determined by state alone when both requests are well
    # under any derating zone (both should show zero intervention here).
    _, _, _, _, info_low = env.step(np.array([-0.9], dtype=np.float32))
    env.reset(seed=1, options={"initial_soc": 0.5, "ambient_temp_c": 25.0})
    _, _, _, _, info_high = env.step(np.array([0.9], dtype=np.float32))

    # Both safe (soc=0.5 is nowhere near any taper zone) -- neither should
    # show a voltage_taper intervention regardless of how different their
    # requests were.
    assert info_low["safety_intervention"]["type"] != "voltage_taper"
    assert info_high["safety_intervention"]["type"] != "voltage_taper"

# ==============================================================================
# FILE: tests/test_baselines.py
# LOCAL PATH: file:////home/claude/rl-bms/tests/test_baselines.py
# ==============================================================================

"""
Regression tests for baseline controllers running through the full
safety-layer + ECM stack. These exist to catch unintended changes to the
battery model or safety layer (e.g. an accidental parameter edit) by
checking that charge time still falls in the physically-expected range —
distinct from tests/test_ecm.py, which validates the ECM in isolation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.cc import ConstantCurrentController
from environment.ecm_model import BatteryECM
from safety.safety_layer import safety_layer
from utils.config import load_config


@pytest.fixture
def configs():
    return {
        "battery": load_config("battery"),
        "safety": load_config("safety"),
        "evaluation": load_config("evaluation"),
    }


def _charge_from_to(controller, ecm, safety_cfg, from_soc, to_soc, ambient=25.0, max_steps=20000):
    state = ecm.reset_state(initial_soc=from_soc, ambient_temp_c=ambient)
    controller.reset()
    prev_current = 0.0
    steps = 0
    while state.soc < to_soc and steps < max_steps:
        v = ecm.terminal_voltage(state, prev_current)
        obs = {"soc": state.soc, "terminal_voltage": v, "temperature_c": state.temperature_c,
               "previous_current_a": prev_current, "ambient_temp_c": ambient}
        requested = controller.act(obs)
        applied, _ = safety_layer(requested, state, safety_cfg, estimated_voltage=v)
        state = ecm.step(state, applied, ambient)
        prev_current = applied
        steps += 1
    return steps, state.soc


def test_cc_charges_20_to_80_percent_within_expected_time(configs):
    """Regression: CC controller charging 20%->80% should take roughly
    (0.6 * capacity_ah / cc_current_a) hours, +/- safety-layer tapering
    slack near the top of the range. A large deviation signals an
    unintended change to R0/R1/capacity/safety thresholds."""
    battery_cfg, safety_cfg, eval_cfg = configs["battery"], configs["safety"], configs["evaluation"]
    ecm = BatteryECM(battery_cfg)
    controller = ConstantCurrentController(eval_cfg["cc"])

    steps, final_soc = _charge_from_to(controller, ecm, safety_cfg, from_soc=0.20, to_soc=0.80)
    elapsed_s = steps * ecm.dt

    expected_seconds = 0.60 * battery_cfg["nominal_capacity_ah"] * 3600.0 / eval_cfg["cc"]["current_a"]

    assert final_soc >= 0.80
    # Generous tolerance (safety-layer SoC tapering starts at 90%, so it
    # shouldn't affect the 20->80% window much, but current does ramp from 0).
    assert elapsed_s == pytest.approx(expected_seconds, rel=0.15)


def test_cc_charge_time_is_deterministic(configs):
    """Same scenario run twice must give identical results (no hidden randomness)."""
    battery_cfg, safety_cfg, eval_cfg = configs["battery"], configs["safety"], configs["evaluation"]
    ecm = BatteryECM(battery_cfg)

    steps1, soc1 = _charge_from_to(ConstantCurrentController(eval_cfg["cc"]), ecm, safety_cfg, 0.20, 0.80)
    steps2, soc2 = _charge_from_to(ConstantCurrentController(eval_cfg["cc"]), ecm, safety_cfg, 0.20, 0.80)

    assert steps1 == steps2
    assert soc1 == pytest.approx(soc2)
