"""
Interactive 2D EV RL-BMS simulator for the RL-BMS-Driving project.

Uses the project's REAL environments, PPO models, safety layer and
drive-cycle data. This file is a visualization/demo layer only.

Controls
--------
TAB           Switch Charging / Driving
SPACE         Play / Pause
RIGHT         Advance one simulation step
R             Reset
B             Toggle PPO / baseline
1-4           UDDS / HWFET / US06 / WLTP
+ / -         Change animation speed
UP / DOWN     Change ambient temperature
ESC / Q       Quit

Mouse
-----
PLAY          Play / Pause
STEP          Advance one simulation step
RESET         Reset
SWITCH MODE   Charging <-> Driving
SWITCH CTRL   PPO <-> Baseline
NEXT CYCLE    Next driving cycle
Ambient +/-   Change ambient temperature
Speed +/-     Change simulation speed
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
from stable_baselines3 import PPO

# ---------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------------------

from baselines.cc import MaxCurrentController
from baselines.rule_based_ems import RuleBasedEMS
from environment.env_factory import make_env
from training.train_drive_ems import make_drive_ems_env


# ---------------------------------------------------------------------
# UI CONSTANTS
# ---------------------------------------------------------------------

WIDTH = 1440
HEIGHT = 900
FPS = 60

BG = (14, 18, 27)
PANEL = (24, 31, 44)
PANEL2 = (31, 40, 56)
PANEL_HOVER = (52, 68, 92)

TEXT = (235, 240, 248)
MUTED = (150, 165, 185)
ACCENT = (70, 180, 255)
GOOD = (80, 210, 130)
WARN = (255, 190, 70)
BAD = (255, 90, 100)
WHITE = (255, 255, 255)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass

    return default


def first_float(
    mapping: dict,
    keys,
    default=0.0,
) -> float:
    """Return the first finite numeric value found in mapping."""
    if not isinstance(mapping, dict):
        return default

    for key in keys:
        if key not in mapping:
            continue

        try:
            value = float(mapping[key])

            if math.isfinite(value):
                return value

        except (TypeError, ValueError):
            continue

    return default


# ---------------------------------------------------------------------
# TRACE
# ---------------------------------------------------------------------


@dataclass
class Trace:
    max_points: int = 360
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.x.append(float(len(self.x)))
        self.y.append(safe_float(value))

        if len(self.x) > self.max_points:
            self.x.pop(0)
            self.y.pop(0)

    def clear(self) -> None:
        self.x.clear()
        self.y.clear()


# ---------------------------------------------------------------------
# UI STATE
# ---------------------------------------------------------------------


@dataclass
class UIState:
    mode: str = "charging"
    playing: bool = False
    controller: str = "ppo"
    speed_multiplier: float = 2.0
    ambient_c: float = 25.0
    initial_soc: float = 0.50
    cycle_index: int = 3
    start_at_first_motion: bool = True


# ---------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------


class InteractiveSimulator:
    def __init__(self) -> None:

        pygame.init()

        pygame.display.set_caption(
            "RL-BMS-Driving — Interactive EV Simulator"
        )

        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT),
            pygame.RESIZABLE,
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "Segoe UI",
            20,
        )

        self.font_small = pygame.font.SysFont(
            "Segoe UI",
            16,
        )

        self.font_big = pygame.font.SysFont(
            "Segoe UI",
            34,
            bold=True,
        )

        self.ui = UIState()

        # -------------------------------------------------------------
        # CONFIG
        # -------------------------------------------------------------

        self.config_dir_charging = (
            ROOT / "configs" / "final_charging"
        )

        self.config_dir_driving = (
            ROOT / "configs" / "final_driving"
        )

        self.final_models = (
            ROOT / "final_models"
        )

        # -------------------------------------------------------------
        # DRIVE CYCLES
        # -------------------------------------------------------------

        self.cycle_paths = (
            self._resolve_drive_cycles()
        )

        # -------------------------------------------------------------
        # MODELS
        # -------------------------------------------------------------

        self.charging_model_path = (
            self._find_model_recursive(
                self.final_models
                / "charging_A1_50k_seed7",
                "trained_model.zip",
            )
        )

        self.driving_model_path = (
            self._find_model_recursive(
                self.final_models
                / "driving_B3_100k_seed7",
                "ppo_driving_100000_steps.zip",
            )
        )

        # -------------------------------------------------------------
        # RUNTIME
        # -------------------------------------------------------------

        self.env = None
        self.model = None
        self.baseline = None

        self.obs = None
        self.info: dict = {}

        self.done = False
        self.sim_time = 0.0

        # Driving display sample: EVEnergyEnv calculates power from the
        # CURRENT drive-cycle sample and then advances the DriveCycle.
        # Cache the sample used for the calculation so speed/acceleration
        # and power remain visually synchronized.
        self._display_speed_mps = 0.0
        self._display_accel_mps2 = 0.0
        self._display_time_s = 0.0

        # -------------------------------------------------------------
        # ANIMATION CLOCK
        # -------------------------------------------------------------

        self.last_step_wall = time.perf_counter()
        self.step_accumulator = 0.0
        self.animation_phase = 0.0

        self.MAX_STEPS_PER_FRAME = 30
        self.ANIM_BOB_RATE = 1.4

        # -------------------------------------------------------------
        # TRACES
        # -------------------------------------------------------------

        self.trace_soc = Trace()
        self.trace_temp = Trace()
        self.trace_power = Trace()
        self.trace_speed = Trace()
        self.trace_action = Trace()

        # -------------------------------------------------------------
        # MESSAGE
        # -------------------------------------------------------------

        self.message = ""
        self.message_until = 0.0

        # -------------------------------------------------------------
        # BUTTON RECTS
        # -------------------------------------------------------------

        self.play_rect = pygame.Rect(
            40,
            118,
            90,
            44,
        )

        self.step_rect = pygame.Rect(
            138,
            118,
            75,
            44,
        )

        self.reset_rect = pygame.Rect(
            221,
            118,
            75,
            44,
        )

        self.mode_rect = pygame.Rect(
            304,
            118,
            120,
            44,
        )

        self.controller_rect = pygame.Rect(
            432,
            118,
            120,
            44,
        )

        self.cycle_rect = pygame.Rect(
            560,
            118,
            115,
            44,
        )

        self.motion_rect = pygame.Rect(
            750,
            118,
            135,
            44,
        )

        self.ambient_down_rect = pygame.Rect(
            898,
            118,
            36,
            44,
        )

        self.ambient_up_rect = pygame.Rect(
            938,
            118,
            36,
            44,
        )

        self.speed_down_rect = pygame.Rect(
            1100,
            118,
            36,
            44,
        )

        self.speed_up_rect = pygame.Rect(
            1140,
            118,
            36,
            44,
        )

        self._load_mode()

    # -----------------------------------------------------------------
    # DRIVE CYCLES
    # -----------------------------------------------------------------

    def _resolve_drive_cycles(self):

        names = [
            ("UDDS", "epa_udds"),
            ("HWFET", "epa_hwfet"),
            ("US06", "epa_us06"),
            ("WLTP", "wltp_class3b"),
        ]

        result = []

        for label, folder in names:

            candidates = [
                (
                    ROOT
                    / "data"
                    / "drive_cycles"
                    / "standard"
                    / folder
                    / "cycle.csv"
                ),
                (
                    ROOT
                    / "data"
                    / "drive_cycles"
                    / f"{folder}.csv"
                ),
                (
                    ROOT
                    / "data"
                    / "drive_cycles"
                    / "standard"
                    / f"{folder}.csv"
                ),
            ]

            selected = next(
                (
                    path
                    for path in candidates
                    if path.exists()
                ),
                candidates[0],
            )

            result.append(
                (
                    label,
                    selected,
                )
            )

        return result

    # -----------------------------------------------------------------
    # MODEL FINDER
    # -----------------------------------------------------------------

    @staticmethod
    def _find_model_recursive(
        folder: Path,
        filename: str,
    ) -> Optional[Path]:

        if not folder.exists():
            return None

        direct = folder / filename

        if direct.exists():
            return direct

        matches = list(
            folder.rglob(filename)
        )

        return matches[0] if matches else None

    # -----------------------------------------------------------------
    # MESSAGE
    # -----------------------------------------------------------------

    def _set_message(
        self,
        text: str,
        seconds: float = 4.0,
    ) -> None:

        self.message = str(text)

        self.message_until = (
            time.perf_counter()
            + seconds
        )

    # -----------------------------------------------------------------
    # LOAD MODE
    # -----------------------------------------------------------------

    def _load_mode(self) -> None:

        try:

            self.ui.playing = False

            if self.ui.mode == "charging":
                self._load_charging_mode()
            else:
                self._load_driving_mode()

            self._reset_env()

        except Exception as exc:

            self.env = None
            self.model = None
            self.baseline = None
            self.obs = None

            error_text = (
                f"{type(exc).__name__}: {exc}"
            )

            self._set_message(
                f"Initialization error: {error_text}",
                15.0,
            )

            print(
                "\n========================================"
            )

            print(
                "VISUALIZER INITIALIZATION ERROR"
            )

            print(
                "========================================"
            )

            print(
                error_text,
                file=sys.stderr,
            )

    # -----------------------------------------------------------------
    # WRAPPER / ENVIRONMENT ACCESS
    # -----------------------------------------------------------------

    def _walk_envs(self):
        """Yield the environment and any nested Gym/Gymnasium wrappers."""
        env = self.env
        seen = set()

        while env is not None and id(env) not in seen:
            seen.add(id(env))
            yield env

            next_env = None
            for attr in ("env", "unwrapped"):
                candidate = getattr(env, attr, None)
                if candidate is not None and candidate is not env:
                    next_env = candidate
                    break

            if next_env is None:
                break

            env = next_env

    def _get_env_attr(self, name: str, default=None):
        """Find an attribute through the active environment/wrapper chain."""
        for env in self._walk_envs():
            try:
                value = getattr(env, name)
            except AttributeError:
                continue
            if value is not None:
                return value
        return default

    def _get_drive_cycle(self):
        """Return the real DriveCycle object through any Gym wrappers."""
        for env in self._walk_envs():
            for name in ("_drive_cycle", "drive_cycle"):
                cycle = getattr(env, name, None)
                if cycle is not None:
                    return cycle

        return None

    # -----------------------------------------------------------------
    # CHARGING MODE
    # -----------------------------------------------------------------

    def _load_charging_mode(self):

        self.env = make_env(
            mode="eval",
            config_dir=str(
                self.config_dir_charging
            ),
        )

        if (
            self.ui.controller == "ppo"
            and self.charging_model_path is not None
        ):

            self.model = PPO.load(
                str(
                    self.charging_model_path
                ),
                device="cpu",
            )

            self.baseline = None

        else:

            self.model = None

            self.baseline = (
                MaxCurrentController(
                    self.env.battery_config
                )
            )

    # -----------------------------------------------------------------
    # DRIVING MODE
    # -----------------------------------------------------------------

    def _load_driving_mode(self):

        cycle_name, cycle_path = (
            self.cycle_paths[
                self.ui.cycle_index
            ]
        )

        if not cycle_path.exists():

            raise FileNotFoundError(
                f"{cycle_name} drive cycle not found:\n"
                f"{cycle_path}"
            )

        self.env = make_drive_ems_env(
            drive_cycle_path=str(
                cycle_path
            ),
            mode="eval",
            config_dir=str(
                self.config_dir_driving
            ),
        )

        if (
            self.ui.controller == "ppo"
            and self.driving_model_path is not None
        ):

            self.model = PPO.load(
                str(
                    self.driving_model_path
                ),
                device="cpu",
            )

            self.baseline = None

        else:

            self.model = None
            self.baseline = (
                RuleBasedEMS()
            )

    # -----------------------------------------------------------------
    # RESET
    # -----------------------------------------------------------------

    def _reset_env(self) -> None:

        if self.env is None:
            return

        self.trace_soc.clear()
        self.trace_temp.clear()
        self.trace_power.clear()
        self.trace_speed.clear()
        self.trace_action.clear()

        self.done = False
        self.sim_time = 0.0

        # Driving display sample: EVEnergyEnv calculates power from the
        # CURRENT drive-cycle sample and then advances the DriveCycle.
        # Cache the sample used for the calculation so speed/acceleration
        # and power remain visually synchronized.
        self._display_speed_mps = 0.0
        self._display_accel_mps2 = 0.0
        self._display_time_s = 0.0

        options = {
            "initial_soc": float(
                np.clip(
                    self.ui.initial_soc,
                    0.05,
                    0.95,
                )
            ),
            "ambient_temp_c": float(
                self.ui.ambient_c
            ),
        }

        try:

            self.obs, self.info = (
                self.env.reset(
                    seed=42,
                    options=options,
                )
            )

        except TypeError:

            self.obs, self.info = (
                self.env.reset(seed=42)
            )

        self.last_step_wall = time.perf_counter()
        self.step_accumulator = 0.0

        # Initialize the displayed drive-cycle sample from the actual
        # current sample at reset.
        self._sync_display_cycle_sample()

        # For demo usability, optionally fast-forward only the initial
        # stationary portion of the REAL drive cycle. The environment is
        # advanced using the currently selected controller; no fake state
        # is inserted. This keeps the demo immediately visual while
        # preserving the real physics/model trajectory.
        if (
            self.ui.mode == "driving"
            and self.ui.start_at_first_motion
        ):
            self._fast_forward_to_first_motion()

        self.trace_soc.clear()
        self.trace_temp.clear()
        self.trace_power.clear()
        self.trace_speed.clear()
        self.trace_action.clear()

        self._append_state_trace(
            info=self.info,
            action=0.0,
            reward=0.0,
        )

    def _sync_display_cycle_sample(self) -> None:
        """Cache the drive-cycle sample used for the most recent env computation."""
        if self.ui.mode != "driving":
            self._display_speed_mps = 0.0
            self._display_accel_mps2 = 0.0
            self._display_time_s = self.sim_time
            return

        cycle = self._get_drive_cycle()
        if cycle is None:
            return

        self._display_speed_mps = safe_float(cycle.current_speed())
        self._display_accel_mps2 = safe_float(cycle.current_acceleration())
        self._display_time_s = safe_float(cycle.current_time())

    def _fast_forward_to_first_motion(self) -> None:
        """Advance the REAL selected controller through the initial stop.

        The WLTP file in this project is stationary for the first 30 s.
        We fast-forward only that real stationary portion, then execute one
        additional real step at the first moving sample so the displayed
        speed/acceleration/power are synchronized.
        """
        cycle = self._get_drive_cycle()
        if cycle is None:
            return

        max_steps = min(max(1, len(cycle)), 120)

        # Move until the cycle reaches its first non-zero speed sample.
        for _ in range(max_steps):
            if safe_float(cycle.current_speed()) > 0.0:
                break

            if self.done:
                return

            self._step_once()

        # At the first moving sample, execute one real controller step.
        # _step_once() caches the pre-step drive-cycle sample, so its power
        # and displayed speed remain synchronized.
        if (
            not self.done
            and safe_float(cycle.current_speed()) > 0.0
        ):
            self._step_once()

    def _current_drive_time(self) -> float:
        if self.ui.mode != "driving":
            return self.sim_time
        return self._display_time_s

    # -----------------------------------------------------------------
    # MODE SWITCH
    # -----------------------------------------------------------------

    def toggle_mode(self):

        self.ui.mode = (
            "driving"
            if self.ui.mode == "charging"
            else "charging"
        )

        self.ui.controller = "ppo"

        self._load_mode()

        self._set_message(
            f"Switched to "
            f"{self.ui.mode.upper()} mode"
        )

    # -----------------------------------------------------------------
    # CONTROLLER SWITCH
    # -----------------------------------------------------------------

    def toggle_controller(self):

        self.ui.controller = (
            "baseline"
            if self.ui.controller == "ppo"
            else "ppo"
        )

        self._load_mode()

        if self.ui.mode == "driving":
            label = (
                "RULE-BASED"
                if self.ui.controller
                == "baseline"
                else "PPO"
            )
        else:
            label = (
                "MAX CURRENT"
                if self.ui.controller
                == "baseline"
                else "PPO"
            )

        self._set_message(
            f"Controller: {label}"
        )

    # -----------------------------------------------------------------
    # CYCLE SWITCH
    # -----------------------------------------------------------------

    def set_cycle(self, index: int):

        if self.ui.mode != "driving":
            return

        self.ui.cycle_index = (
            index
            % len(self.cycle_paths)
        )

        self._load_mode()

        self._set_message(
            "Drive cycle: "
            + self.cycle_paths[
                self.ui.cycle_index
            ][0]
        )

    # -----------------------------------------------------------------
    # ACTION
    # -----------------------------------------------------------------

    def _action(self) -> np.ndarray:

        if self.obs is None:
            return np.array(
                [0.0],
                dtype=np.float32,
            )

        # PPO
        if (
            self.ui.controller == "ppo"
            and self.model is not None
        ):

            action, _ = (
                self.model.predict(
                    self.obs,
                    deterministic=True,
                )
            )

            return np.asarray(
                action,
                dtype=np.float32,
            ).reshape(-1)

        # Charging baseline
        if (
            self.ui.mode == "charging"
            and self.baseline is not None
        ):

            current = safe_float(
                self.baseline.act(
                    self.obs
                )
            )

            i_max = safe_float(
                getattr(
                    self.env,
                    "i_max",
                    160.0,
                ),
                160.0,
            )

            action = (
                2.0
                * current
                / max(i_max, 1e-6)
                - 1.0
            )

            return np.array(
                [
                    np.clip(
                        action,
                        -1.0,
                        1.0,
                    )
                ],
                dtype=np.float32,
            )

        # Driving baseline
        if (
            self.ui.mode == "driving"
            and self.baseline is not None
        ):

            return np.array(
                [
                    safe_float(
                        self.baseline.act(
                            self.obs
                        )
                    )
                ],
                dtype=np.float32,
            )

        return np.array(
            [0.0],
            dtype=np.float32,
        )

    # -----------------------------------------------------------------
    # STEP ONE REAL SIMULATION STEP
    # -----------------------------------------------------------------

    def _step_once(self) -> None:

        if (
            self.env is None
            or self.obs is None
            or self.done
        ):
            return

        # EVEnergyEnv computes power from the CURRENT drive-cycle sample
        # and advances the cycle only at the end of env.step(). Cache the
        # pre-step sample so displayed speed/acceleration match the power
        # and reward produced by this step.
        self._sync_display_cycle_sample()

        action = self._action()

        try:

            result = self.env.step(
                action
            )

        except Exception as exc:

            self.ui.playing = False

            error_text = (
                f"{type(exc).__name__}: {exc}"
            )

            self._set_message(
                f"Step error: {error_text}",
                15.0,
            )

            print(
                f"[SIMULATOR STEP ERROR] "
                f"{error_text}",
                file=sys.stderr,
            )

            return

        # Gymnasium 5-item API
        if len(result) == 5:

            (
                self.obs,
                reward,
                terminated,
                truncated,
                info,
            ) = result

            self.done = bool(
                terminated or truncated
            )

        # Old Gym 4-item API
        elif len(result) == 4:

            (
                self.obs,
                reward,
                done,
                info,
            ) = result

            self.done = bool(done)

        else:

            raise RuntimeError(
                "Unexpected env.step() "
                f"result length: {len(result)}"
            )

        self.info = (
            info
            if isinstance(info, dict)
            else {}
        )

        self.sim_time += safe_float(
            self._get_env_attr("dt", 1.0),
            1.0,
        )

        self._append_state_trace(
            info=self.info,
            action=safe_float(
                action[0]
            ),
            reward=safe_float(
                reward
            ),
        )

        if self.done:

            self.ui.playing = False

            self._set_message(
                "Episode finished — "
                "press RESET"
            )

    # -----------------------------------------------------------------
    # TEMPERATURE
    # -----------------------------------------------------------------

    def _get_temperature(self) -> float:

        info_value = first_float(
            self.info,
            [
                "battery_temperature_c",
                "temperature_c",
                "temperature",
            ],
            default=float("nan"),
        )

        if math.isfinite(
            info_value
        ):
            return info_value

        state = self._get_env_attr(
            "_state",
            None,
        )

        if state is not None:

            state_value = safe_float(
                getattr(
                    state,
                    "temperature_c",
                    float("nan"),
                ),
                float("nan"),
            )

            if math.isfinite(
                state_value
            ):
                return state_value

        for attr in (
            "temperature_c",
            "battery_temperature_c",
        ):

            attr_value = safe_float(
                getattr(
                    self.env,
                    attr,
                    float("nan"),
                ),
                float("nan"),
            )

            if math.isfinite(
                attr_value
            ):
                return attr_value

        return self.ui.ambient_c

    # -----------------------------------------------------------------
    # SOC
    # -----------------------------------------------------------------

    def _get_soc(self) -> float:

        info_value = first_float(
            self.info,
            [
                "soc",
                "battery_soc",
            ],
            default=float("nan"),
        )

        if math.isfinite(
            info_value
        ):
            return info_value

        state = getattr(
            self.env,
            "_state",
            None,
        )

        if state is not None:

            state_value = safe_float(
                getattr(
                    state,
                    "soc",
                    float("nan"),
                ),
                float("nan"),
            )

            if math.isfinite(
                state_value
            ):
                return state_value

        return self.ui.initial_soc

    # -----------------------------------------------------------------
    # SPEED
    # -----------------------------------------------------------------

    def _get_speed(self) -> float:

        if self.ui.mode == "driving":
            return self._display_speed_mps * 3.6

        return 0.0

    def _get_acceleration(self) -> float:

        if self.ui.mode == "driving":
            return self._display_accel_mps2

        return 0.0

    # -----------------------------------------------------------------
    # POWER
    # -----------------------------------------------------------------

    def _get_power(self) -> float:

        if self.ui.mode == "charging":

            current = first_float(
                self.info,
                [
                    "applied_current_a",
                    "applied_current",
                ],
                default=float("nan"),
            )

            voltage = first_float(
                self.info,
                [
                    "terminal_voltage",
                    "voltage_v",
                    "voltage",
                ],
                default=float("nan"),
            )

            if (
                math.isfinite(current)
                and math.isfinite(voltage)
            ):
                return (
                    current
                    * voltage
                    / 1000.0
                )

            return 0.0

        power_w = first_float(
            self.info,
            [
                "applied_power_w",
                "battery_power_w",
            ],
            default=0.0,
        )

        return power_w / 1000.0

    # -----------------------------------------------------------------
    # REGEN
    # -----------------------------------------------------------------

    def _get_regen(self) -> float:

        value = first_float(
            self.info,
            [
                "regen_power_w",
                "applied_regen_power_w",
            ],
            default=float("nan"),
        )

        if math.isfinite(
            value
        ):
            return value / 1000.0

        applied_power = first_float(
            self.info,
            [
                "applied_power_w",
            ],
            default=0.0,
        )

        # Project convention:
        # positive applied battery power = charging/regen.
        if applied_power > 0:
            return (
                applied_power
                / 1000.0
            )

        return 0.0

    # -----------------------------------------------------------------
    # TRACE
    # -----------------------------------------------------------------

    def _append_state_trace(
        self,
        info: Optional[dict] = None,
        action: float = 0.0,
        reward: float = 0.0,
    ) -> None:

        self.trace_soc.add(
            self._get_soc() * 100.0
        )

        self.trace_temp.add(
            self._get_temperature()
        )

        self.trace_action.add(
            action
        )

        self.trace_power.add(
            self._get_power()
        )

        if self.ui.mode == "driving":

            self.trace_speed.add(
                self._get_speed()
            )

        else:

            self.trace_speed.add(
                0.0
            )

    # -----------------------------------------------------------------
    # CURRENT METRICS
    # -----------------------------------------------------------------

    def _current_metrics(self) -> dict:

        if self.env is None:
            return {}

        temperature = (
            self._get_temperature()
        )

        soc = self._get_soc()

        metrics = {
            "SOC": soc * 100.0,
            "Temperature": temperature,
            "Ambient": first_float(
                self.info,
                [
                    "ambient_temp_c",
                ],
                default=self.ui.ambient_c,
            ),
        }

        if self.ui.mode == "charging":

            voltage = first_float(
                self.info,
                [
                    "terminal_voltage",
                    "voltage_v",
                    "voltage",
                ],
                default=float("nan"),
            )

            if not math.isfinite(
                voltage
            ):

                state = getattr(
                    self.env,
                    "_state",
                    None,
                )

                if state is not None:

                    try:

                        voltage = safe_float(
                            self.env.ecm.terminal_voltage(
                                state,
                                safe_float(
                                    getattr(
                                        self.env,
                                        "_prev_current_a",
                                        0.0,
                                    )
                                ),
                            )
                        )

                    except Exception:
                        voltage = 0.0

            applied_current = first_float(
                self.info,
                [
                    "applied_current_a",
                    "applied_current",
                ],
                default=float("nan"),
            )

            if not math.isfinite(
                applied_current
            ):

                applied_current = safe_float(
                    getattr(
                        self.env,
                        "_prev_current_a",
                        0.0,
                    )
                )

            metrics.update(
                {
                    "Voltage": voltage,
                    "Applied Current": (
                        applied_current
                    ),
                    "Target": (
                        safe_float(
                            getattr(
                                self.env,
                                "target_soc",
                                0.95,
                            )
                        )
                        * 100.0
                    ),
                    "Step": int(
                        getattr(
                            self.env,
                            "_step_count",
                            0,
                        )
                    ),
                    "Safety": str(
                        (
                            self.info.get(
                                "safety_intervention",
                                {},
                            )
                            or {}
                        ).get(
                            "type",
                            "none",
                        )
                    ),
                }
            )

        else:

            deficit_w = first_float(
                self.info,
                [
                    "power_deficit_w",
                ],
                default=0.0,
            )

            dt = safe_float(
                self._get_env_attr("dt", 1.0),
                1.0,
            )

            metrics.update(
                {
                    "Speed": self._get_speed(),
                    "Battery Power": (
                        self._get_power()
                    ),
                    "Regen": self._get_regen(),
                    "Deficit Wh": (
                        deficit_w
                        * dt
                        / 3600.0
                    ),
                    "Safety": str(
                        (
                            self.info.get(
                                "safety_intervention",
                                {},
                            )
                            or {}
                        ).get(
                            "type",
                            "none",
                        )
                    ),
                }
            )

        return metrics

    # -----------------------------------------------------------------
    # HANDLE EVENTS
    # -----------------------------------------------------------------

    def handle_event(
        self,
        event: pygame.event.Event,
    ) -> None:

        # =============================================================
        # WINDOW
        # =============================================================

        if event.type == pygame.QUIT:
            raise SystemExit

        # =============================================================
        # KEYBOARD
        # =============================================================

        if event.type == pygame.KEYDOWN:

            if event.key in (
                pygame.K_ESCAPE,
                pygame.K_q,
            ):
                raise SystemExit

            elif event.key == pygame.K_TAB:

                self.toggle_mode()

            elif event.key == pygame.K_SPACE:

                self.ui.playing = (
                    not self.ui.playing
                )

                self._set_message(
                    "PLAYING"
                    if self.ui.playing
                    else "PAUSED"
                )

            elif event.key == pygame.K_RIGHT:

                self._step_once()

            elif event.key == pygame.K_r:

                self._reset_env()

                self._set_message(
                    "RESET"
                )

            elif event.key == pygame.K_b:

                self.toggle_controller()

            elif event.key == pygame.K_m:

                self.ui.start_at_first_motion = (
                    not self.ui.start_at_first_motion
                )
                self._set_message(
                    "Start at first motion: "
                    + ("ON" if self.ui.start_at_first_motion else "OFF")
                )
                if self.ui.mode == "driving":
                    self._reset_env()

            elif event.key == pygame.K_1:

                self.set_cycle(0)

            elif event.key == pygame.K_2:

                self.set_cycle(1)

            elif event.key == pygame.K_3:

                self.set_cycle(2)

            elif event.key == pygame.K_4:

                self.set_cycle(3)

            elif event.key in (
                pygame.K_EQUALS,
                pygame.K_PLUS,
            ):

                self.ui.speed_multiplier = min(
                    8.0,
                    self.ui.speed_multiplier
                    * 1.5,
                )

                self._set_message(
                    f"Speed: "
                    f"{self.ui.speed_multiplier:.1f}x"
                )

            elif event.key == pygame.K_MINUS:

                self.ui.speed_multiplier = max(
                    0.25,
                    self.ui.speed_multiplier
                    / 1.5,
                )

                self._set_message(
                    f"Speed: "
                    f"{self.ui.speed_multiplier:.1f}x"
                )

            elif event.key == pygame.K_UP:

                self.ui.ambient_c = min(
                    50.0,
                    self.ui.ambient_c + 2.0,
                )

                self._reset_env()

                self._set_message(
                    f"Ambient: "
                    f"{self.ui.ambient_c:.0f} °C"
                )

            elif event.key == pygame.K_DOWN:

                self.ui.ambient_c = max(
                    15.0,
                    self.ui.ambient_c - 2.0,
                )

                self._reset_env()

                self._set_message(
                    f"Ambient: "
                    f"{self.ui.ambient_c:.0f} °C"
                )

        # =============================================================
        # MOUSE
        # =============================================================

        elif event.type == pygame.MOUSEBUTTONDOWN:

            if event.button != 1:
                return

            mouse_pos = event.pos

            if self.play_rect.collidepoint(
                mouse_pos
            ):

                self.ui.playing = (
                    not self.ui.playing
                )

                self._set_message(
                    "PLAYING"
                    if self.ui.playing
                    else "PAUSED"
                )

                return

            if self.step_rect.collidepoint(
                mouse_pos
            ):

                self._step_once()

                self._set_message(
                    "Advanced 1 simulation step"
                )

                return

            if self.reset_rect.collidepoint(
                mouse_pos
            ):

                self._reset_env()

                self._set_message(
                    "RESET"
                )

                return

            if self.mode_rect.collidepoint(
                mouse_pos
            ):

                self.toggle_mode()

                return

            if self.controller_rect.collidepoint(
                mouse_pos
            ):

                self.toggle_controller()

                return

            if self.cycle_rect.collidepoint(
                mouse_pos
            ):

                self.set_cycle(
                    self.ui.cycle_index + 1
                )

                return

            if self.motion_rect.collidepoint(
                mouse_pos
            ):

                self.ui.start_at_first_motion = (
                    not self.ui.start_at_first_motion
                )

                self._set_message(
                    "Start at first motion: "
                    + (
                        "ON"
                        if self.ui.start_at_first_motion
                        else "OFF"
                    )
                )

                if self.ui.mode == "driving":
                    self._reset_env()

                return

            if self.ambient_down_rect.collidepoint(
                mouse_pos
            ):

                self.ui.ambient_c = max(
                    15.0,
                    self.ui.ambient_c - 2.0,
                )

                self._reset_env()

                self._set_message(
                    f"Ambient: "
                    f"{self.ui.ambient_c:.0f} °C"
                )

                return

            if self.ambient_up_rect.collidepoint(
                mouse_pos
            ):

                self.ui.ambient_c = min(
                    50.0,
                    self.ui.ambient_c + 2.0,
                )

                self._reset_env()

                self._set_message(
                    f"Ambient: "
                    f"{self.ui.ambient_c:.0f} °C"
                )

                return

            if self.speed_down_rect.collidepoint(
                mouse_pos
            ):

                self.ui.speed_multiplier = max(
                    0.25,
                    self.ui.speed_multiplier
                    / 1.5,
                )

                self._set_message(
                    f"Speed: "
                    f"{self.ui.speed_multiplier:.1f}x"
                )

                return

            if self.speed_up_rect.collidepoint(
                mouse_pos
            ):

                self.ui.speed_multiplier = min(
                    8.0,
                    self.ui.speed_multiplier
                    * 1.5,
                )

                self._set_message(
                    f"Speed: "
                    f"{self.ui.speed_multiplier:.1f}x"
                )

                return

        elif event.type == pygame.MOUSEWHEEL:

            if event.y > 0:

                self.ui.speed_multiplier = min(
                    8.0,
                    self.ui.speed_multiplier
                    * 1.2,
                )

            elif event.y < 0:

                self.ui.speed_multiplier = max(
                    0.25,
                    self.ui.speed_multiplier
                    / 1.2,
                )

            self._set_message(
                f"Speed: "
                f"{self.ui.speed_multiplier:.1f}x"
            )

    # -----------------------------------------------------------------
    # UPDATE
    # -----------------------------------------------------------------

    def update(self):

        now = time.perf_counter()

        dt_wall = (
            now - self.last_step_wall
        )

        self.last_step_wall = now

        dt_wall = min(
            dt_wall,
            0.25,
        )

        self.animation_phase += (
            dt_wall
            * self.ANIM_BOB_RATE
        )

        if not self.ui.playing:
            return

        env_dt = safe_float(
            self._get_env_attr("dt", 1.0),
            1.0,
        )

        self.step_accumulator += (
            dt_wall
            * self.ui.speed_multiplier
        )

        steps_to_run = int(
            self.step_accumulator
            / max(env_dt, 1e-6)
        )

        steps_to_run = min(
            steps_to_run,
            self.MAX_STEPS_PER_FRAME,
        )

        self.step_accumulator -= (
            steps_to_run
            * env_dt
        )

        for _ in range(
            steps_to_run
        ):

            self._step_once()

            if self.done:

                self.step_accumulator = 0.0
                break

    # -----------------------------------------------------------------
    # DRAW TEXT
    # -----------------------------------------------------------------

    def draw_text(
        self,
        text: str,
        x: int,
        y: int,
        color=TEXT,
        font=None,
    ):

        surface = (
            font or self.font
        ).render(
            str(text),
            True,
            color,
        )

        self.screen.blit(
            surface,
            (x, y),
        )

    # -----------------------------------------------------------------
    # PANEL
    # -----------------------------------------------------------------

    def rounded_panel(
        self,
        rect: pygame.Rect,
        color=PANEL,
        radius=18,
    ):

        pygame.draw.rect(
            self.screen,
            color,
            rect,
            border_radius=radius,
        )

    # -----------------------------------------------------------------
    # HEADER
    # -----------------------------------------------------------------

    def draw_header(self):

        self.draw_text(
            "RL-BMS-Driving",
            30,
            24,
            WHITE,
            self.font_big,
        )

        self.draw_text(
            "Interactive EV "
            "energy-management simulator",
            36,
            66,
            MUTED,
            self.font_small,
        )

        mode = (
            self.ui.mode.upper()
        )

        if self.ui.controller == "ppo":

            controller = "PPO"

        elif self.ui.mode == "charging":

            controller = "MAX CURRENT"

        else:

            controller = "RULE-BASED"

        self.draw_text(
            f"MODE: {mode}",
            1080,
            28,
            ACCENT,
            self.font,
        )

        self.draw_text(
            f"CONTROLLER: {controller}",
            1080,
            58,
            (
                GOOD
                if self.ui.controller == "ppo"
                else WARN
            ),
            self.font_small,
        )

    # -----------------------------------------------------------------
    # BUTTON
    # -----------------------------------------------------------------

    def draw_button(
        self,
        rect: pygame.Rect,
        label: str,
        mouse_pos,
    ):

        hovered = (
            rect.collidepoint(
                mouse_pos
            )
        )

        color = (
            PANEL_HOVER
            if hovered
            else PANEL2
        )

        self.rounded_panel(
            rect,
            color,
            12,
        )

        if hovered:

            pygame.draw.rect(
                self.screen,
                ACCENT,
                rect,
                2,
                border_radius=12,
            )

        text_surface = (
            self.font_small.render(
                label,
                True,
                TEXT,
            )
        )

        text_rect = (
            text_surface.get_rect(
                center=rect.center
            )
        )

        self.screen.blit(
            text_surface,
            text_rect,
        )

    # -----------------------------------------------------------------
    # CONTROLS
    # -----------------------------------------------------------------

    def draw_controls(self):

        mouse_pos = pygame.mouse.get_pos()

        self.draw_button(
            self.play_rect,
            "PAUSE" if self.ui.playing else "PLAY",
            mouse_pos,
        )

        self.draw_button(
            self.step_rect,
            "STEP",
            mouse_pos,
        )

        self.draw_button(
            self.reset_rect,
            "RESET",
            mouse_pos,
        )

        self.draw_button(
            self.mode_rect,
            "SWITCH MODE",
            mouse_pos,
        )

        self.draw_button(
            self.controller_rect,
            "SWITCH CTRL",
            mouse_pos,
        )

        self.draw_button(
            self.cycle_rect,
            "NEXT CYCLE",
            mouse_pos,
        )

        if self.ui.mode == "driving":
            cycle = self.cycle_paths[self.ui.cycle_index][0]
            self.draw_text(
                cycle,
                683,
                129,
                ACCENT,
                self.font_small,
            )

        self.draw_button(
            self.motion_rect,
            "START: MOTION" if self.ui.start_at_first_motion else "START: 0s",
            mouse_pos,
        )

        self.draw_button(
            self.ambient_down_rect,
            "-",
            mouse_pos,
        )

        self.draw_button(
            self.ambient_up_rect,
            "+",
            mouse_pos,
        )

        self.draw_text(
            f"Amb: {self.ui.ambient_c:.0f} °C",
            982,
            129,
            MUTED,
            self.font_small,
        )

        self.draw_button(
            self.speed_down_rect,
            "-",
            mouse_pos,
        )

        self.draw_button(
            self.speed_up_rect,
            "+",
            mouse_pos,
        )

        self.draw_text(
            f"Sim: {self.ui.speed_multiplier:.1f}x",
            1184,
            129,
            MUTED,
            self.font_small,
        )

    # -----------------------------------------------------------------
    # BATTERY
    # -----------------------------------------------------------------

    def draw_battery(
        self,
        rect: pygame.Rect,
        soc: float,
        temp: float,
    ):

        pygame.draw.rect(
            self.screen,
            (60, 70, 90),
            rect,
            3,
            border_radius=14,
        )

        fill = np.clip(
            soc / 100.0,
            0.0,
            1.0,
        )

        inner = pygame.Rect(
            rect.x + 8,
            rect.y + 8,
            int(
                (rect.w - 16)
                * fill
            ),
            rect.h - 16,
        )

        fill_color = (
            GOOD
            if temp < 40
            else WARN
            if temp < 45
            else BAD
        )

        pygame.draw.rect(
            self.screen,
            fill_color,
            inner,
            border_radius=10,
        )

        pygame.draw.rect(
            self.screen,
            (75, 85, 105),
            (
                rect.right,
                rect.centery - 12,
                14,
                24,
            ),
            border_radius=5,
        )

        self.draw_text(
            f"{soc:.1f}%",
            rect.x
            + rect.w // 2
            - 28,
            rect.y
            + rect.h // 2
            - 12,
            WHITE,
            self.font,
        )

    # -----------------------------------------------------------------
    # STAT CARD
    # -----------------------------------------------------------------

    def draw_stat_card(
        self,
        x: int,
        y: int,
        w: int,
        label: str,
        value: str,
        color=TEXT,
    ):

        rect = pygame.Rect(
            x,
            y,
            w,
            82,
        )

        self.rounded_panel(
            rect,
            PANEL2,
            14,
        )

        self.draw_text(
            label,
            x + 14,
            y + 12,
            MUTED,
            self.font_small,
        )

        self.draw_text(
            value,
            x + 14,
            y + 39,
            color,
            self.font,
        )

    # -----------------------------------------------------------------
    # CHART
    # -----------------------------------------------------------------

    def draw_chart(
        self,
        rect: pygame.Rect,
        traces,
        title: str,
    ):

        self.rounded_panel(
            rect,
            PANEL,
            16,
        )

        self.draw_text(
            title,
            rect.x + 14,
            rect.y + 10,
            TEXT,
            self.font_small,
        )

        inner = pygame.Rect(
            rect.x + 14,
            rect.y + 34,
            rect.w - 28,
            rect.h - 48,
        )

        pygame.draw.rect(
            self.screen,
            (18, 24, 35),
            inner,
            border_radius=10,
        )

        for index, (
            trace,
            label,
            color,
            lo,
            hi,
        ) in enumerate(
            traces
        ):

            if len(trace.y) < 2:
                continue

            points = []

            span = max(
                1.0,
                hi - lo,
            )

            for i, value in enumerate(
                trace.y
            ):

                px = (
                    inner.x
                    + int(
                        i
                        / max(
                            1,
                            len(trace.y) - 1,
                        )
                        * inner.w
                    )
                )

                norm = np.clip(
                    (
                        value - lo
                    )
                    / span,
                    0.0,
                    1.0,
                )

                py = (
                    inner.bottom
                    - int(
                        norm
                        * (inner.h - 12)
                    )
                    - 6
                )

                points.append(
                    (px, py)
                )

            if len(points) > 1:

                pygame.draw.lines(
                    self.screen,
                    color,
                    False,
                    points,
                    2,
                )

            self.draw_text(
                label,
                inner.right - 125,
                inner.y + 8
                + index * 18,
                color,
                self.font_small,
            )

    # -----------------------------------------------------------------
    # VEHICLE
    # -----------------------------------------------------------------

    def draw_vehicle(
        self,
        x: int,
        y: int,
        braking: bool = False,
    ):

        pygame.draw.rect(
            self.screen,
            (45, 120, 210),
            pygame.Rect(
                x,
                y + 20,
                180,
                55,
            ),
            border_radius=18,
        )

        pygame.draw.polygon(
            self.screen,
            (75, 155, 230),
            [
                (x + 35, y + 20),
                (x + 70, y - 10),
                (x + 125, y - 10),
                (x + 155, y + 20),
            ],
        )

        pygame.draw.circle(
            self.screen,
            (28, 30, 38),
            (x + 45, y + 80),
            18,
        )

        pygame.draw.circle(
            self.screen,
            (28, 30, 38),
            (x + 140, y + 80),
            18,
        )

        if braking:

            pygame.draw.line(
                self.screen,
                BAD,
                (x - 18, y + 45),
                (x - 60, y + 45),
                5,
            )

            pygame.draw.line(
                self.screen,
                BAD,
                (x - 18, y + 60),
                (x - 60, y + 60),
                5,
            )

    # -----------------------------------------------------------------
    # POWER FLOW
    # -----------------------------------------------------------------

    def draw_power_flow(
        self,
        x1: int,
        y: int,
        x2: int,
        label: str,
        color,
        reverse=False,
    ):

        if reverse:

            x1, x2 = (
                x2,
                x1,
            )

        pygame.draw.line(
            self.screen,
            color,
            (x1, y),
            (x2, y),
            5,
        )

        head = (
            12
            if x2 > x1
            else -12
        )

        pygame.draw.polygon(
            self.screen,
            color,
            [
                (x2, y),
                (
                    x2 - head,
                    y - 10,
                ),
                (
                    x2 - head,
                    y + 10,
                ),
            ],
        )

        self.draw_text(
            label,
            min(x1, x2) + 25,
            y - 30,
            color,
            self.font_small,
        )

    # -----------------------------------------------------------------
    # MAIN DRAW
    # -----------------------------------------------------------------

    def draw(self):

        self.screen.fill(BG)

        self.draw_header()
        self.draw_controls()

        if self.env is None:

            self.rounded_panel(
                pygame.Rect(
                    40,
                    190,
                    WIDTH - 80,
                    620,
                ),
                PANEL,
                18,
            )

            self.draw_text(
                "Visualizer could not initialize.",
                80,
                240,
                BAD,
                self.font_big,
            )

            self.draw_text(
                self.message,
                80,
                300,
                MUTED,
                self.font_small,
            )

            return

        metrics = (
            self._current_metrics()
        )

        anim_rect = pygame.Rect(
            40,
            185,
            820,
            365,
        )

        self.rounded_panel(
            anim_rect,
            PANEL,
            18,
        )

        if self.ui.mode == "charging":

            self.draw_text(
                "FAST CHARGING",
                64,
                208,
                ACCENT,
                self.font,
            )

            self.draw_text(
                "FAST CHARGER",
                80,
                275,
                TEXT,
                self.font_small,
            )

            self.draw_power_flow(
                170,
                335,
                360,
                (
                    f"{metrics.get('Applied Current', 0):.1f}"
                    " A"
                ),
                (
                    GOOD
                    if metrics.get(
                        "Applied Current",
                        0,
                    ) > 0
                    else MUTED
                ),
            )

            self.draw_battery(
                pygame.Rect(
                    410,
                    275,
                    300,
                    110,
                ),
                metrics.get(
                    "SOC",
                    0,
                ),
                metrics.get(
                    "Temperature",
                    self.ui.ambient_c,
                ),
            )

            self.draw_text(
                (
                    f"{metrics.get('Voltage', 0):.1f}"
                    " V"
                ),
                515,
                410,
                TEXT,
                self.font_small,
            )

            self.draw_text(
                (
                    f"T = "
                    f"{metrics.get('Temperature', self.ui.ambient_c):.2f}"
                    " °C"
                ),
                515,
                438,
                (
                    WARN
                    if metrics.get(
                        "Temperature",
                        self.ui.ambient_c,
                    ) >= 40
                    else GOOD
                ),
                self.font_small,
            )

            self.draw_text(
                (
                    f"Ambient = "
                    f"{metrics.get('Ambient', self.ui.ambient_c):.1f}"
                    " °C"
                ),
                64,
                500,
                MUTED,
                self.font_small,
            )

            self.draw_text(
                "Click PLAY or STEP to advance the real battery simulation",
                64,
                520,
                MUTED,
                self.font_small,
            )

        else:

            speed = metrics.get(
                "Speed",
                0.0,
            )

            braking = False

            drive_cycle = getattr(
                self.env,
                "_drive_cycle",
                None,
            )

            if drive_cycle is not None:

                try:

                    braking = self._get_acceleration() < -0.05

                except Exception:
                    braking = False

            progress = 0.0

            if drive_cycle is not None:

                total_time = None

                for name in (
                    "total_time",
                    "duration",
                ):

                    try:

                        attr = getattr(
                            drive_cycle,
                            name,
                        )

                        value = (
                            attr()
                            if callable(attr)
                            else attr
                        )

                        total_time = safe_float(
                            value,
                            0.0,
                        )

                        if total_time > 0:
                            break

                    except Exception:
                        continue

                if (
                    total_time
                    and total_time > 0
                ):

                    progress = np.clip(
                        self.sim_time
                        / total_time,
                        0.0,
                        1.0,
                    )

            vehicle_x = (
                100
                + int(
                    progress * 650
                )
            )

            bob = int(
                math.sin(
                    self.animation_phase
                )
                * 4
            )

            self.draw_text(
                (
                    "DRIVE CYCLE: "
                    f"{self.cycle_paths[self.ui.cycle_index][0]}"
                ),
                64,
                208,
                ACCENT,
                self.font,
            )

            pygame.draw.line(
                self.screen,
                (80, 90, 110),
                (70, 400),
                (810, 400),
                6,
            )

            self.draw_vehicle(
                vehicle_x,
                300 + bob,
                braking=braking,
            )

            self.draw_text(
                f"{speed:.1f} km/h",
                vehicle_x + 25,
                410,
                TEXT,
                self.font,
            )

            battery_power = (
                metrics.get(
                    "Battery Power",
                    0.0,
                )
            )

            regen = (
                metrics.get(
                    "Regen",
                    0.0,
                )
            )

            is_regen = (
                regen > 0.001
                or battery_power > 0.001
            )

            self.draw_power_flow(
                390,
                330,
                690,
                (
                    f"{abs(battery_power):.1f}"
                    " kW"
                ),
                (
                    GOOD
                    if is_regen
                    else BAD
                ),
                reverse=is_regen,
            )

            self.draw_battery(
                pygame.Rect(
                    470,
                    450,
                    260,
                    85,
                ),
                metrics.get(
                    "SOC",
                    0,
                ),
                metrics.get(
                    "Temperature",
                    self.ui.ambient_c,
                ),
            )

            self.draw_text(
                (
                    f"Regen: "
                    f"{regen:.1f} kW"
                ),
                64,
                470,
                GOOD,
                self.font_small,
            )

            self.draw_text(
                (
                    f"Deficit: "
                    f"{metrics.get('Deficit Wh', 0):.2f}"
                    " Wh"
                ),
                64,
                505,
                (
                    WARN
                    if metrics.get(
                        "Deficit Wh",
                        0,
                    ) > 0
                    else GOOD
                ),
                self.font_small,
            )

        right = pygame.Rect(
            885,
            185,
            WIDTH - 925,
            365,
        )

        self.rounded_panel(
            right,
            PANEL,
            18,
        )

        self.draw_text(
            "LIVE STATE",
            right.x + 22,
            right.y + 20,
            TEXT,
            self.font,
        )

        if self.ui.mode == "charging":

            self.draw_stat_card(
                right.x + 20,
                right.y + 65,
                165,
                "SOC",
                f"{metrics.get('SOC', 0):.1f}%",
                GOOD,
            )

            self.draw_stat_card(
                right.x + 200,
                right.y + 65,
                165,
                "Temperature",
                (
                    f"{metrics.get('Temperature', 0):.2f}"
                    " °C"
                ),
                (
                    WARN
                    if metrics.get(
                        "Temperature",
                        0,
                    ) >= 40
                    else GOOD
                ),
            )

            self.draw_stat_card(
                right.x + 20,
                right.y + 160,
                165,
                "Voltage",
                (
                    f"{metrics.get('Voltage', 0):.1f}"
                    " V"
                ),
            )

            self.draw_stat_card(
                right.x + 200,
                right.y + 160,
                165,
                "Current",
                (
                    f"{metrics.get('Applied Current', 0):.1f}"
                    " A"
                ),
            )

            self.draw_stat_card(
                right.x + 20,
                right.y + 255,
                165,
                "Target",
                (
                    f"{metrics.get('Target', 95):.0f}%"
                ),
            )

            safety = str(
                metrics.get(
                    "Safety",
                    "none",
                )
            )

            self.draw_stat_card(
                right.x + 200,
                right.y + 255,
                165,
                "Safety",
                safety.upper(),
                (
                    BAD
                    if safety.lower()
                    != "none"
                    else GOOD
                ),
            )

        else:

            self.draw_stat_card(
                right.x + 20,
                right.y + 65,
                165,
                "SOC",
                (
                    f"{metrics.get('SOC', 0):.1f}%"
                ),
                GOOD,
            )

            self.draw_stat_card(
                right.x + 200,
                right.y + 65,
                165,
                "Speed",
                (
                    f"{metrics.get('Speed', 0):.1f}"
                    " km/h"
                ),
                ACCENT,
            )

            self.draw_stat_card(
                right.x + 20,
                right.y + 160,
                165,
                "Battery Power",
                (
                    f"{metrics.get('Battery Power', 0):.1f}"
                    " kW"
                ),
            )

            self.draw_stat_card(
                right.x + 200,
                right.y + 160,
                165,
                "Regen",
                (
                    f"{metrics.get('Regen', 0):.1f}"
                    " kW"
                ),
                GOOD,
            )

            self.draw_stat_card(
                right.x + 20,
                right.y + 255,
                165,
                "Deficit",
                (
                    f"{metrics.get('Deficit Wh', 0):.2f}"
                    " Wh"
                ),
                (
                    WARN
                    if metrics.get(
                        "Deficit Wh",
                        0,
                    ) > 0
                    else GOOD
                ),
            )

            safety = str(
                metrics.get(
                    "Safety",
                    "none",
                )
            )

            self.draw_stat_card(
                right.x + 200,
                right.y + 255,
                165,
                "Safety",
                safety.upper(),
                (
                    BAD
                    if safety.lower()
                    != "none"
                    else GOOD
                ),
            )

        chart1 = pygame.Rect(
            40,
            570,
            415,
            270,
        )

        chart2 = pygame.Rect(
            470,
            570,
            415,
            270,
        )

        chart3 = pygame.Rect(
            900,
            570,
            500,
            270,
        )

        self.draw_chart(
            chart1,
            [
                (
                    self.trace_soc,
                    "SOC %",
                    GOOD,
                    0,
                    100,
                )
            ],
            "SOC",
        )

        self.draw_chart(
            chart2,
            [
                (
                    self.trace_temp,
                    "Temp °C",
                    WARN,
                    15,
                    55,
                )
            ],
            "Battery Temperature",
        )

        if self.ui.mode == "charging":

            self.draw_chart(
                chart3,
                [
                    (
                        self.trace_power,
                        "Power kW",
                        ACCENT,
                        0,
                        80,
                    ),
                    (
                        self.trace_action,
                        "Action",
                        GOOD,
                        -1,
                        1,
                    ),
                ],
                "Power / PPO Action",
            )

        else:

            self.draw_chart(
                chart3,
                [
                    (
                        self.trace_speed,
                        "Speed km/h",
                        ACCENT,
                        0,
                        120,
                    ),
                    (
                        self.trace_action,
                        "Action",
                        GOOD,
                        -1,
                        1,
                    ),
                ],
                "Speed / PPO Action",
            )

        if (
            self.message
            and time.perf_counter()
            < self.message_until
        ):

            message_rect = pygame.Rect(
                400,
                40,
                620,
                48,
            )

            self.rounded_panel(
                message_rect,
                PANEL2,
                12,
            )

            self.draw_text(
                self.message,
                420,
                54,
                WHITE,
                self.font_small,
            )

    # -----------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------

    def run(self):

        while True:

            for event in pygame.event.get():
                self.handle_event(event)

            self.update()
            self.draw()

            pygame.display.flip()

            self.clock.tick(FPS)


def main():

    simulator = InteractiveSimulator()
    simulator.run()


if __name__ == "__main__":
    main()