"""
Interactive 2D EV RL-BMS Simulator for the RL-BMS-Driving Project.

Authoritative simulator layer providing:
- Real drive-cycle trace following
- Real PPO driving EMS and baseline controllers
- Bidirectional BMS safety layer integration
- Authoritative 9-state thermal state machine with internal hysteresis
- Demo Safety Stop Controller with controlled deceleration and pure ECM passive cooling
- Battery-life-oriented driver guidance & speed recommendations
- Research / Demo data and benchmark isolation
- Real-time power flows, thermal scale, and dynamic current ceiling gauges

Controls:
---------
TAB           Switch Charging / Driving mode
M             Toggle Simulation Mode (Research <-> Demo)
SPACE         Play / Pause
RIGHT         Advance exactly one simulation step
R             Reset
B             Toggle PPO / Baseline controller
1-4           UDDS / HWFET / US06 / WLTP cycles
V             Toggle Speed Recommendation
S             Stop Vehicle (Demo safety stop; logged in Research)
U             Resume Vehicle (available only in SAFE_TO_RESUME)
+ / -         Change simulation speed
UP / DOWN     Change ambient temperature
ESC / Q       Quit
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
from stable_baselines3 import PPO

# ---------------------------------------------------------------------
# PROJECT ROOT & IMPORTS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.cc import MaxCurrentController
from baselines.rule_based_ems import RuleBasedEMS
from environment.env_factory import make_env
from training.train_drive_ems import make_drive_ems_env
from app.thermal_state_machine import (
    ThermalState,
    calculate_recommended_speed,
    determine_state,
    get_driver_guidance,
    load_thermal_config,
)
from app.safety_stop_controller import DemoSafetyStopController
from app.logger import SimulatorLogger

# ---------------------------------------------------------------------
# UI COLOR CONSTANTS & THEME
# Professional, subtle, engineering-style color system (§1 - §6)
# ---------------------------------------------------------------------

WIDTH = 1440
HEIGHT = 900
FPS = 60

# Base Engineering Palette (§1)
UI_BACKGROUND         = (13, 17, 23)        # Very dark neutral blue/charcoal (#0D1117)
UI_PANEL              = (22, 27, 34)        # Main dark surface (#161B22)
UI_PANEL_ALT          = (30, 36, 46)        # Secondary panel / plot background (#1E242E)
UI_PANEL_BUTTON       = (33, 40, 52)        # Neutral button fill
UI_PANEL_BUTTON_HOVER = (45, 54, 70)        # Button hover surface
UI_BORDER             = (48, 54, 61)        # Muted blue-gray border (#30363D)
UI_BORDER_SUBTLE      = (36, 42, 50)        # Low-contrast inner border
UI_BORDER_FOCUS       = (88, 166, 255)      # Active focus ring

# Text Hierarchy (§3, §4)
UI_TEXT_PRIMARY       = (240, 246, 252)     # High contrast near-white for critical values (Level 1)
UI_TEXT_SECONDARY     = (139, 148, 158)     # Medium contrast light gray for supporting labels (Level 2)
UI_TEXT_MUTED         = (110, 118, 129)     # Low contrast darker gray for metadata (Level 3)
UI_TEXT_DISABLED      = (75, 85, 99)        # Disabled controls

# Semantic Accent Colors (Restrained, ~10-20% of UI)
UI_ACCENT             = (56, 139, 253)      # Technical engineering blue (#388BFD)
UI_NORMAL             = (46, 160, 67)       # Subtle automotive green (#2EA043)
UI_WARNING            = (210, 153, 34)      # Muted amber/yellow (#D29922)
UI_WARNING_STRONG     = (230, 120, 30)      # Strong amber/orange for active derating
UI_CRITICAL           = (248, 81, 73)       # Controlled red for critical state (#F85149)
UI_COOLING            = (88, 166, 255)      # Restrained cyan/blue (#58A6FF)
UI_SAFE_RESUME        = (63, 185, 80)       # Subtle green/blue (#3FB950)
UI_PURPLE             = (163, 113, 247)     # Restrained purple for action trace (#A371F7)

# Backward-compatibility aliases
BG            = UI_BACKGROUND
PANEL         = UI_PANEL
PANEL2        = UI_PANEL_ALT
PANEL3        = UI_PANEL_BUTTON
PANEL_BORDER  = UI_BORDER
PANEL_HOVER   = UI_PANEL_BUTTON_HOVER

TEXT          = UI_TEXT_PRIMARY
MUTED         = UI_TEXT_SECONDARY
ACCENT        = UI_ACCENT
GOOD          = UI_NORMAL
WARN          = UI_WARNING
BAD           = UI_CRITICAL
CYAN          = UI_COOLING
WHITE         = UI_TEXT_PRIMARY

# Thermal region colors (§5)
COLOR_OPTIMAL  = UI_NORMAL
COLOR_ELEVATED = UI_WARNING
COLOR_DERATING = UI_WARNING_STRONG
COLOR_CRITICAL = UI_CRITICAL
COLOR_COOLING  = UI_COOLING
COLOR_RESUME   = UI_SAFE_RESUME

# Car body colors (restrained, non-neon dark fills with distinct outlines)
CAR_BODY_OPTIMAL  = (20, 45, 30)
CAR_BODY_ELEVATED = (50, 42, 15)
CAR_BODY_DERATING = (55, 32, 12)
CAR_BODY_CRITICAL = (60, 20, 20)
CAR_BODY_COOLING  = (15, 38, 58)
CAR_BODY_RESUME   = (18, 48, 28)

# Thermal states that constitute a Demo-Mode safety stop. Entering any of
# these (Demo Mode only) freezes the real benchmark environment (§14).
DEMO_STOP_STATES = {
    ThermalState.STOP_REQUESTED,
    ThermalState.DECELERATING,
    ThermalState.STOPPED,
    ThermalState.COOLING,
    ThermalState.SAFE_TO_RESUME,
}


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    """Parse a float safely, falling back to default if non-finite or malformed."""
    try:
        val = float(value)
        if math.isfinite(val):
            return val
    except (TypeError, ValueError):
        pass
    return default


def first_float(mapping: Any, keys: List[str], default: float = 0.0) -> float:
    """Return the first finite numeric value found for keys in mapping."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping:
            try:
                val = float(mapping[key])
                if math.isfinite(val):
                    return val
            except (TypeError, ValueError):
                continue
    return default


# ---------------------------------------------------------------------
# DATA TRACE
# ---------------------------------------------------------------------

@dataclass
class Trace:
    max_points: int = 360
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    _step: int = 0

    def add(self, value: float, x_val: Optional[float] = None) -> None:
        val_x = float(self._step if x_val is None else x_val)
        self.x.append(val_x)
        self.y.append(safe_float(value))
        self._step += 1
        if len(self.x) > self.max_points:
            self.x.pop(0)
            self.y.pop(0)

    def clear(self) -> None:
        self.x.clear()
        self.y.clear()
        self._step = 0


# ---------------------------------------------------------------------
# UI STATE
# ---------------------------------------------------------------------

@dataclass
class UIState:
    mode: str = "charging"                 # "charging" | "driving"
    sim_mode: str = "research"             # "research" | "demo"
    playing: bool = False
    controller: str = "ppo"                # "ppo" | "baseline"
    speed_multiplier: float = 2.0
    ambient_c: float = 25.0
    initial_soc: float = 0.50
    cycle_index: int = 3                   # default WLTP
    start_at_first_motion: bool = False
    show_speed_recommendation: bool = True
    thermal_state: ThermalState = ThermalState.OPTIMAL


# ---------------------------------------------------------------------
# MAIN SIMULATOR APPLICATION
# ---------------------------------------------------------------------

class InteractiveSimulator:
    """Production-grade interactive EV RL-BMS simulator and visualization layer."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("RL-BMS-Driving — Interactive EV Simulator")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        # Typography System with clear hierarchy (§2)
        self.font_title = pygame.font.SysFont("Segoe UI", 15, bold=True)
        self.font_big   = pygame.font.SysFont("Segoe UI", 28, bold=True)
        self.font_mid   = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font_val   = pygame.font.SysFont("Segoe UI", 15, bold=True)
        self.font       = pygame.font.SysFont("Segoe UI", 14)
        self.font_unit  = pygame.font.SysFont("Segoe UI", 11)
        self.font_small = pygame.font.SysFont("Segoe UI", 12)
        self.font_btn   = pygame.font.SysFont("Segoe UI", 12, bold=True)
        self.font_hud   = pygame.font.SysFont("Segoe UI", 13, bold=True)
        self.font_badge = pygame.font.SysFont("Segoe UI", 11, bold=True)
        self.font_mono  = pygame.font.SysFont("Consolas", 12)

        self.ui = UIState()
        self.thermal_config = load_thermal_config()
        self.safety_stop_ctrl = DemoSafetyStopController(self.thermal_config)
        self.logger = SimulatorLogger(ROOT, mode=self.ui.sim_mode)

        # Config paths
        self.config_dir_charging = ROOT / "configs" / "final_charging"
        self.config_dir_driving = ROOT / "configs" / "final_driving"
        self.final_models = ROOT / "final_models"

        # Drive cycles & models
        self.cycle_paths = self._resolve_drive_cycles()
        self.charging_model_path = self._find_model_recursive(
            self.final_models / "charging_A1_50k_seed7", "trained_model.zip"
        )
        self.driving_model_path = self._find_model_recursive(
            self.final_models / "driving_B3_100k_seed7", "ppo_driving_100000_steps.zip"
        )

        # Runtime state
        self.env = None
        self.model = None
        self.baseline = None
        self.obs = None
        self.info: Dict[str, Any] = {}
        self.done: bool = False
        self.sim_time: float = 0.0
        self.cumulative_distance_m: float = 0.0

        self._display_speed_mps = 0.0
        self._display_accel_mps2 = 0.0
        self._display_time_s = 0.0
        self._last_thermal_state = ThermalState.OPTIMAL

        # Demo-Mode benchmark freeze (§14). When a demo safety stop begins,
        # the real EVEnergyEnv is frozen: the drive cycle, benchmark power
        # demand, power deficit and tracking error stop evolving while the
        # Demo Safety Stop Controller decelerates the vehicle and the battery
        # cools via the authoritative ECM passive-cooling model. Research Mode
        # is NEVER frozen.
        self.benchmark_frozen: bool = False
        self.benchmark_snapshot: Dict[str, Any] = {}

        # Animation & Timing
        self.last_step_wall = time.perf_counter()
        self.step_accumulator = 0.0
        self.animation_phase = 0.0
        self.vehicle_x_pos = 0.0
        self.wheel_rotation_deg = 0.0
        self.MAX_STEPS_PER_FRAME = 30

        # Traces for live charts
        self.trace_soc = Trace()
        self.trace_temp = Trace()
        self.trace_power = Trace()
        self.trace_speed = Trace()
        self.trace_rec_speed = Trace()
        self.trace_action = Trace()
        self.trace_ceiling = Trace()
        self.trace_deficit = Trace()

        # Messages
        self.message = ""
        self.message_until = 0.0

        # UI Button Rectangles (Interactive controls)
        self._init_buttons()
        self._load_mode()

    def _init_buttons(self) -> None:
        """Initialize button bounding boxes aligned to a unified 24px/8px grid (§3, §11, §12)."""
        y = 94
        h = 34
        self.play_rect       = pygame.Rect(24,  y, 76,  h)
        self.step_rect       = pygame.Rect(108, y, 58,  h)
        self.reset_rect      = pygame.Rect(174, y, 60,  h)
        self.mode_rect       = pygame.Rect(242, y, 92,  h)
        self.sim_mode_rect   = pygame.Rect(342, y, 102, h)
        self.controller_rect = pygame.Rect(452, y, 92,  h)
        self.cycle_rect      = pygame.Rect(552, y, 82,  h)
        self.rec_speed_btn   = pygame.Rect(642, y, 88,  h)
        self.stop_veh_btn    = pygame.Rect(738, y, 76,  h)
        self.resume_btn      = pygame.Rect(822, y, 84,  h)

        # Ambient stepper capsule (X = 920..1152)
        self.ambient_down_rect = pygame.Rect(1074, y + 3, 32, 28)
        self.ambient_up_rect   = pygame.Rect(1110, y + 3, 32, 28)

        # Speed stepper capsule (X = 1176..1416)
        self.speed_down_rect   = pygame.Rect(1334, y + 3, 32, 28)
        self.speed_up_rect     = pygame.Rect(1370, y + 3, 32, 28)

        # Legacy compatibility rect
        self.motion_rect = pygame.Rect(1370, y, 46, h)

    def _resolve_drive_cycles(self) -> List[Tuple[str, Path]]:
        names = [
            ("UDDS", "epa_udds"),
            ("HWFET", "epa_hwfet"),
            ("US06", "epa_us06"),
            ("WLTP", "wltp_class3b"),
        ]
        result = []
        for label, folder in names:
            candidates = [
                ROOT / "data" / "drive_cycles" / "standard" / folder / "cycle.csv",
                ROOT / "data" / "drive_cycles" / f"{folder}.csv",
                ROOT / "data" / "drive_cycles" / "standard" / f"{folder}.csv",
            ]
            selected = next((p for p in candidates if p.exists()), candidates[0])
            result.append((label, selected))
        return result

    @staticmethod
    def _find_model_recursive(folder: Path, filename: str) -> Optional[Path]:
        if not folder.exists():
            return None
        direct = folder / filename
        if direct.exists():
            return direct
        matches = list(folder.rglob(filename))
        return matches[0] if matches else None

    def _set_message(self, text: str, seconds: float = 4.0) -> None:
        self.message = str(text)
        self.message_until = time.perf_counter() + seconds

    def _walk_envs(self):
        env = self.env
        seen = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            yield env
            next_env = None
            for attr in ("env", "unwrapped"):
                cand = getattr(env, attr, None)
                if cand is not None and cand is not env:
                    next_env = cand
                    break
            if next_env is None:
                break
            env = next_env

    def _get_env_attr(self, name: str, default: Any = None) -> Any:
        for env in self._walk_envs():
            try:
                val = getattr(env, name)
                if val is not None:
                    return val
            except AttributeError:
                continue
        return default

    def _get_drive_cycle(self):
        for env in self._walk_envs():
            for name in ("_drive_cycle", "drive_cycle"):
                cycle = getattr(env, name, None)
                if cycle is not None:
                    return cycle
        return None

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
            err = f"{type(exc).__name__}: {exc}"
            self._set_message(f"Initialization error: {err}", 15.0)
            print(f"[SIMULATOR INIT ERROR] {err}", file=sys.stderr)

    def _load_charging_mode(self) -> None:
        self.env = make_env(mode="eval", config_dir=str(self.config_dir_charging))
        if self.ui.controller == "ppo" and self.charging_model_path is not None:
            self.model = PPO.load(str(self.charging_model_path), device="cpu")
            self.baseline = None
        else:
            self.model = None
            self.baseline = MaxCurrentController(self.env.battery_config)

    def _load_driving_mode(self) -> None:
        cycle_name, cycle_path = self.cycle_paths[self.ui.cycle_index]
        if not cycle_path.exists():
            raise FileNotFoundError(f"{cycle_name} drive cycle not found: {cycle_path}")

        self.env = make_drive_ems_env(
            drive_cycle_path=str(cycle_path),
            mode="eval",
            config_dir=str(self.config_dir_driving),
        )
        if self.ui.controller == "ppo" and self.driving_model_path is not None:
            self.model = PPO.load(str(self.driving_model_path), device="cpu")
            self.baseline = None
        else:
            self.model = None
            self.baseline = RuleBasedEMS()

    def _reset_env(self) -> None:
        if self.env is None:
            return

        self.trace_soc.clear()
        self.trace_temp.clear()
        self.trace_power.clear()
        self.trace_speed.clear()
        self.trace_rec_speed.clear()
        self.trace_action.clear()
        self.trace_ceiling.clear()
        self.trace_deficit.clear()

        self.done = False
        self.sim_time = 0.0
        self.cumulative_distance_m = 0.0
        self._display_speed_mps = 0.0
        self._display_accel_mps2 = 0.0
        self._display_time_s = 0.0
        self.safety_stop_ctrl.reset()
        self._unfreeze_benchmark()
        self.logger = SimulatorLogger(ROOT, mode=self.ui.sim_mode)

        options = {
            "initial_soc": float(np.clip(self.ui.initial_soc, 0.05, 0.95)),
            "ambient_temp_c": float(self.ui.ambient_c),
        }
        try:
            self.obs, self.info = self.env.reset(seed=42, options=options)
        except TypeError:
            self.obs, self.info = self.env.reset(seed=42)

        self.last_step_wall = time.perf_counter()
        self.step_accumulator = 0.0
        self._sync_display_cycle_sample()

        if self.ui.mode == "driving" and self.ui.start_at_first_motion:
            self._fast_forward_to_first_motion()

        self.ui.thermal_state = determine_state(
            ThermalState.OPTIMAL,
            self._get_temperature(),
            self._get_speed(),
            self.thermal_config,
            mode=self.ui.sim_mode,
        )
        self._last_thermal_state = self.ui.thermal_state

        self._append_state_trace(info=self.info, action=0.0, reward=0.0)

    def _sync_display_cycle_sample(self) -> None:
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
        cycle = self._get_drive_cycle()
        if cycle is None:
            return
        max_steps = min(max(1, len(cycle)), 120)
        for _ in range(max_steps):
            if safe_float(cycle.current_speed()) > 0.0 or self.done:
                break
            self._step_once()
        if not self.done and safe_float(cycle.current_speed()) > 0.0:
            self._step_once()

    def toggle_mode(self) -> None:
        self.ui.mode = "driving" if self.ui.mode == "charging" else "charging"
        self.ui.controller = "ppo"
        self._load_mode()
        self._set_message(f"Switched to {self.ui.mode.upper()} mode")

    def toggle_sim_mode(self) -> None:
        """Toggle Research Benchmark vs Interactive Demo Mode."""
        self.ui.sim_mode = "demo" if self.ui.sim_mode == "research" else "research"
        self.safety_stop_ctrl.reset()
        self._unfreeze_benchmark()
        self.logger = SimulatorLogger(ROOT, mode=self.ui.sim_mode)
        self._set_message(f"Simulation Mode: {self.ui.sim_mode.upper()} MODE")

    def toggle_controller(self) -> None:
        self.ui.controller = "baseline" if self.ui.controller == "ppo" else "ppo"
        self._load_mode()
        label = "RULE-BASED" if (self.ui.mode == "driving" and self.ui.controller == "baseline") else (
            "MAX CURRENT" if self.ui.controller == "baseline" else "PPO"
        )
        self._set_message(f"Controller: {label}")

    def set_cycle(self, index: int) -> None:
        if self.ui.mode != "driving":
            return
        self.ui.cycle_index = index % len(self.cycle_paths)
        self._load_mode()
        self._set_message(f"Drive cycle: {self.cycle_paths[self.ui.cycle_index][0]}")

    def toggle_speed_recommendation(self) -> None:
        self.ui.show_speed_recommendation = not self.ui.show_speed_recommendation
        self._set_message(
            f"Speed Recommendation: {'VISIBLE' if self.ui.show_speed_recommendation else 'HIDDEN'}"
        )

    def trigger_stop_vehicle(self) -> None:
        """Handle Stop Vehicle button click."""
        speed_mps = self._display_speed_mps
        temp_c = self._get_temperature()
        soc = self._get_soc()
        ceiling_a = self._get_current_ceiling()

        if self.ui.sim_mode == "demo":
            self.safety_stop_ctrl.trigger_stop(speed_mps, manual=True)
            self.ui.thermal_state = ThermalState.STOP_REQUESTED
            self._freeze_benchmark()
            self.logger.log_event(
                "manual_stop_intervention",
                self.sim_time, temp_c, soc, speed_mps * 3.6, ceiling_a, self.ui.thermal_state.value
            )
            self._set_message("DEMO SAFETY STOP TRIGGERED — Decelerating")
        else:
            # Research Mode: Log manual intervention and do not disrupt benchmark trajectory
            self.logger.log_event(
                "manual_stop_intervention",
                self.sim_time, temp_c, soc, speed_mps * 3.6, ceiling_a, self.ui.thermal_state.value,
                details={"status": "INTERVENED — NOT STANDARD BENCHMARK"}
            )
            self.ui.playing = False
            self._set_message("PAUSED (Research run marked INTERVENED)")

    def trigger_resume(self) -> None:
        """Handle manual safe resume button click."""
        temp_c = self._get_temperature()
        soc = self._get_soc()
        ceiling_a = self._get_current_ceiling()

        if self.ui.sim_mode == "demo":
            success, next_st = self.safety_stop_ctrl.resume(temp_c, self.ui.thermal_state)
            if success:
                self.ui.thermal_state = next_st
                self._unfreeze_benchmark()
                self.logger.log_event(
                    "manual_resume",
                    self.sim_time, temp_c, soc, 0.0, ceiling_a, self.ui.thermal_state.value
                )
                self._set_message("BATTERY SAFE — RESUMED DRIVING")
            else:
                self._set_message("CANNOT RESUME: Temperature not below safe threshold (42.0°C)")
        else:
            self.logger.log_event(
                "manual_resume_intervention",
                self.sim_time, temp_c, soc, self._get_speed(), ceiling_a, self.ui.thermal_state.value
            )
            self.ui.playing = True
            self._set_message("RESEARCH RUN RESUMED")

    def _action(self) -> np.ndarray:
        if self.obs is None:
            return np.array([0.0], dtype=np.float32)

        if self.ui.controller == "ppo" and self.model is not None:
            action, _ = self.model.predict(self.obs, deterministic=True)
            return np.asarray(action, dtype=np.float32).reshape(-1)

        if self.ui.mode == "charging" and self.baseline is not None:
            current = safe_float(self.baseline.act(self.obs))
            i_max = safe_float(getattr(self.env, "i_max", 160.0), 160.0)
            action = 2.0 * current / max(i_max, 1e-6) - 1.0
            return np.array([np.clip(action, -1.0, 1.0)], dtype=np.float32)

        if self.ui.mode == "driving" and self.baseline is not None:
            return np.array([safe_float(self.baseline.act(self.obs))], dtype=np.float32)

        return np.array([0.0], dtype=np.float32)

    def _step_once(self) -> None:
        """Execute EXACTLY one environment step with synchronized thermal state progression.

        In Demo Mode, once a safety stop is requested the real benchmark
        environment is FROZEN (§14): the drive cycle, benchmark power demand,
        power deficit and tracking error stop evolving. The Demo Safety Stop
        Controller then decelerates the demo vehicle and the battery cools via
        the authoritative ECM passive-cooling model, entirely separately from
        the frozen benchmark. Research Mode is never frozen.
        """
        if self.env is None or self.obs is None or self.done:
            return

        dt = safe_float(self._get_env_attr("dt", 1.0), 1.0)

        # ── Demo Mode with a frozen benchmark: advance only the demo layer ──
        if self.ui.sim_mode == "demo" and self.benchmark_frozen:
            self._advance_demo_frozen(dt)
            return

        self._sync_display_cycle_sample()
        action = self._action()

        try:
            result = self.env.step(action)
        except Exception as exc:
            self.ui.playing = False
            err = f"{type(exc).__name__}: {exc}"
            self._set_message(f"Step error: {err}", 15.0)
            print(f"[SIMULATOR STEP ERROR] {err}", file=sys.stderr)
            return

        if len(result) == 5:
            self.obs, reward, terminated, truncated, info = result
            self.done = bool(terminated or truncated)
        elif len(result) == 4:
            self.obs, reward, done, info = result
            self.done = bool(done)
        else:
            raise RuntimeError(f"Unexpected env.step() result length: {len(result)}")

        self.info = info if isinstance(info, dict) else {}
        self.sim_time += dt

        # Update physical distance
        current_speed_mps = self._display_speed_mps
        self.cumulative_distance_m += current_speed_mps * dt

        # Update thermal state machine
        temp_c = self._get_temperature()
        raw_speed_kmh = current_speed_mps * 3.6

        should_freeze = False
        if self.ui.sim_mode == "demo":
            # Safety stop controller updates speed and state in demo mode
            demo_speed_mps, next_st, is_overriding = self.safety_stop_ctrl.step(
                dt_s=dt,
                reference_speed_mps=current_speed_mps,
                temperature_c=temp_c,
                current_thermal_state=self.ui.thermal_state,
            )
            self.ui.thermal_state = next_st
            if is_overriding:
                self._display_speed_mps = demo_speed_mps
            # A safety stop has begun -> freeze the last real benchmark state
            # (§14) so it no longer evolves while the demo vehicle decelerates
            # and the battery cools. Defer the actual freeze until AFTER this
            # step's real values are logged: the triggering step is still a
            # genuine benchmark step; the freeze takes effect from the next one.
            if not self.benchmark_frozen and next_st in DEMO_STOP_STATES:
                should_freeze = True
        else:
            # Research mode preserves reference speed strictly
            self.ui.thermal_state = determine_state(
                self.ui.thermal_state,
                temp_c,
                vehicle_speed_kmh=raw_speed_kmh,
                config=self.thermal_config,
                mode="research",
            )

        self._log_state_transition(temp_c)
        self._append_state_trace(info=self.info, action=safe_float(action[0]), reward=safe_float(reward))
        self._log_trajectory(temp_c)

        if should_freeze:
            self._freeze_benchmark()

        if self.done:
            self.ui.playing = False
            self.logger.save_session(self.thermal_config)
            self._set_message("Episode finished — Session saved")

    def _advance_demo_frozen(self, dt: float) -> None:
        """Advance ONLY the demo safety layer while the benchmark is frozen (§14).

        The real EVEnergyEnv is not stepped: reference speed, battery power,
        power deficit and tracking error remain fixed at the frozen snapshot.
        The battery cools via the authoritative ECM passive-cooling model at
        zero current, and the Demo Safety Stop Controller decelerates the demo
        vehicle and drives the thermal-recovery state machine.
        """
        # Authoritative passive cooling on the real ECM (never a UI decrement).
        self._advance_passive_cooling(dt)
        temp_c = self._get_temperature()

        demo_speed_mps, next_st, _ = self.safety_stop_ctrl.step(
            dt_s=dt,
            reference_speed_mps=0.0,   # benchmark frozen: no reference demand during a stop
            temperature_c=temp_c,
            current_thermal_state=self.ui.thermal_state,
        )
        self.ui.thermal_state = next_st
        self._display_speed_mps = demo_speed_mps
        self.sim_time += dt
        self.cumulative_distance_m += demo_speed_mps * dt

        self._log_state_transition(temp_c)
        self._append_state_trace(info=self.info, action=0.0, reward=0.0)
        self._log_trajectory(temp_c)

    def _advance_passive_cooling(self, dt: float) -> None:
        """Cool the real battery ECM at zero current (authoritative passive cooling).

        Uses environment.ecm_model.BatteryECM.step -- the same validated model
        used everywhere else -- so the demonstrated cooling is physically real,
        never a UI-only 'temperature -= 1' decrement (forbidden practice §51).

        After writing the cooled state back, reset _prev_battery_power_w to 0
        (the battery drew zero power during cooling) and refresh self.obs so the
        PPO policy receives a fresh observation at resume (Issues D & E)."""
        ecm = self._get_env_attr("ecm", None)
        state = self._get_env_attr("_state", None)
        if ecm is None or state is None:
            return
        ambient = safe_float(
            self._get_env_attr("_ambient_temp_c", self.ui.ambient_c), self.ui.ambient_c
        )
        try:
            new_state = ecm.step(state, current_a=0.0, ambient_temp_c=ambient)
        except Exception:
            return
        # Write the cooled state back onto the (frozen) benchmark env so all
        # authoritative state reads reflect the real cooling progression.
        for env in self._walk_envs():
            if getattr(env, "_state", None) is not None:
                env._state = new_state
                # Reset accumulated power history: battery drew 0 W during cooling
                if hasattr(env, "_prev_battery_power_w"):
                    env._prev_battery_power_w = 0.0
                break
        # Refresh the policy observation so PPO does not act on a stale snapshot
        if self.env is not None and hasattr(self.env, "get_observation"):
            try:
                self.obs = self.env.get_observation()
            except Exception:
                pass

    def _freeze_benchmark(self) -> None:
        """Capture the last real EVEnergyEnv benchmark state and freeze it (§14).

        Called when a Demo-Mode safety stop begins. After this, the benchmark
        environment is not stepped until resume: the snapshot below is the
        'LAST BENCHMARK STATE' shown in the UI and never changes while frozen.
        """
        # Ensure the demo stop controller is decelerating from the real speed
        # at the instant the stop began (so it does not snap straight to zero).
        if not self.safety_stop_ctrl.state.is_active:
            self.safety_stop_ctrl.trigger_stop(self._display_speed_mps, manual=False)

        self.benchmark_snapshot = {
            "reference_speed_kmh": self._get_reference_speed(),
            "battery_power_kw": self._get_power(),
            "power_deficit_w": self._get_power_deficit(),
            "soc_pct": self._get_soc() * 100.0,
            "temperature_c": self._get_temperature(),
            "ceiling_a": self._get_current_ceiling(),
            "applied_current_a": self._get_applied_current(),
            "requested_current_a": self._get_requested_current(),
            "voltage_v": self._get_voltage(),
            "regen_kw": self._get_regen(),
            "tracking_error_w": abs(self._get_power_deficit()),
            "sim_time_s": self.sim_time,
            "thermal_state": self.ui.thermal_state.value,
        }
        self.benchmark_frozen = True
        # Live metric reads now reflect the demo-safety / cooling layer (power
        # and deficit are zero while stopped; temperature cools). The frozen
        # benchmark numbers live only in benchmark_snapshot.
        self.info = {}
        self.logger.log_event(
            "benchmark_frozen",
            self.sim_time,
            self.benchmark_snapshot["temperature_c"],
            self.benchmark_snapshot["soc_pct"] / 100.0,
            self.benchmark_snapshot["reference_speed_kmh"],
            self.benchmark_snapshot["ceiling_a"],
            self.ui.thermal_state.value,
            details={"snapshot": self.benchmark_snapshot},
        )

    def _unfreeze_benchmark(self) -> None:
        """Clear the frozen benchmark snapshot and resume normal env stepping.

        Also resets _prev_battery_power_w on the underlying env (to avoid a
        stale power value from the last real step before the stop corrupting the
        first post-resume PPO observation) and refreshes self.obs (Issues D & E).
        """
        self.benchmark_frozen = False
        self.benchmark_snapshot = {}
        # Clear stale power history so resumed observation is coherent
        for env in self._walk_envs():
            if hasattr(env, "_prev_battery_power_w"):
                env._prev_battery_power_w = 0.0
                break
        # Re-build the PPO observation from current (cooled) battery state
        if self.env is not None and hasattr(self.env, "get_observation"):
            try:
                self.obs = self.env.get_observation()
            except Exception:
                pass

    def _log_state_transition(self, temp_c: float) -> None:
        """Log a discrete thermal-state transition event if the state changed."""
        if self.ui.thermal_state == self._last_thermal_state:
            return
        event_map = {
            ThermalState.ELEVATED_THERMAL: "thermal_stress_entered",
            ThermalState.DERATING_ACTIVE: "derating_started",
            ThermalState.CRITICAL: "critical_entered",
            ThermalState.STOP_REQUESTED: "stop_requested",
            ThermalState.DECELERATING: "deceleration_started",
            ThermalState.STOPPED: "vehicle_stopped",
            ThermalState.COOLING: "cooling_started",
            ThermalState.SAFE_TO_RESUME: "safe_to_resume",
        }
        evt_name = event_map.get(self.ui.thermal_state, "thermal_state_change")
        self.logger.log_event(
            evt_name,
            self.sim_time, temp_c, self._get_soc(), self._get_speed(),
            self._get_current_ceiling(), self.ui.thermal_state.value
        )
        self._last_thermal_state = self.ui.thermal_state

    def _log_trajectory(self, temp_c: float) -> None:
        """Append one synchronized timestep to the trajectory record."""
        self.logger.log_trajectory_step({
            "simulation_time": self.sim_time,
            "temperature_c": temp_c,
            "soc": self._get_soc(),
            "speed_kmh": self._get_speed(),
            "power_w": self._get_power() * 1000.0,
            "current_ceiling_a": self._get_current_ceiling(),
            "thermal_state": self.ui.thermal_state.value,
            "benchmark_frozen": self.benchmark_frozen,
        })

    # -----------------------------------------------------------------
    # METRIC EXTRACTORS
    # -----------------------------------------------------------------

    def _get_temperature(self) -> float:
        val = first_float(self.info, ["battery_temperature_c", "temperature_c", "temperature"], default=float("nan"))
        if math.isfinite(val):
            return val
        state = self._get_env_attr("_state", None)
        if state is not None:
            s_val = safe_float(getattr(state, "temperature_c", float("nan")), float("nan"))
            if math.isfinite(s_val):
                return s_val
        for attr in ("temperature_c", "battery_temperature_c"):
            a_val = safe_float(getattr(self.env, attr, float("nan")), float("nan"))
            if math.isfinite(a_val):
                return a_val
        return self.ui.ambient_c

    def _get_soc(self) -> float:
        val = first_float(self.info, ["soc", "battery_soc"], default=float("nan"))
        if math.isfinite(val):
            return val
        state = self._get_env_attr("_state", None)
        if state is not None:
            s_val = safe_float(getattr(state, "soc", float("nan")), float("nan"))
            if math.isfinite(s_val):
                return s_val
        return self.ui.initial_soc

    def _get_speed(self) -> float:
        if self.ui.mode == "driving":
            return self._display_speed_mps * 3.6
        return 0.0

    def _get_acceleration(self) -> float:
        if self.ui.mode == "driving":
            return self._display_accel_mps2
        return 0.0

    def _get_power(self) -> float:
        if self.ui.mode == "charging":
            curr = first_float(self.info, ["applied_current_a", "applied_current"], default=float("nan"))
            volt = first_float(self.info, ["terminal_voltage", "voltage_v", "voltage"], default=float("nan"))
            if math.isfinite(curr) and math.isfinite(volt):
                return (curr * volt) / 1000.0
            return 0.0
        power_w = first_float(self.info, ["applied_power_w", "battery_power_w"], default=0.0)
        return power_w / 1000.0

    def _get_regen(self) -> float:
        val = first_float(self.info, ["regen_power_w", "applied_regen_power_w"], default=float("nan"))
        if math.isfinite(val):
            return val / 1000.0
        applied_power = first_float(self.info, ["applied_power_w"], default=0.0)
        if applied_power > 0:
            return applied_power / 1000.0
        return 0.0

    def _get_current_ceiling(self) -> float:
        """Extract authoritative safe current ceiling from safety info."""
        s_info = self.info.get("safety_intervention", {})
        if isinstance(s_info, dict) and "safe_current_ceiling" in s_info:
            return abs(safe_float(s_info["safe_current_ceiling"], 160.0))
        # Compute from state-based temperature derating if not available in info
        temp = self._get_temperature()
        if temp <= 45.0:
            return 160.0
        elif temp >= 55.0:
            return 0.0
        return 160.0 * (1.0 - (temp - 45.0) / 10.0)

    def _get_power_deficit(self) -> float:
        return first_float(self.info, ["power_deficit_w"], default=0.0)

    def _get_friction_braking(self) -> float:
        return first_float(self.info, ["friction_braking_w"], default=0.0)

    def _get_applied_current(self) -> float:
        """Applied battery current (A), signed. Read from the authoritative env
        info / safety-layer output -- never fabricated."""
        val = first_float(self.info, ["applied_current_a", "applied_current"], default=float("nan"))
        if math.isfinite(val):
            return val
        s_info = self.info.get("safety_intervention", {})
        if isinstance(s_info, dict):
            return safe_float(s_info.get("applied_current", 0.0), 0.0)
        return 0.0

    def _get_requested_current(self) -> float:
        """PPO-requested battery current (A) BEFORE the safety ceiling was applied,
        taken from the authoritative safety-layer output (§28/§35)."""
        s_info = self.info.get("safety_intervention", {})
        if isinstance(s_info, dict) and "requested_current" in s_info:
            return safe_float(s_info["requested_current"], 0.0)
        return self._get_applied_current()

    def _get_voltage(self) -> float:
        """Authoritative battery terminal voltage (V). Prefer info; else derive
        the exact v_est used this step from applied power/current; else ECM."""
        volt = first_float(self.info, ["terminal_voltage", "voltage_v", "voltage"], default=float("nan"))
        if math.isfinite(volt) and volt > 0:
            return volt
        applied_p = first_float(self.info, ["applied_power_w", "battery_power_w"], default=float("nan"))
        applied_i = self._get_applied_current()
        if math.isfinite(applied_p) and abs(applied_i) > 1e-6:
            return abs(applied_p / applied_i)
        state = self._get_env_attr("_state", None)
        ecm = self._get_env_attr("ecm", None)
        if state is not None and ecm is not None:
            try:
                v = safe_float(ecm.terminal_voltage(state, 0.0), float("nan"))
                if math.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
        return 370.0

    def _get_requested_power(self) -> float:
        """Requested propulsion/charge power magnitude (kW): |requested current| x V (§28)."""
        return abs(self._get_requested_current() * self._get_voltage()) / 1000.0

    def _get_safe_available_power(self) -> float:
        """Safety-allowed power magnitude (kW): safe current ceiling x V (§28)."""
        return abs(self._get_current_ceiling() * self._get_voltage()) / 1000.0

    def _get_available_regen(self) -> float:
        """Total regen available at the wheels (kW) = battery-accepted regen +
        friction-dissipated braking (§30). Both terms come from env info."""
        return self._get_regen() + (self._get_friction_braking() / 1000.0)

    def _get_reference_speed(self) -> float:
        """Drive-cycle reference speed (km/h) at step t -- the imposed research trace,
        before any demo safe-stop override (§33). Never modifies the research cycle.

        Uses the pre-step snapshot captured by _sync_display_cycle_sample() rather
        than calling cycle.current_speed() directly: by the time this is called,
        env.step() has already advanced the cycle pointer to t+1, so a direct call
        would return the *next* step's target speed (Issue F)."""
        # _display_speed_mps is populated from cycle.current_speed() BEFORE env.step()
        speed_mps = safe_float(self._display_speed_mps, float("nan"))
        if math.isfinite(speed_mps):
            return speed_mps * 3.6
        return self._get_speed()

    def _get_safety_status(self) -> Tuple[str, Tuple[int, int, int]]:
        """Real BMS safety-layer status from info['safety_intervention']['type'] (§26).
        Never fabricate an intervention: a 'none' type is reported as NORMAL, not as
        an active hard protection just because the temperature region looks elevated."""
        s_info = self.info.get("safety_intervention", {})
        s_type = "none"
        if isinstance(s_info, dict):
            s_type = str(s_info.get("type", "none")).strip().lower()
        if s_type in ("none", "", "nan", "normal"):
            return ("NORMAL", GOOD)
        label = s_type.replace("_", " ").upper()
        if s_type in ("temperature", "thermal", "soc", "voltage", "derating", "soft"):
            return (label, WARN)
        return (label, BAD)

    def _append_state_trace(self, info: Optional[dict] = None, action: float = 0.0, reward: float = 0.0) -> None:
        self.trace_soc.add(self._get_soc() * 100.0)
        self.trace_temp.add(self._get_temperature())
        self.trace_action.add(action)
        self.trace_power.add(self._get_power())
        self.trace_ceiling.add(self._get_current_ceiling())
        self.trace_deficit.add(self._get_power_deficit() / 1000.0)

        speed_kmh = self._get_speed()
        self.trace_speed.add(speed_kmh if self.ui.mode == "driving" else 0.0)
        rec_speed = calculate_recommended_speed(
            self.ui.thermal_state, speed_kmh, self._get_current_ceiling(), 160.0, self.thermal_config
        )
        self.trace_rec_speed.add(rec_speed if self.ui.mode == "driving" else 0.0)

    def _current_metrics(self) -> Dict[str, Any]:
        if self.env is None:
            return {}
        temperature = self._get_temperature()
        soc = self._get_soc()
        metrics = {
            "SOC": soc * 100.0,
            "Temperature": temperature,
            "Ambient": first_float(self.info, ["ambient_temp_c"], default=self.ui.ambient_c),
        }
        if self.ui.mode == "charging":
            volt = first_float(self.info, ["terminal_voltage", "voltage_v", "voltage"], default=float("nan"))
            if not math.isfinite(volt):
                state = getattr(self.env, "_state", None)
                if state is not None and hasattr(self.env, "ecm"):
                    volt = safe_float(self.env.ecm.terminal_voltage(state, 0.0))
                else:
                    volt = 370.0
            curr = first_float(self.info, ["applied_current_a", "applied_current"], default=0.0)
            metrics.update({
                "Voltage": volt,
                "Applied Current": curr,
                "Target": safe_float(getattr(self.env, "target_soc", 0.95), 0.95) * 100.0,
                "Step": int(getattr(self.env, "_step_count", 0)),
                "Safety": str((self.info.get("safety_intervention", {}) or {}).get("type", "none")),
            })
        else:
            deficit_w = self._get_power_deficit()
            dt = safe_float(self._get_env_attr("dt", 1.0), 1.0)
            metrics.update({
                "Speed": self._get_speed(),
                "Battery Power": self._get_power(),
                "Regen": self._get_regen(),
                "Deficit Wh": deficit_w * dt / 3600.0,
                "Safety": str((self.info.get("safety_intervention", {}) or {}).get("type", "none")),
            })
        return metrics

    # -----------------------------------------------------------------
    # EVENT HANDLING
    # -----------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            raise SystemExit

        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                raise SystemExit
            elif event.key == pygame.K_TAB:
                self.toggle_mode()
            elif event.key == pygame.K_m:
                self.toggle_sim_mode()
            elif event.key == pygame.K_SPACE:
                self.ui.playing = not self.ui.playing
                self._set_message("PLAYING" if self.ui.playing else "PAUSED")
            elif event.key == pygame.K_RIGHT:
                self._step_once()
            elif event.key == pygame.K_r:
                self._reset_env()
                self._set_message("RESET")
            elif event.key == pygame.K_b:
                self.toggle_controller()
            elif event.key == pygame.K_v:
                self.toggle_speed_recommendation()
            elif event.key == pygame.K_s:
                self.trigger_stop_vehicle()
            elif event.key == pygame.K_u:
                self.trigger_resume()
            elif event.key == pygame.K_1:
                self.set_cycle(0)
            elif event.key == pygame.K_2:
                self.set_cycle(1)
            elif event.key == pygame.K_3:
                self.set_cycle(2)
            elif event.key == pygame.K_4:
                self.set_cycle(3)
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                self.ui.speed_multiplier = min(8.0, self.ui.speed_multiplier * 1.5)
                self._set_message(f"Speed: {self.ui.speed_multiplier:.1f}x")
            elif event.key == pygame.K_MINUS:
                self.ui.speed_multiplier = max(0.25, self.ui.speed_multiplier / 1.5)
                self._set_message(f"Speed: {self.ui.speed_multiplier:.1f}x")
            elif event.key == pygame.K_UP:
                self.ui.ambient_c = min(50.0, self.ui.ambient_c + 2.0)
                self._reset_env()
                self._set_message(f"Ambient: {self.ui.ambient_c:.0f} °C")
            elif event.key == pygame.K_DOWN:
                self.ui.ambient_c = max(15.0, self.ui.ambient_c - 2.0)
                self._reset_env()
                self._set_message(f"Ambient: {self.ui.ambient_c:.0f} °C")

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            if self.play_rect.collidepoint(pos):
                self.ui.playing = not self.ui.playing
                self._set_message("PLAYING" if self.ui.playing else "PAUSED")
            elif self.step_rect.collidepoint(pos):
                self._step_once()
                self._set_message("Advanced 1 simulation step")
            elif self.reset_rect.collidepoint(pos):
                self._reset_env()
                self._set_message("RESET")
            elif self.mode_rect.collidepoint(pos):
                self.toggle_mode()
            elif self.sim_mode_rect.collidepoint(pos):
                self.toggle_sim_mode()
            elif self.controller_rect.collidepoint(pos):
                self.toggle_controller()
            elif self.cycle_rect.collidepoint(pos):
                self.set_cycle(self.ui.cycle_index + 1)
            elif self.rec_speed_btn.collidepoint(pos):
                self.toggle_speed_recommendation()
            elif self.stop_veh_btn.collidepoint(pos):
                self.trigger_stop_vehicle()
            elif self.resume_btn.collidepoint(pos):
                self.trigger_resume()
            elif self.ambient_down_rect.collidepoint(pos):
                self.ui.ambient_c = max(15.0, self.ui.ambient_c - 2.0)
                self._reset_env()
                self._set_message(f"Ambient: {self.ui.ambient_c:.0f} °C")
            elif self.ambient_up_rect.collidepoint(pos):
                self.ui.ambient_c = min(50.0, self.ui.ambient_c + 2.0)
                self._reset_env()
                self._set_message(f"Ambient: {self.ui.ambient_c:.0f} °C")
            elif self.speed_down_rect.collidepoint(pos):
                self.ui.speed_multiplier = max(0.25, self.ui.speed_multiplier / 1.5)
                self._set_message(f"Speed: {self.ui.speed_multiplier:.1f}x")
            elif self.speed_up_rect.collidepoint(pos):
                self.ui.speed_multiplier = min(8.0, self.ui.speed_multiplier * 1.5)
                self._set_message(f"Speed: {self.ui.speed_multiplier:.1f}x")

    def update(self) -> None:
        """Frame update step managing wall clock accumulation and physics rate."""
        now = time.perf_counter()
        dt_wall = min(now - self.last_step_wall, 0.25)
        self.last_step_wall = now

        # Update vehicle animation parameters
        speed_mps = self._display_speed_mps
        self.wheel_rotation_deg = (self.wheel_rotation_deg + speed_mps * dt_wall * 180.0) % 360.0
        self.animation_phase += dt_wall * 2.0

        if not self.ui.playing:
            return

        env_dt = safe_float(self._get_env_attr("dt", 1.0), 1.0)
        self.step_accumulator += dt_wall * self.ui.speed_multiplier
        steps_to_run = int(self.step_accumulator / max(env_dt, 1e-6))
        steps_to_run = min(steps_to_run, self.MAX_STEPS_PER_FRAME)
        self.step_accumulator -= steps_to_run * env_dt

        for _ in range(steps_to_run):
            self._step_once()
            if self.done:
                self.step_accumulator = 0.0
                break

    # -----------------------------------------------------------------
    # RENDERING & DRAWING
    # Professional, subtle, engineering-style visual hierarchy (§1 - §25)
    # -----------------------------------------------------------------

    def draw_text(self, text: str, x: int, y: int, color=UI_TEXT_PRIMARY, font=None) -> None:
        surface = (font or self.font).render(str(text), True, color)
        self.screen.blit(surface, (x, y))

    def draw_value_unit(
        self,
        val_str: str,
        unit_str: str,
        x: int,
        y: int,
        val_col=UI_TEXT_PRIMARY,
        unit_col=UI_TEXT_SECONDARY,
        font_val=None,
        font_unit=None,
    ) -> int:
        """Render numeric value prominently with smaller, muted unit text beside it (§2)."""
        f_val = font_val or self.font_val
        f_unit = font_unit or self.font_unit
        s_val = f_val.render(str(val_str), True, val_col)
        self.screen.blit(s_val, (x, y))
        if unit_str:
            s_unit = f_unit.render(" " + str(unit_str), True, unit_col)
            unit_y = y + (s_val.get_height() - s_unit.get_height()) - 1
            self.screen.blit(s_unit, (x + s_val.get_width(), unit_y))
            return s_val.get_width() + s_unit.get_width()
        return s_val.get_width()

    def rounded_panel(self, rect: pygame.Rect, color=UI_PANEL, radius=10, border_color=UI_BORDER) -> None:
        """Render elevated card panel with subtle drop shadow and border (§5)."""
        # Subtle dark elevation shadow
        shadow_rect = pygame.Rect(rect.x + 1, rect.y + 2, rect.width, rect.height)
        pygame.draw.rect(self.screen, (8, 11, 16), shadow_rect, border_radius=radius)
        # Main surface
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)
        if border_color is not None:
            pygame.draw.rect(self.screen, border_color, rect, width=1, border_radius=radius)

    def draw(self) -> None:
        """Render complete application frame with unified 24px grid alignment (§3)."""
        self.screen.fill(UI_BACKGROUND)

        self._draw_header()
        self._draw_toolbar()
        self._draw_thermal_and_safety_panel()
        self._draw_power_flow_panel()
        self._draw_vehicle_and_motion_panel()

        # Bottom 2-column layout (24px gap, perfectly anchored to left X=24 and right X=1416)
        if self.ui.sim_mode == "demo":
            self._draw_dual_state_panels()
        else:
            self._draw_state_action_panel()

        self._draw_live_charts()
        self._draw_message_bar()

        pygame.display.flip()

    def _draw_header(self) -> None:
        """Header with clear technical typography, mode descriptor, and semantic badges."""
        self.draw_text("RL-BMS DRIVING EMS", 24, 18, UI_TEXT_PRIMARY, self.font_big)
        mode_desc = (
            "Interactive Driving Mode Thermal Protection & Battery-Life Energy Management System"
            if self.ui.mode == "driving"
            else "DC Fast-Charging Thermal Protection & Dynamic Current Optimization"
        )
        self.draw_text(mode_desc, 26, 54, UI_TEXT_SECONDARY, self.font_small)

        # Simulation Mode Badge (Selection uses UI_ACCENT/UI_WARNING, strictly avoiding health green overload)
        is_demo = (self.ui.sim_mode == "demo")
        badge_bg = (40, 28, 18) if is_demo else (18, 30, 48)
        badge_color = UI_WARNING if is_demo else UI_ACCENT
        badge_icon = "▲" if is_demo else "●"
        badge_text = f"{badge_icon} DEMO MODE" if is_demo else f"{badge_icon} RESEARCH BENCHMARK"
        badge_rect = pygame.Rect(984, 22, 214, 32)
        pygame.draw.rect(self.screen, badge_bg, badge_rect, border_radius=6)
        pygame.draw.rect(self.screen, badge_color, badge_rect, width=1, border_radius=6)
        self.draw_text(badge_text, 998, 29, badge_color, self.font_hud)

        # Controller indicator badge
        ctrl_name = "PPO" if self.ui.controller == "ppo" else ("RULE-BASED" if self.ui.mode == "driving" else "MAX-I")
        ctrl_rect = pygame.Rect(1222, 22, 194, 32)
        pygame.draw.rect(self.screen, (18, 30, 48), ctrl_rect, border_radius=6)
        pygame.draw.rect(self.screen, UI_ACCENT, ctrl_rect, width=1, border_radius=6)
        self.draw_text(f"CTRL: {ctrl_name}", 1238, 29, UI_ACCENT, self.font_hud)

    def _draw_btn(
        self,
        rect: pygame.Rect,
        label: str,
        active: bool = False,
        disabled: bool = False,
        variant: str = "default",  # "default" | "accent" | "stop" | "resume"
    ) -> None:
        """Standardized button component with explicit states: active/selected, available, disabled (§11, §12)."""
        if disabled:
            fill = (18, 22, 28)
            brd = UI_BORDER_SUBTLE
            text_col = UI_TEXT_DISABLED
        elif active:
            # Active/Selected state: neutral blue accent theme
            fill = (24, 45, 75)
            brd = UI_ACCENT
            text_col = (255, 255, 255)
        elif variant == "stop":
            # Routine stop control: subtle restrained brown/red border, not glaring alarm (§1)
            fill = (34, 22, 24)
            brd = (90, 42, 46)
            text_col = (235, 175, 175)
        elif variant == "resume":
            fill = (18, 36, 26)
            brd = (40, 95, 52)
            text_col = UI_SAFE_RESUME
        elif variant == "accent":
            fill = (20, 32, 50)
            brd = UI_ACCENT
            text_col = UI_ACCENT
        else:
            fill = UI_PANEL_BUTTON
            brd = UI_BORDER
            text_col = UI_TEXT_PRIMARY

        pygame.draw.rect(self.screen, fill, rect, border_radius=6)
        pygame.draw.rect(self.screen, brd, rect, width=1, border_radius=6)
        surf = self.font_btn.render(label, True, text_col)
        tx = rect.x + (rect.width  - surf.get_width())  // 2
        ty = rect.y + (rect.height - surf.get_height()) // 2
        self.screen.blit(surf, (tx, ty))

    def _draw_toolbar(self) -> None:
        """Render standardized interactive toolbar with unified padding and clear states (§11, §12)."""
        playing  = self.ui.playing
        is_demo  = self.ui.sim_mode == "demo"
        can_resume = self.ui.thermal_state == ThermalState.SAFE_TO_RESUME

        cycle_label = self.cycle_paths[self.ui.cycle_index][0] if self.ui.mode == "driving" else "N/A"
        ctrl_label  = "PPO" if self.ui.controller == "ppo" else "BASELINE"

        # ── Main Control Buttons ────────────────────────────────────────────
        self._draw_btn(self.play_rect, "PAUSE" if playing else "PLAY", active=playing)
        self._draw_btn(self.step_rect, "STEP")
        self._draw_btn(self.reset_rect, "RESET")

        mode_lbl = "DRIVING" if self.ui.mode == "driving" else "CHARGING"
        self._draw_btn(self.mode_rect, mode_lbl, active=(self.ui.mode == "driving"))

        self._draw_btn(self.sim_mode_rect, self.ui.sim_mode.upper(), active=is_demo)
        self._draw_btn(self.controller_rect, ctrl_label, active=(self.ui.controller == "ppo"))
        self._draw_btn(self.cycle_rect, cycle_label)

        self._draw_btn(self.rec_speed_btn, "SPD REC", active=self.ui.show_speed_recommendation)

        # Stop Button (restrained subtle stop styling, not alarm-red)
        self._draw_btn(self.stop_veh_btn, "STOP", variant="stop")

        # Resume Button (active only when safe to resume)
        if can_resume:
            self._draw_btn(self.resume_btn, "RESUME", variant="resume", active=True)
        else:
            self._draw_btn(self.resume_btn, "RESUME", disabled=True)

        # ── Ambient Temperature Capsule (X = 920..1152) ────────────────────
        amb_box = pygame.Rect(920, 94, 232, 34)
        self.rounded_panel(amb_box, color=UI_PANEL, radius=6, border_color=UI_BORDER_SUBTLE)
        self.draw_text("Ambient:", 934, 104, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{self.ui.ambient_c:.0f}", "°C", 996, 103, val_col=UI_TEXT_PRIMARY)
        self._draw_btn(self.ambient_down_rect, "-")
        self._draw_btn(self.ambient_up_rect,   "+")

        # ── Simulation Speed Capsule (X = 1176..1416) ──────────────────────
        spd_box = pygame.Rect(1176, 94, 240, 34)
        self.rounded_panel(spd_box, color=UI_PANEL, radius=6, border_color=UI_BORDER_SUBTLE)
        self.draw_text("Sim Speed:", 1190, 104, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{self.ui.speed_multiplier:.1f}", "x", 1264, 103, val_col=UI_TEXT_PRIMARY)
        self._draw_btn(self.speed_down_rect, "-")
        self._draw_btn(self.speed_up_rect,   "+")

    def _get_thermal_badge_info(self, state: ThermalState) -> Tuple[str, Tuple[int, int, int]]:
        """Return icon + label + color for thermal state to aid colorblind accessibility (§5, §6)."""
        mapping = {
            ThermalState.OPTIMAL:          ("● OPTIMAL", COLOR_OPTIMAL),
            ThermalState.ELEVATED_THERMAL: ("▲ ELEVATED", COLOR_ELEVATED),
            ThermalState.DERATING_ACTIVE:  ("▲ DERATING", COLOR_DERATING),
            ThermalState.CRITICAL:         ("■ CRITICAL", COLOR_CRITICAL),
            ThermalState.STOP_REQUESTED:   ("■ STOP REQ", COLOR_CRITICAL),
            ThermalState.DECELERATING:     ("▼ BRAKING", COLOR_CRITICAL),
            ThermalState.STOPPED:          ("■ STOPPED", COLOR_COOLING),
            ThermalState.COOLING:          ("❄ COOLING", COLOR_COOLING),
            ThermalState.SAFE_TO_RESUME:   ("✓ SAFE RESUME", COLOR_RESUME),
        }
        return mapping.get(state, ("● OPTIMAL", COLOR_OPTIMAL))

    def _get_thermal_color(self, state: ThermalState) -> Tuple[int, int, int]:
        _, col = self._get_thermal_badge_info(state)
        return col

    def _get_car_body_color(self, state: ThermalState) -> Tuple[int, int, int]:
        return {
            ThermalState.OPTIMAL:          CAR_BODY_OPTIMAL,
            ThermalState.ELEVATED_THERMAL: CAR_BODY_ELEVATED,
            ThermalState.DERATING_ACTIVE:  CAR_BODY_DERATING,
            ThermalState.CRITICAL:         CAR_BODY_CRITICAL,
            ThermalState.STOP_REQUESTED:   CAR_BODY_CRITICAL,
            ThermalState.DECELERATING:     CAR_BODY_CRITICAL,
            ThermalState.STOPPED:          CAR_BODY_COOLING,
            ThermalState.COOLING:          CAR_BODY_COOLING,
            ThermalState.SAFE_TO_RESUME:   CAR_BODY_RESUME,
        }.get(state, CAR_BODY_OPTIMAL)

    def _draw_thermal_and_safety_panel(self) -> None:
        """Render Battery Thermal & Safety HUD (Card 1: 24px grid aligned, snapped ticks, §5, §6, §9)."""
        rect = pygame.Rect(24, 146, 432, 240)
        st = self.ui.thermal_state
        badge_text, st_color = self._get_thermal_badge_info(st)

        # Dynamic subtle panel border based on thermal state (§15)
        if st in (ThermalState.CRITICAL, ThermalState.STOP_REQUESTED, ThermalState.DECELERATING):
            panel_border = UI_CRITICAL
        elif st in (ThermalState.ELEVATED_THERMAL, ThermalState.DERATING_ACTIVE):
            panel_border = UI_WARNING
        elif st in (ThermalState.COOLING, ThermalState.STOPPED):
            panel_border = UI_COOLING
        elif st == ThermalState.SAFE_TO_RESUME:
            panel_border = UI_SAFE_RESUME
        else:
            panel_border = UI_BORDER

        self.rounded_panel(rect, radius=10, border_color=panel_border)

        # Card Title Header with high contrast (§2)
        self.draw_text("BATTERY THERMAL & SAFETY", rect.x + 16, rect.y + 14, UI_TEXT_PRIMARY, self.font_title)

        # State badge with icon for colorblind accessibility (§5, §6)
        state_badge_rect = pygame.Rect(rect.right - 146, rect.y + 12, 130, 24)
        badge_bg = (st_color[0] // 6, st_color[1] // 6, st_color[2] // 6)
        pygame.draw.rect(self.screen, badge_bg, state_badge_rect, border_radius=5)
        pygame.draw.rect(self.screen, st_color, state_badge_rect, width=1, border_radius=5)
        self.draw_text(badge_text, state_badge_rect.x + 8, state_badge_rect.y + 4, st_color, self.font_badge)

        # Temperature Readings with Unit Differentiation (§2)
        temp = self._get_temperature()
        self.draw_text("Pack Temp:", rect.x + 16, rect.y + 46, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{temp:.2f}", "°C", rect.x + 88, rect.y + 44, val_col=st_color)

        self.draw_text("Ambient:", rect.x + 230, rect.y + 46, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{self.ui.ambient_c:.1f}", "°C", rect.x + 292, rect.y + 44, val_col=UI_TEXT_PRIMARY)

        # ── Thermal Scale Progress Bar with Exact Snapped Ticks (§5, §9) ─────
        bar_x = rect.x + 16
        bar_y = rect.y + 74
        bar_w = rect.width - 32   # 400px
        bar_h = 12

        pygame.draw.rect(self.screen, UI_PANEL_ALT, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, UI_BORDER_SUBTLE, (bar_x, bar_y, bar_w, bar_h), width=1, border_radius=4)

        t_norm = np.clip((temp - 25.0) / 35.0, 0.0, 1.0)
        fill_w = int(bar_w * t_norm)
        pygame.draw.rect(self.screen, st_color, (bar_x, bar_y, fill_w, bar_h), border_radius=4)

        # Snapped Ticks and precise labels (25°C, 33°C, 45°C, 55°C)
        ticks = [
            (25.0, "25°C"),
            (33.0, "33°C"),
            (45.0, "45°C"),
            (55.0, "55°C"),
        ]
        for t_val, t_lbl in ticks:
            tx = bar_x + int(bar_w * np.clip((t_val - 25.0) / 35.0, 0.0, 1.0))
            pygame.draw.line(self.screen, UI_BORDER, (tx, bar_y + bar_h), (tx, bar_y + bar_h + 4), 1)
            lbl_surf = self.font_unit.render(t_lbl, True, UI_TEXT_MUTED)
            # Center label under tick mark
            lx = tx - lbl_surf.get_width() // 2
            lx = max(bar_x, min(lx, bar_x + bar_w - lbl_surf.get_width()))
            self.screen.blit(lbl_surf, (lx, bar_y + bar_h + 5))

        # ── Current Ceiling & Safety Layer Status (§6) ──────────────────────
        ceiling_a = self._get_current_ceiling()
        derate_pct = max(0.0, (1.0 - ceiling_a / 160.0) * 100.0)
        self.draw_text("Current Ceiling:", bar_x, bar_y + 40, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{ceiling_a:.1f}", f"A ({derate_pct:.0f}% Derated)", bar_x + 98, bar_y + 38, val_col=UI_TEXT_PRIMARY)

        # Real BMS Safety Status Badge with icon
        safety_label, safety_col = self._get_safety_status()
        sf_icon = "✓" if safety_label == "NORMAL" else "▲"
        sfx_surf = self.font_small.render(f"Safety: {sf_icon} {safety_label}", True, safety_col)
        self.screen.blit(sfx_surf, (rect.right - sfx_surf.get_width() - 16, bar_y + 40))

        ceil_bar_y = bar_y + 60
        pygame.draw.rect(self.screen, UI_PANEL_ALT, (bar_x, ceil_bar_y, bar_w, 10), border_radius=4)
        pygame.draw.rect(self.screen, UI_BORDER_SUBTLE, (bar_x, ceil_bar_y, bar_w, 10), width=1, border_radius=4)
        c_norm = np.clip(ceiling_a / 160.0, 0.0, 1.0)
        c_color = UI_NORMAL if c_norm > 0.8 else (UI_WARNING if c_norm > 0.3 else UI_CRITICAL)
        pygame.draw.rect(self.screen, c_color, (bar_x, ceil_bar_y, int(bar_w * c_norm), 10), border_radius=4)

        # ── Driver Guidance HUD Card (§18, §21: cleanly split, zero overflow) ──
        guidance_raw = get_driver_guidance(st)
        raw_lines = [l.strip() for l in guidance_raw.replace("\r", "").split("\n") if l.strip()]
        headline = raw_lines[0] if raw_lines else "NORMAL OPERATION"
        subtext = " — ".join(raw_lines[1:]) if len(raw_lines) > 1 else ""

        g_box_y = bar_y + 80
        g_box_h = 38
        g_bg = (st_color[0] // 8, st_color[1] // 8, st_color[2] // 8)
        pygame.draw.rect(self.screen, g_bg, (bar_x, g_box_y, bar_w, g_box_h), border_radius=4)
        pygame.draw.rect(self.screen, st_color, (bar_x, g_box_y, bar_w, g_box_h), width=1, border_radius=4)

        self.draw_text("GUIDANCE: " + headline, bar_x + 8, g_box_y + 4, st_color, self.font_btn)
        if subtext:
            max_w = bar_w - 16
            while len(subtext) > 4 and self.font_small.render(subtext, True, UI_TEXT_SECONDARY).get_width() > max_w:
                subtext = subtext[:-5] + "..."
            self.draw_text(subtext, bar_x + 8, g_box_y + 20, UI_TEXT_SECONDARY, self.font_unit)

    def _draw_power_flow_panel(self) -> None:
        """Render Power Protection & Energy Flow Diagrams (Card 2: 24px grid aligned, §7, §28, §29, §30)."""
        rect = pygame.Rect(480, 146, 480, 240)
        self.rounded_panel(rect, radius=10)

        self.draw_text("POWER PROTECTION & PROPULSION FLOW", rect.x + 16, rect.y + 14, UI_TEXT_PRIMARY, self.font_title)

        power_kw  = self._get_power()
        deficit_w = self._get_power_deficit()
        regen_kw  = self._get_regen()
        friction_w = self._get_friction_braking()
        req_kw    = self._get_requested_power()
        avail_kw  = self._get_safe_available_power()
        avail_regen_kw = self._get_available_regen()

        # ── Row 1: Requested vs. Safe Available Power ──
        self.draw_text("Requested Power:", rect.x + 16, rect.y + 44, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{req_kw:.2f}", "kW", rect.x + 120, rect.y + 42, val_col=UI_TEXT_PRIMARY)

        avail_col = UI_NORMAL if avail_kw >= req_kw - 1e-3 else UI_WARNING
        self.draw_text("Safe Available:", rect.x + 250, rect.y + 44, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{avail_kw:.2f}", "kW", rect.x + 348, rect.y + 42, val_col=avail_col)

        # ── Row 2: Applied Battery Power vs. Power Deficit ──
        self.draw_text("Applied Power:", rect.x + 16, rect.y + 66, UI_TEXT_SECONDARY, self.font_small)
        p_col = UI_COOLING if power_kw >= 0 else UI_NORMAL
        self.draw_value_unit(f"{power_kw:+.2f}", "kW", rect.x + 120, rect.y + 64, val_col=p_col)

        deficit_col = UI_CRITICAL if deficit_w > 10.0 else (UI_WARNING if deficit_w > 0 else UI_TEXT_MUTED)
        self.draw_text("Power Deficit:", rect.x + 250, rect.y + 66, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{deficit_w:.0f}", "W", rect.x + 348, rect.y + 64, val_col=deficit_col)

        # ── Row 3: Regenerative Capture vs. Mechanical Friction ──
        self.draw_text("Regen Accepted:", rect.x + 16, rect.y + 88, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{regen_kw:.2f}", f"/ {avail_regen_kw:.2f} kW", rect.x + 120, rect.y + 86, val_col=UI_NORMAL)

        fric_col = UI_WARNING if friction_w > 0 else UI_TEXT_MUTED
        self.draw_text("Friction Braking:", rect.x + 250, rect.y + 88, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{friction_w:.0f}", "W", rect.x + 348, rect.y + 86, val_col=fric_col)

        # ── Flow Diagram (§7) ──
        flow_y   = rect.y + 116
        box_w, box_h = 82, 30
        total_boxes = 4
        usable_w = rect.width - 32
        step_w   = usable_w // (total_boxes - 1)

        node_labels = ["BATTERY", "INVERTER", "MOTOR", "WHEELS"]
        node_xs = [rect.x + 16 + i * step_w for i in range(total_boxes)]
        node_xs[-1] = min(node_xs[-1], rect.right - 16 - box_w)

        arrow_color = UI_COOLING if power_kw > 0.05 else (UI_NORMAL if regen_kw > 0.05 else UI_BORDER)
        flow_frac = min(1.0, abs(power_kw) / 60.0)
        arrow_w = max(2, int(2 + flow_frac * 4))

        for i, (label, bx) in enumerate(zip(node_labels, node_xs)):
            node_fill = UI_PANEL_ALT
            pygame.draw.rect(self.screen, node_fill, (bx, flow_y, box_w, box_h), border_radius=5)
            pygame.draw.rect(self.screen, arrow_color, (bx, flow_y, box_w, box_h), width=1, border_radius=5)
            surf = self.font_btn.render(label, True, UI_TEXT_PRIMARY)
            tx = bx + (box_w - surf.get_width()) // 2
            ty = flow_y + (box_h - surf.get_height()) // 2
            self.screen.blit(surf, (tx, ty))

            if i < total_boxes - 1:
                ax1 = bx + box_w + 3
                ax2 = node_xs[i + 1] - 3
                if ax2 > ax1 + 4:
                    ay = flow_y + box_h // 2
                    pygame.draw.line(self.screen, arrow_color, (ax1, ay), (ax2 - 6, ay), arrow_w)
                    ah = 3 + arrow_w
                    pygame.draw.polygon(self.screen, arrow_color, [
                        (ax2 - 6, ay - ah),
                        (ax2,     ay),
                        (ax2 - 6, ay + ah),
                    ])

        # ── Objective Footer ──
        self.draw_text("OBJECTIVE: BATTERY-LIFE-ORIENTED PROTECTION",
                       rect.x + 16, rect.y + 202, UI_TEXT_MUTED, self.font_small)
        self.draw_text("Efficiency · Thermal Protection · Current Stress · Peak Demand · Safety",
                       rect.x + 16, rect.y + 220, UI_TEXT_MUTED, self.font_unit)

    def _draw_spoke(self, cx: int, cy: int, r: int, angle_deg: float, color: Tuple[int, int, int]) -> None:
        a = math.radians(angle_deg)
        ex = cx + int(r * math.cos(a))
        ey = cy + int(r * math.sin(a))
        pygame.draw.line(self.screen, color, (cx, cy), (ex, ey), 2)

    def _draw_vehicle_and_motion_panel(self) -> None:
        """Render Live Vehicle & Dynamic Car Representation (Card 3: 24px grid aligned, §33, §38)."""
        rect = pygame.Rect(984, 146, 432, 240)
        self.rounded_panel(rect, radius=10)

        self.draw_text("LIVE VEHICLE TELEMETRY", rect.x + 16, rect.y + 14, UI_TEXT_PRIMARY, self.font_title)

        speed_kmh = self._get_speed()
        rec_speed = calculate_recommended_speed(
            self.ui.thermal_state, speed_kmh, self._get_current_ceiling(), 160.0, self.thermal_config
        )
        accel = self._get_acceleration()
        dist_km = self.cumulative_distance_m / 1000.0

        # Level 1 Large Speedometer Readout with Unit Differentiation (§2, §4)
        self.draw_value_unit(f"{speed_kmh:.1f}", "km/h", rect.x + 16, rect.y + 38, val_col=UI_TEXT_PRIMARY, font_val=self.font_big, font_unit=self.font)

        ref_kmh = self._get_reference_speed()
        self.draw_text("REF:", rect.x + 210, rect.y + 44, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{ref_kmh:.1f}", "km/h", rect.x + 242, rect.y + 42, val_col=UI_TEXT_PRIMARY)

        if self.ui.sim_mode == "demo" and (ref_kmh - speed_kmh) > 0.5:
            self.draw_text("SAFE-STOP:", rect.x + 210, rect.y + 64, UI_WARNING, self.font_small)
            self.draw_value_unit(f"{speed_kmh:.1f}", "km/h", rect.x + 284, rect.y + 62, val_col=UI_WARNING)

        if self.ui.show_speed_recommendation:
            rec_col = UI_WARNING if rec_speed < speed_kmh - 1.0 else UI_NORMAL
            self.draw_text("REC:", rect.x + 16, rect.y + 76, UI_TEXT_SECONDARY, self.font_small)
            self.draw_value_unit(f"{rec_speed:.1f}", "km/h", rect.x + 48, rect.y + 74, val_col=rec_col)

        # Level 2 Telemetry Row
        tele_y = rect.y + 96
        self.draw_text("Dist:", rect.x + 16, tele_y, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{dist_km:.2f}", "km", rect.x + 48, tele_y - 2)

        self.draw_text("Accel:", rect.x + 158, tele_y, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{accel:+.2f}", "m/s²", rect.x + 196, tele_y - 2)

        self.draw_text("Time:", rect.x + 306, tele_y, UI_TEXT_SECONDARY, self.font_small)
        self.draw_value_unit(f"{self._display_time_s:.0f}", "s", rect.x + 342, tele_y - 2)

        # ── Technical Vehicle Chassis Graphic Area ──
        car_area_y = rect.y + 116
        car_area_h = rect.height - 116 - 8

        WHEEL_R = 14
        BODY_H  = 30
        ROOF_H  = 16
        BODY_W  = 210
        car_x   = rect.x + (rect.width - BODY_W) // 2
        body_y  = car_area_y + car_area_h - WHEEL_R - BODY_H
        wheel_y = body_y + BODY_H

        body_fill   = self._get_car_body_color(self.ui.thermal_state)
        body_border = self._get_thermal_color(self.ui.thermal_state)
        wheel_col   = (140, 148, 160)
        hub_col     = (200, 208, 220)
        spoke_col   = (60, 68, 78)

        # Road Line
        road_y = wheel_y + WHEEL_R + 2
        pygame.draw.line(self.screen, UI_BORDER_SUBTLE, (rect.x + 12, road_y), (rect.right - 12, road_y), 1)

        # Car Body
        pygame.draw.rect(self.screen, body_fill, (car_x, body_y, BODY_W, BODY_H), border_radius=6)
        pygame.draw.rect(self.screen, body_border, (car_x, body_y, BODY_W, BODY_H), width=1, border_radius=6)

        # Roof
        roof_margin = 28
        roof_x = car_x + roof_margin
        roof_w = BODY_W - 2 * roof_margin
        roof_y = body_y - ROOF_H
        pygame.draw.rect(self.screen, body_fill, (roof_x, roof_y, roof_w, ROOF_H + 4), border_radius=6)
        pygame.draw.rect(self.screen, body_border, (roof_x, roof_y, roof_w, ROOF_H + 4), width=1, border_radius=6)

        win_col = tuple(min(c + 40, 255) for c in body_fill)
        pygame.draw.line(self.screen, win_col, (roof_x + 4, roof_y + 4), (car_x + 44, body_y), 2)
        pygame.draw.line(self.screen, win_col, (roof_x + roof_w - 4, roof_y + 4), (car_x + BODY_W - 44, body_y), 2)

        # Wheels with rotating spokes
        w1_x = car_x + 38
        w2_x = car_x + BODY_W - 38
        for wx in (w1_x, w2_x):
            pygame.draw.circle(self.screen, (18, 22, 28), (wx, wheel_y), WHEEL_R)
            pygame.draw.circle(self.screen, wheel_col, (wx, wheel_y), WHEEL_R, width=2)
            for angle_offset in (0, 90, 180, 270):
                a = self.wheel_rotation_deg + angle_offset
                self._draw_spoke(wx, wheel_y, WHEEL_R - 3, a, spoke_col)
            pygame.draw.circle(self.screen, hub_col, (wx, wheel_y), 3)

        # Status text inside vehicle chassis
        if speed_kmh <= 0.01:
            status_str = "● STOPPED"
        elif accel < -0.5:
            status_str = "▼ BRAKING"
        else:
            status_str = f"▶ {speed_kmh:.0f} km/h"
        st_surf = self.font_btn.render(status_str, True, body_border)
        sx = car_x + (BODY_W - st_surf.get_width()) // 2
        sy = body_y + (BODY_H - st_surf.get_height()) // 2
        self.screen.blit(st_surf, (sx, sy))

    def _draw_metric_row(
        self,
        rect: pygame.Rect,
        label: str,
        val: str,
        unit: str,
        col: Tuple[int, int, int],
        y: int,
        pill: bool = False,
        icon: str = "",
    ) -> None:
        """Draw a key-value metric row with unit de-emphasis and accessibility icons (§2, §6)."""
        self.draw_text(label, rect.x + 16, y, UI_TEXT_SECONDARY, self.font_small)
        val_x = rect.x + 155

        if pill:
            pill_text = (icon + " " + val) if icon else val
            val_surf = self.font_badge.render(pill_text, True, col)
            vw = val_surf.get_width() + 12
            vh = val_surf.get_height() + 4
            pill_rect = pygame.Rect(val_x, y - 2, vw, vh)
            pill_fill = tuple(c // 6 for c in col)
            pygame.draw.rect(self.screen, pill_fill, pill_rect, border_radius=4)
            pygame.draw.rect(self.screen, col, pill_rect, width=1, border_radius=4)
            self.screen.blit(val_surf, (pill_rect.x + 6, pill_rect.y + 2))
        else:
            self.draw_value_unit(val, unit, val_x, y - 2, val_col=col)

    def _benchmark_display_values(self) -> Dict[str, float]:
        """Return the six 'LAST BENCHMARK STATE' values."""
        if self.benchmark_frozen and self.benchmark_snapshot:
            s = self.benchmark_snapshot
            return {
                "reference_speed_kmh": s["reference_speed_kmh"],
                "battery_power_kw": s["battery_power_kw"],
                "power_deficit_w": s["power_deficit_w"],
                "soc_pct": s["soc_pct"],
                "temperature_c": s["temperature_c"],
                "ceiling_a": s["ceiling_a"],
            }
        return {
            "reference_speed_kmh": self._get_reference_speed(),
            "battery_power_kw": self._get_power(),
            "power_deficit_w": self._get_power_deficit(),
            "soc_pct": self._get_soc() * 100.0,
            "temperature_c": self._get_temperature(),
            "ceiling_a": self._get_current_ceiling(),
        }

    def _demo_cooling_status(self) -> Tuple[str, Tuple[int, int, int]]:
        """Human-readable passive-cooling status for the DEMO SAFETY STATE panel."""
        st = self.ui.thermal_state
        temp = self._get_temperature()
        if st == ThermalState.COOLING:
            return f"❄ COOLING ({temp:.1f} °C)", COLOR_COOLING
        if st == ThermalState.STOPPED:
            return f"■ HOLDING ({temp:.1f} °C)", COLOR_COOLING
        if st == ThermalState.SAFE_TO_RESUME:
            return "✓ COOLED (SAFE)", COLOR_RESUME
        if st in (ThermalState.STOP_REQUESTED, ThermalState.DECELERATING):
            return "▼ STOPPING", COLOR_CRITICAL
        return "● NOMINAL", UI_TEXT_SECONDARY

    def _draw_dual_state_panels(self) -> None:
        """Render Demo split panels with exact 24px grid alignment: LAST BENCHMARK and DEMO SAFETY STATE."""
        tc = self._get_thermal_color(self.ui.thermal_state)

        # ── Panel 1: LAST BENCHMARK STATE (X=24, Y=410, W=360, H=221) ────
        r1 = pygame.Rect(24, 410, 360, 221)
        border_r1 = UI_ACCENT if self.benchmark_frozen else UI_BORDER
        self.rounded_panel(r1, radius=10, border_color=border_r1)
        title1 = "LAST BENCHMARK STATE" + ("  (FROZEN)" if self.benchmark_frozen else "")
        self.draw_text(title1, r1.x + 16, r1.y + 12, UI_TEXT_PRIMARY, self.font_title)
        note = ("benchmark paused during safety stop" if self.benchmark_frozen else "live research benchmark")
        self.draw_text(note, r1.x + 16, r1.y + 32, UI_TEXT_MUTED, self.font_unit)

        bm = self._benchmark_display_values()
        deficit = bm["power_deficit_w"]
        b_rows = [
            ("Reference Speed", f"{bm['reference_speed_kmh']:.1f}", "km/h", UI_TEXT_PRIMARY, False, ""),
            ("Battery Power",   f"{bm['battery_power_kw']:+.2f}", "kW",   UI_COOLING, False, ""),
            ("Power Deficit",   f"{deficit:.0f}", "W", UI_CRITICAL if deficit > 0 else UI_NORMAL, False, ""),
            ("SOC",             f"{bm['soc_pct']:.1f}", "%", UI_NORMAL if bm['soc_pct'] > 20.0 else UI_CRITICAL, True, "●"),
            ("Temperature",     f"{bm['temperature_c']:.2f}", "°C", tc, True, "▲" if tc != COLOR_OPTIMAL else "●"),
            ("Safety Ceiling",  f"{bm['ceiling_a']:.1f}", "A", UI_ACCENT, False, ""),
        ]
        y = r1.y + 54
        for label, val, unit, col, pill, icon in b_rows:
            self._draw_metric_row(r1, label, val, unit, col, y, pill=pill, icon=icon)
            y += 27

        # ── Panel 2: DEMO SAFETY STATE (X=24, Y=655, W=360, H=221) ───────
        r2 = pygame.Rect(24, 655, 360, 221)
        self.rounded_panel(r2, radius=10)
        self.draw_text("DEMO SAFETY STATE", r2.x + 16, r2.y + 12, UI_WARNING, self.font_title)
        self.draw_text("live demo vehicle & safety controller", r2.x + 16, r2.y + 32, UI_TEXT_MUTED, self.font_unit)

        demo_speed_kmh = self._display_speed_mps * 3.6
        rec_speed = calculate_recommended_speed(
            self.ui.thermal_state, self._get_reference_speed(),
            self._get_current_ceiling(), 160.0, self.thermal_config,
        )
        active = self.safety_stop_ctrl.state.is_active
        if active and self.safety_stop_ctrl.state.manually_stopped:
            stop_status, stop_col, s_icon = "MANUAL STOP", UI_WARNING, "■"
        elif active:
            stop_status, stop_col, s_icon = "AUTO STOP", COLOR_CRITICAL, "■"
        else:
            stop_status, stop_col, s_icon = "NONE", UI_NORMAL, "✓"
        cooling_label, cooling_col = self._demo_cooling_status()
        badge_lbl, badge_col = self._get_thermal_badge_info(self.ui.thermal_state)

        d_rows = [
            ("Demo Veh. Speed", f"{demo_speed_kmh:.1f}", "km/h",
                                UI_NORMAL if demo_speed_kmh > 0.05 else COLOR_COOLING, True, "▶" if demo_speed_kmh > 0.05 else "■"),
            ("Safety State",    badge_lbl, "", tc, True, ""),
            ("Stop Status",     stop_status, "", stop_col, True, s_icon),
            ("Cooling Status",  cooling_label, "", cooling_col, False, ""),
            ("Recommended Spd", f"{rec_speed:.1f}", "km/h", UI_WARNING, False, ""),
        ]
        y = r2.y + 54
        for label, val, unit, col, pill, icon in d_rows:
            self._draw_metric_row(r2, label, val, unit, col, y, pill=pill, icon=icon)
            y += 32

    def _draw_state_action_panel(self) -> None:
        """Render Live State / Action summary panel (Research Mode: X=24, Y=410, W=360, H=466, §34, §35)."""
        rect = pygame.Rect(24, 410, 360, 466)
        self.rounded_panel(rect, radius=10)

        self.draw_text("STATE / ACTION SUMMARY", rect.x + 16, rect.y + 14, UI_TEXT_PRIMARY, self.font_title)

        soc  = self._get_soc()
        temp = self._get_temperature()
        badge_lbl, tc = self._get_thermal_badge_info(self.ui.thermal_state)
        safety_label, safety_col = self._get_safety_status()
        volt = self._get_voltage()
        applied_i = self._get_applied_current()
        requested_i = self._get_requested_current()
        ceiling_a = self._get_current_ceiling()
        deficit_w = self._get_power_deficit()

        rows = [
            ("SOC",            f"{soc*100.0:.1f}", "%",    UI_NORMAL if soc > 0.2 else UI_CRITICAL, True, "●"),
            ("Terminal Voltage", f"{volt:.1f}", "V",       UI_TEXT_PRIMARY, False, ""),
            ("Pack Temp",      f"{temp:.2f}", "°C",        tc, True, "▲" if tc != COLOR_OPTIMAL else "●"),
            ("Ambient",        f"{self.ui.ambient_c:.1f}", "°C", UI_TEXT_SECONDARY, False, ""),
            ("Thermal State",  badge_lbl, "",              tc, True, ""),
            ("Safety Layer",   safety_label, "",           safety_col, True, "✓" if safety_label == "NORMAL" else "▲"),
            ("PPO Action",     f"{safe_float(self._action()[0]):+.3f}", "", UI_PURPLE, False, ""),
            ("Requested I",    f"{requested_i:+.1f}", "A",  UI_TEXT_MUTED, False, ""),
            ("Safety Ceiling", f"{ceiling_a:.1f}", "A",     UI_ACCENT, False, ""),
            ("Applied I",      f"{applied_i:+.1f}", "A",    UI_TEXT_PRIMARY, False, ""),
            ("Power Deficit",  f"{deficit_w:.0f}", "W",     UI_CRITICAL if deficit_w > 0 else UI_NORMAL, False, ""),
            ("Regen Power",    f"{self._get_regen():.2f}", "kW", UI_NORMAL, False, ""),
        ]

        y_offset = rect.y + 44
        for label, val, unit, col, use_pill, icon in rows:
            self._draw_metric_row(rect, label, val, unit, col, y_offset, pill=use_pill, icon=icon)
            y_offset += 34

    def _draw_live_charts(self) -> None:
        """Render synchronized oscilloscope traces (X=408, Y=410, W=1008, H=466, §13, §14)."""
        chart_rect = pygame.Rect(408, 410, 1008, 466)
        self.rounded_panel(chart_rect, radius=10)

        self.draw_text("LIVE DYNAMICS & CONTROL TRACES", chart_rect.x + 16, chart_rect.y + 14, UI_TEXT_PRIMARY, self.font_title)

        # 4 Subplots with 14px gap and consistent scientific mappings (§13):
        # 1. Temperature: Red / thermal
        # 2. Speed: White / Light neutral (Ref) vs Amber (Rec)
        # 3. Power: Cyan / Blue (Power) vs Red (Deficit)
        # 4. Multi-scale Control: Green (SOC), Blue (Safety Ceiling), Purple (Action)
        sub_h = 88
        gap = 14
        y1 = chart_rect.y + 44
        y2 = y1 + sub_h + gap
        y3 = y2 + sub_h + gap
        y4 = y3 + sub_h + gap
        chart_w = 976

        self._draw_single_trace(
            chart_rect.x + 16, y1, chart_w, sub_h,
            [self.trace_temp], ["Temperature (°C)"], [UI_CRITICAL],
            y_min=20.0, y_max=60.0
        )
        self._draw_single_trace(
            chart_rect.x + 16, y2, chart_w, sub_h,
            [self.trace_speed, self.trace_rec_speed], ["Ref Speed (km/h)", "Recommended (km/h)"], [UI_TEXT_PRIMARY, UI_WARNING],
            y_min=0.0, y_max=130.0
        )
        self._draw_single_trace(
            chart_rect.x + 16, y3, chart_w, sub_h,
            [self.trace_power, self.trace_deficit], ["Battery Power (kW)", "Deficit (kW)"], [UI_COOLING, UI_CRITICAL],
            y_min=-30.0, y_max=90.0
        )
        self._draw_single_trace(
            chart_rect.x + 16, y4, chart_w, sub_h,
            [self.trace_soc, self.trace_ceiling, self.trace_action],
            ["SOC (%)", "Safety Ceiling (A)", "PPO Action"],
            [UI_NORMAL, UI_ACCENT, UI_PURPLE],
            y_min=0.0, y_max=100.0,
            scales=[(0.0, 100.0), (0.0, 160.0), (-1.0, 1.0)],
        )

    def _draw_single_trace(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        traces: List[Trace],
        labels: List[str],
        colors: List[Tuple[int, int, int]],
        y_min: float,
        y_max: float,
        scales: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Render single oscilloscope subplot with empty-state placeholders & clean grid (§4)."""
        LEGEND_H = 22
        plot_y = y + LEGEND_H
        plot_h = h - LEGEND_H

        # Background surface & subtle border
        pygame.draw.rect(self.screen, UI_PANEL_ALT, (x, plot_y, w, plot_h), border_radius=4)
        pygame.draw.rect(self.screen, UI_BORDER_SUBTLE, (x, plot_y, w, plot_h), width=1, border_radius=4)

        # Subtle horizontal grid lines (4 divisions)
        for gi in range(1, 4):
            gy = plot_y + int(gi * plot_h / 4)
            pygame.draw.line(self.screen, (22, 28, 36), (x + 1, gy), (x + w - 2, gy), 1)

        # Legend above plot area
        lx = x + 6
        for li, (label, color) in enumerate(zip(labels, colors)):
            pygame.draw.rect(self.screen, color, (lx, y + 6, 8, 8), border_radius=2)
            if scales is not None and li < len(scales):
                s_min, s_max = scales[li]
                label = f"{label} [{s_min:g}..{s_max:g}]"
            surf = self.font_small.render(label, True, UI_TEXT_SECONDARY)
            self.screen.blit(surf, (lx + 12, y + 4))
            lx += surf.get_width() + 24

        # Check if traces have active recorded points
        has_data = any(len(t.y) >= 2 for t in traces)
        if not has_data:
            # Active empty/rest state placeholder (§4)
            placeholder = "● Awaiting telemetry — Press PLAY or STEP to record live dynamics"
            p_surf = self.font_small.render(placeholder, True, UI_TEXT_MUTED)
            px = x + (w - p_surf.get_width()) // 2
            py = plot_y + (plot_h - p_surf.get_height()) // 2
            self.screen.blit(p_surf, (px, py))
            return

        # Plot traces
        for ti, (trace, color) in enumerate(zip(traces, colors)):
            if len(trace.y) < 2:
                continue
            if scales is not None and ti < len(scales):
                t_min, t_max = scales[ti]
            else:
                t_min, t_max = y_min, y_max
            pts = []
            for i, val in enumerate(trace.y):
                px = x + int(i * (w - 2) / max(1, trace.max_points - 1))
                norm_y = np.clip((val - t_min) / max(1e-6, t_max - t_min), 0.0, 1.0)
                py = plot_y + plot_h - 2 - int(norm_y * (plot_h - 4))
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, color, False, pts, 2)
            if pts:
                pygame.draw.circle(self.screen, color, pts[-1], 3)

    def _draw_message_bar(self) -> None:
        if self.message and time.perf_counter() < self.message_until:
            msg_rect = pygame.Rect(WIDTH // 2 - 250, HEIGHT - 45, 500, 32)
            pygame.draw.rect(self.screen, UI_PANEL_ALT, msg_rect, border_radius=6)
            pygame.draw.rect(self.screen, UI_ACCENT, msg_rect, width=1, border_radius=6)
            self.draw_text(self.message, msg_rect.x + 20, msg_rect.y + 6, UI_TEXT_PRIMARY, self.font_small)

    def run(self) -> None:
        """Main application execution loop."""
        while True:
            for event in pygame.event.get():
                self.handle_event(event)
            self.update()
            self.draw()
            self.clock.tick(FPS)


if __name__ == "__main__":
    simulator = InteractiveSimulator()
    simulator.run()