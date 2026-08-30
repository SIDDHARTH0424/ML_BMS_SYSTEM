"""
Structured Event Logger and Data Isolation Manager for RL-BMS-Driving.

Guarantees strict data isolation between Research Benchmark outputs (runs/, audit/, RESULTS/)
and Demo Mode outputs (demo_runs/).
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ThermalEvent:
    event_name: str
    timestamp: float
    simulation_time: float
    temperature_c: float
    soc: float
    speed_kmh: float
    safety_ceiling_a: float
    thermal_state: str
    mode: str
    details: Optional[Dict[str, Any]] = None


class SimulatorLogger:
    """Manages event logging and trajectory data storage with Research/Demo isolation."""

    def __init__(self, root_dir: Path, mode: str = "research"):
        self.root_dir = Path(root_dir)
        self.mode = mode.lower()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        self.events: List[ThermalEvent] = []
        self.trajectory_rows: List[Dict[str, Any]] = []
        self.intervened: bool = False
        self.intervention_reasons: List[str] = []

        if self.mode == "demo":
            self.output_dir = self.root_dir / "demo_runs" / f"demo_{self.session_id}"
        else:
            self.output_dir = self.root_dir / "runs" / f"research_{self.session_id}"

    def log_event(
        self,
        event_name: str,
        simulation_time: float,
        temperature_c: float,
        soc: float,
        speed_kmh: float,
        safety_ceiling_a: float,
        thermal_state: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a discrete thermal or safety intervention event."""
        event = ThermalEvent(
            event_name=event_name,
            timestamp=time.time(),
            simulation_time=float(simulation_time),
            temperature_c=float(temperature_c),
            soc=float(soc),
            speed_kmh=float(speed_kmh),
            safety_ceiling_a=float(safety_ceiling_a),
            thermal_state=str(thermal_state),
            mode=self.mode,
            details=details or {},
        )
        self.events.append(event)

        if self.mode == "research" and (
            "manual" in event_name.lower() or "override" in event_name.lower()
        ):
            self.intervened = True
            self.intervention_reasons.append(event_name)

    def log_trajectory_step(self, step_data: Dict[str, Any]) -> None:
        """Append one synchronized timestep to the trajectory record."""
        row = dict(step_data)
        row["mode"] = self.mode
        row["timestamp"] = time.time()
        self.trajectory_rows.append(row)

    def save_session(self, config_data: Optional[Dict[str, Any]] = None) -> Path:
        """Persist session outputs to disk in the isolated directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. config.json
        cfg_path = self.output_dir / "config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config_data or {"mode": self.mode}, f, indent=2)

        # 2. events.json
        events_path = self.output_dir / "events.json"
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)

        # 3. trajectory.csv
        if self.trajectory_rows:
            traj_path = self.output_dir / "trajectory.csv"
            fieldnames = list(self.trajectory_rows[0].keys())
            with open(traj_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.trajectory_rows)

        # 4. summary.json
        summary_path = self.output_dir / "summary.json"
        summary = {
            "mode": self.mode,
            "session_id": self.session_id,
            "total_steps": len(self.trajectory_rows),
            "total_events": len(self.events),
            "intervened": self.intervened,
            "benchmark_status": (
                "INTERVENED — NOT STANDARD BENCHMARK" if self.intervened else "STANDARD BENCHMARK"
            ),
            "intervention_reasons": self.intervention_reasons,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return self.output_dir
