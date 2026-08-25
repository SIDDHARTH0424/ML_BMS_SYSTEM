# Interactive EV Simulator

A 2D real-time visualization layer for the existing RL-BMS-Driving project.
It does **not** replace the battery, drivetrain, vehicle, safety, or PPO models.
It runs the existing environments and trained checkpoints and visualizes their state/action flow.

## Install

From the project root:

```powershell
pip install pygame
```

## Launch

```powershell
python -m app.interactive_ev_simulator
```

If you are running the source as a package, an `app/__init__.py` file is included.

## Interaction

- `TAB`: switch Charging / Driving
- `SPACE`: play / pause
- `RIGHT`: single simulation step
- `R`: reset
- `B`: switch PPO / baseline controller
- `1`–`4`: select UDDS / HWFET / US06 / WLTP in Driving mode
- `+` / `-`: animation speed
- `UP` / `DOWN`: change ambient temperature and reset
- `Q` / `ESC`: quit

The visualizer looks for validated demo models under `final_models/`:

```text
final_models/charging_A1_50k_seed7/trained_model.zip
final_models/driving_B3_100k_seed7/ppo_driving_100000_steps.zip
```

If a requested model is absent, the app falls back to the deterministic baseline controller for the selected mode when available.

## Demo sequence

1. Start in Charging mode.
2. Press `SPACE` to play.
3. Press `TAB` to switch to Driving.
4. Press `1`–`4` to select a standardized cycle.
5. Press `B` to compare PPO with the deterministic baseline.
6. Pause at a braking event and use `RIGHT` to step one environment timestep at a time.
7. Change ambient temperature with `UP` / `DOWN` to demonstrate thermal response in simulation.
