# Phase 3: Action Mapping and Trace-Following Verification

## Action Space Verification

### From environment/ev_energy_env.py:
```python
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
```

**Action Dimension**: 1 (scalar action in range [-1.0, 1.0])
**Action Lower Bound**: -1.0
**Action Upper Bound**: 1.0

### Action-to-Control Mapping Verification

From environment/ev_energy_env.py lines 138-171:

#### Propulsion Case (forces.p_wheel >= 0.0):
1. **action_val** ∈ [-1.0, 1.0] (from PPO/controller)
2. **desired_battery_power_w** = action_val * self.max_discharge_power_w (if action_val < 0) or action_val * self.max_charge_power_w (if action_val >= 0)
3. **required_discharge_w** = drivetrain_out.battery_power_w (positive magnitude required from vehicle dynamics)
4. **requested_discharge_w** = max(0.0, -desired_battery_power_w) (magnitude PPO is offering)
5. **supplied_discharge_w** = min(requested_discharge_w, required_discharge_w) (actual supplied, clamped by availability)
6. **power_deficit_w** = required_discharge_w - supplied_discharge_w (unmet demand)
7. **feasible_desired_power_w** = -supplied_discharge_w (signed, discharge = negative)
8. **v_est** = self.ecm.terminal_voltage(self._state, 0.0) (pre-step terminal voltage)
9. **requested_current_a** = feasible_desired_power_w / v_est (if v_est > 0.0 else 0.0) (Power -> current conversion)
10. **applied_current_a, safety_info** = safety_layer_bidirectional(requested_current_a, ...) (Safety layer application)
11. **applied_power_w** = applied_current_a * v_est (final applied power)

#### Regeneration Case (forces.p_wheel < 0.0):
1. **action_val** ∈ [-1.0, 1.0] (from PPO/controller)
2. **desired_battery_power_w** = action_val * self.max_charge_power_w (if action_val >= 0) or action_val * self.max_discharge_power_w (if action_val < 0)
3. **available_w** = drivetrain_out.available_regenerative_power_w (mechanical power available at wheels)
4. **requested_charge_w** = max(0.0, desired_battery_power_w) (charge magnitude PPO is requesting)
5. **used_regen_w** = min(requested_charge_w, available_w) (actual regen used, clamped by availability)
6. **friction_braking_w** = available_w - used_regen_w (regen not used -> friction loss)
7. **feasible_desired_power_w** = used_regen_w (signed, charge = positive)
8. **v_est** = self.ecm.terminal_voltage(self._state, 0.0) (pre-step terminal voltage)
9. **requested_current_a** = feasible_desired_power_w / v_est (if v_est > 0.0 else 0.0) (Power -> current conversion)
10. **applied_current_a, safety_info** = safety_layer_bidirectional(requested_current_a, ...) (Safety layer application)
11. **applied_power_w** = applied_current_a * v_est (final applied power)

### Trace-Following Behavior Verification

From environment/ev_energy_env.py line 132:
```python
speed = self._drive_cycle.current_speed()
```

And from environment/drive_cycle.py:
```python
def current_speed(self) -> float:
    return self._samples[self._idx].speed_mps
```

**Conclusion**: The environment uses trace-following where:
1. DriveCycle provides the reference speed/acceleration/grade at current timestep
2. Vehicle dynamics compute wheel power demand from (speed, acceleration, grade) 
3. PPO/EMS determines if battery can supply/absorb the implied power
4. Power deficit is computed but does NOT alter vehicle_dynamics inputs (per module docstring)
5. Reference speed remains unchanged regardless of battery power availability

This confirms the architecture:
```
Drive Cycle
   ↓
Reference Speed
   ↓
Vehicle Dynamics
   ↓
Wheel Power Demand
   ↓
EMS/PPO Controller
   ↓
Requested Battery Power
   ↓
Bidirectional Safety Layer
   ↓
Feasible Current
   ↓
ECM Step
   ↓
Battery State Update
```

## PPO Action Mapping Details

### From app/interactive_ev_simulator.py (_action method):
```python
def _action(self) -> np.ndarray:
    if self.obs is None:
        return np.array([0.0], dtype=np.float32)

    if self.ui.controller == "ppo" and self.model is not None:
        action, _ = self.model.predict(self.obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    # ... baseline controller handling ...

    return np.array([0.0], dtype=np.float32)
```

**PPO Action Processing**:
1. PPO model predicts action from observation: `action, _ = self.model.predict(self.obs, deterministic=True)`
2. Action is reshaped to 1D array: `np.asarray(action, dtype=np.float32).reshape(-1)`
3. Action value is in range [-1.0, 1.0] matching environment action space
4. This action represents the normalized requested battery power (positive = charging, negative = discharging)

### Action Scaling Verification

From ev_energy_env.py lines 141-144:
```python
if action_val >= 0.0:
    desired_battery_power_w = action_val * self.max_charge_power_w
else:
    desired_battery_power_w = action_val * self.max_discharge_power_w  # already negative
```

Where:
- `self.max_charge_power_w` = energy_config["max_desired_charge_power_w"] 
- `self.max_discharge_power_w` = energy_config["max_desired_discharge_power_w"]

This maps the [-1.0, 1.0] action to actual power limits:
- action = -1.0 → maximum discharge power (negative value)
- action = 0.0 → zero power 
- action = +1.0 → maximum charge power (positive value)

## Summary

✅ **Action Space**: Properly defined as Box(-1.0, 1.0, shape=(1,))  
✅ **Action Mapping**: Clear path from PPO action → requested power → requested current → safety layer → applied current  
✅ **Trace-Following**: Confirmed - drive cycle provides reference speed that is NOT altered by power deficits  
✅ **Safety Layer Integration**: Proper bidirectional safety layer application after power-to-current conversion  
✅ **Power Deficit Handling**: Computed and reported but does not feedback to alter reference speed (open-loop prescribed speed)  

The implementation correctly follows the specified architecture where PPO is an optimizer that requests power, but the safety layer has final authority over what current is actually applied to the battery.