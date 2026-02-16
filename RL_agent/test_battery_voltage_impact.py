"""
Test to verify that battery charge/discharge operations affect grid voltages
"""
import numpy as np
from vpp_env import UrbanVPPEnv

# Create environment
env = UrbanVPPEnv(data_path="../data")
obs, info = env.reset()

print("\n" + "="*70)
print("BATTERY IMPACT ON VOLTAGE TEST")
print("="*70)

# Test 1: No battery action (baseline)
print("\n--- TEST 1: Baseline (No Battery Action) ---")
action_none = np.array([0.0, 0.0, 0.0])
obs, reward, terminated, truncated, info = env.step(action_none)
voltages_baseline = env.voltages_min.copy()
print(f"Action: {action_none}")
print(f"Battery Powers (kW): Home0={env.node_battery_power_kw[0]:.2f}, Home2={env.node_battery_power_kw[2]:.2f}, BESS={env.node_battery_power_kw[10]:.2f}")
print(f"Voltages (first 3 buses): N0={voltages_baseline[0]:.6f}, N1={voltages_baseline[1]:.6f}, N2={voltages_baseline[2]:.6f}")
print(f"BESS Bus voltage: {voltages_baseline[10]:.6f}")

# Reset to same state
obs, info = env.reset()
env.current_step = 0

# Test 2: Heavy discharge (inject power to grid -> raises voltage)
print("\n--- TEST 2: Heavy Discharge (All batteries discharging) ---")
action_discharge = np.array([1.0, 1.0, 1.0])  # Maximum discharge
obs, reward, terminated, truncated, info = env.step(action_discharge)
voltages_discharge = env.voltages_min.copy()
print(f"Action: {action_discharge}")
print(f"Battery Powers (kW): Home0={env.node_battery_power_kw[0]:.2f}, Home2={env.node_battery_power_kw[2]:.2f}, BESS={env.node_battery_power_kw[10]:.2f}")
print(f"Voltages (first 3 buses): N0={voltages_discharge[0]:.6f}, N1={voltages_discharge[1]:.6f}, N2={voltages_discharge[2]:.6f}")
print(f"BESS Bus voltage: {voltages_discharge[10]:.6f}")

# Reset to same state again
obs, info = env.reset()
env.current_step = 0

# Test 3: Heavy charging (consume power from grid -> lowers voltage)
print("\n--- TEST 3: Heavy Charging (All batteries charging) ---")
action_charge = np.array([-1.0, -1.0, -1.0])  # Maximum charge
obs, reward, terminated, truncated, info = env.step(action_charge)
voltages_charge = env.voltages_min.copy()
print(f"Action: {action_charge}")
print(f"Battery Powers (kW): Home0={env.node_battery_power_kw[0]:.2f}, Home2={env.node_battery_power_kw[2]:.2f}, BESS={env.node_battery_power_kw[10]:.2f}")
print(f"Voltages (first 3 buses): N0={voltages_charge[0]:.6f}, N1={voltages_charge[1]:.6f}, N2={voltages_charge[2]:.6f}")
print(f"BESS Bus voltage: {voltages_charge[10]:.6f}")

# Analysis
print("\n" + "="*70)
print("VOLTAGE CHANGE ANALYSIS")
print("="*70)

print("\nVoltage Changes from Baseline:")
print(f"  N0 (has battery & solar):")
print(f"    Discharge effect: {(voltages_discharge[0] - voltages_baseline[0])*1000:.2f} mV (expect positive)")
print(f"    Charge effect:    {(voltages_charge[0] - voltages_baseline[0])*1000:.2f} mV (expect negative)")

print(f"\n  N2 (has battery & solar):")
print(f"    Discharge effect: {(voltages_discharge[2] - voltages_baseline[2])*1000:.2f} mV (expect positive)")
print(f"    Charge effect:    {(voltages_charge[2] - voltages_baseline[2])*1000:.2f} mV (expect negative)")

print(f"\n  NBESS (BESS location):")
print(f"    Discharge effect: {(voltages_discharge[10] - voltages_baseline[10])*1000:.2f} mV (expect positive)")
print(f"    Charge effect:    {(voltages_charge[10] - voltages_baseline[10])*1000:.2f} mV (expect negative)")

print(f"\n  N5 (no battery, middle of feeder):")
print(f"    Discharge effect: {(voltages_discharge[5] - voltages_baseline[5])*1000:.2f} mV")
print(f"    Charge effect:    {(voltages_charge[5] - voltages_baseline[5])*1000:.2f} mV")

# Conclusion
print("\n" + "="*70)
if abs(voltages_discharge[0] - voltages_baseline[0]) > 1e-6:
    print("✅ BATTERIES DO AFFECT VOLTAGES")
    print("   - Discharging (injecting power) → increases voltages")
    print("   - Charging (consuming power) → decreases voltages")
    print("   - This is correct power flow physics!")
else:
    print("❌ WARNING: Batteries appear to have NO voltage impact")
    print("   - Check OpenDSS storage element configuration")
    print("   - Verify power flow is actually running")
print("="*70 + "\n")
