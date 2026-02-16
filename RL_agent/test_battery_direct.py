"""
Direct OpenDSS test to demonstrate battery impact on voltages
(bypassing RL environment's automatic charging logic)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openDSS.run_opendss import VPPDSSRunner
from pathlib import Path

# Initialize OpenDSS
dss_path = Path(r"D:\UoM\FYP\VPP-Aggregator\openDSS\feeder_houses.dss")
runner = VPPDSSRunner(dss_path)

# Fixed load and solar scenario
loads_kw = [5.0] * 10  # 5 kW each house
pv_kw = {0: 3.0, 1: 3.0, 2: 3.0, 4: 3.0, 6: 3.0, 8: 3.0}  # 3 kW each PV

print("\n" + "="*70)
print("BATTERY VOLTAGE IMPACT TEST (Direct OpenDSS)")
print("="*70)

# Test 1: No battery action
print("\n--- SCENARIO 1: No Battery Action (Baseline) ---")
result1 = runner.step(loads_kw, pv_kw, batt_home0_kw=0.0, batt_home2_kw=0.0, bess_kw=0.0)
print(f"Battery Powers: Home0=0.0 kW, Home2=0.0 kW, BESS=0.0 kW")
print(f"N0 voltage:   {result1.vmin_pu_by_bus[0]:.6f} p.u.")
print(f"N2 voltage:   {result1.vmin_pu_by_bus[2]:.6f} p.u.")
print(f"NBESS voltage: {result1.vmin_pu_by_bus[10]:.6f} p.u.")

# Test 2: Heavy battery discharge (inject power -> increase voltage)
print("\n--- SCENARIO 2: Heavy Discharge (Batteries injecting power) ---")
result2 = runner.step(loads_kw, pv_kw, batt_home0_kw=5.0, batt_home2_kw=5.0, bess_kw=40.0)
print(f"Battery Powers: Home0=+5.0 kW, Home2=+5.0 kW, BESS=+40.0 kW")
print(f"N0 voltage:   {result2.vmin_pu_by_bus[0]:.6f} p.u.")
print(f"N2 voltage:   {result2.vmin_pu_by_bus[2]:.6f} p.u.")
print(f"NBESS voltage: {result2.vmin_pu_by_bus[10]:.6f} p.u.")

# Test 3: Heavy battery charging (consume power -> decrease voltage)
print("\n--- SCENARIO 3: Heavy Charging (Batteries consuming power) ---")
result3 = runner.step(loads_kw, pv_kw, batt_home0_kw=-5.0, batt_home2_kw=-5.0, bess_kw=-40.0)
print(f"Battery Powers: Home0=-5.0 kW, Home2=-5.0 kW, BESS=-40.0 kW")
print(f"N0 voltage:   {result3.vmin_pu_by_bus[0]:.6f} p.u.")
print(f"N2 voltage:   {result3.vmin_pu_by_bus[2]:.6f} p.u.")
print(f"NBESS voltage: {result3.vmin_pu_by_bus[10]:.6f} p.u.")

# Analysis
print("\n" + "="*70)
print("VOLTAGE IMPACT ANALYSIS")
print("="*70)

v0_baseline = result1.vmin_pu_by_bus[0]
v2_baseline = result1.vmin_pu_by_bus[2]
vbess_baseline = result1.vmin_pu_by_bus[10]

print(f"\nBus N0 (Home Battery Location):")
print(f"  Discharge impact: {(result2.vmin_pu_by_bus[0] - v0_baseline)*1000:+.2f} mV")
print(f"  Charge impact:    {(result3.vmin_pu_by_bus[0] - v0_baseline)*1000:+.2f} mV")

print(f"\nBus N2 (Home Battery Location):")
print(f"  Discharge impact: {(result2.vmin_pu_by_bus[2] - v2_baseline)*1000:+.2f} mV")
print(f"  Charge impact:    {(result3.vmin_pu_by_bus[2] - v2_baseline)*1000:+.2f} mV")

print(f"\nBus NBESS (BESS Location):")
print(f"  Discharge impact: {(result2.vmin_pu_by_bus[10] - vbess_baseline)*1000:+.2f} mV")
print(f"  Charge impact:    {(result3.vmin_pu_by_bus[10] - vbess_baseline)*1000:+.2f} mV")

# Check physics
discharge_raises = result2.vmin_pu_by_bus[10] > vbess_baseline
charge_lowers = result3.vmin_pu_by_bus[10] < vbess_baseline

print("\n" + "="*70)
if discharge_raises and charge_lowers:
    print("✅ CORRECT PHYSICS:")
    print("   → Battery DISCHARGE (inject power) INCREASES voltages")
    print("   → Battery CHARGE (consume power) DECREASES voltages")
    print("\n   This is the expected behavior for distributed energy resources!")
elif discharge_raises or charge_lowers:
    print("⚠️  PARTIAL IMPACT:")
    print("   → Batteries affect voltages but behavior may be inconsistent")
else:
    print("❌ NO VOLTAGE IMPACT:")
    print("   → Batteries do not affect voltages (check OpenDSS configuration)")

print("="*70 + "\n")
