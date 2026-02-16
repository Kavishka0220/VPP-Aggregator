"""
Check actual bus-level power injections with storage
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import opendssdirect as dss
from pathlib import Path

dss_path = Path(r"D:\UoM\FYP\VPP-Aggregator\openDSS\feeder_houses.dss")

def get_total_power():
    """Get total power from source"""
    dss.Circuit.SetActiveElement("vsource.source")
    powers = dss.CktElement.Powers()
    total_kw = sum(powers[::2]) / 1000  # Convert W to kW, sum phases
    return total_kw

print("\n" + "="*70)
print("Bus-Level Power Flow Analysis")
print("="*70)

# Scenario 1: Discharge
print("\n--- DISCHARGE: Storage injecting 40 kW ---")
dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')
for i in range(10):
    dss.Command(f"edit load.Load{i} kw=5")  # Total 50 kW load
dss.Command("edit storage.BESS state=discharging kw=40")
dss.Solution.Solve()
source_kw_discharge = get_total_power()
print(f"Source power: {source_kw_discharge:.2f} kW")
print(f"Expected: ~10 kW (50 kW load - 40 kW battery discharge)")

# Scenario 2: Charging  
print("\n--- CHARGING: Storage consuming 40 kW ---")
dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')
for i in range(10):
    dss.Command(f"edit load.Load{i} kw=5")  # Total 50 kW load
dss.Command("edit storage.BESS state=charging kw=40")
dss.Solution.Solve()
source_kw_charge = get_total_power()
print(f"Source power: {source_kw_charge:.2f} kW")
print(f"Expected: ~90 kW (50 kW load + 40 kW battery charging)")

# Scenario 3: Idle
print("\n--- IDLE: No storage ---")
dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')
for i in range(10):
    dss.Command(f"edit load.Load{i} kw=5")  # Total 50 kW load
dss.Command("edit storage.BESS state=idling")
dss.Solution.Solve()
source_kw_idle = get_total_power()
print(f"Source power: {source_kw_idle:.2f} kW")
print(f"Expected: ~50 kW (just the load)")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print(f"\nSource power difference:")
print(f"  Discharge vs Idle: {source_kw_discharge - source_kw_idle:.2f} kW (should be -40 kW)")
print(f"  Charge vs Idle:    {source_kw_charge - source_kw_idle:.2f} kW (should be +40 kW)")

if abs((source_kw_discharge - source_kw_idle) + 40) < 5:
    discharge_ok = "✓"
else:
    discharge_ok = "✗"
    
if abs((source_kw_charge - source_kw_idle) - 40) < 5:
    charge_ok = "✓"
else:
    charge_ok = "✗"

print(f"\nDischarge working correctly: {discharge_ok}")
print(f"Charging working correctly: {charge_ok}")

if discharge_ok == "✓" and charge_ok == "✓":
    print("\n✅ YES - Batteries properly affect power flow and voltages!")
else:
    print(f"\n❌ Storage not working correctly in OpenDSS")

print("="*70 + "\n")
