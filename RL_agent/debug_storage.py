"""
Debug OpenDSS storage element state
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import opendssdirect as dss
from pathlib import Path

dss_path = Path(r"D:\UoM\FYP\VPP-Aggregator\openDSS\feeder_houses.dss")

print("\n" + "="*70)
print("OpenDSS Storage Element Debug")
print("="*70)

dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')

# Set loads and PV
dss.Command("edit load.Load0 kw=5")
dss.Command("edit generator.PV0 kw=3")

print("\n--- Test 1: Discharge Mode ---")
dss.Command("edit storage.BESS state=discharging kw=40")
dss.Solution.Solve()

# Query storage element
dss.Circuit.SetActiveElement("storage.BESS")
print(f"State: {dss.CktElement.Name()}")
print(f"Powers: {dss.CktElement.Powers()}")
print(f"Voltages: {dss.CktElement.VoltagesMagAng()[::2]}")

print("\n--- Test 2: Charging Mode ---")
dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')
dss.Command("edit load.Load0 kw=5")
dss.Command("edit generator.PV0 kw=3")
dss.Command("edit storage.BESS state=charging kw=40")
dss.Solution.Solve()

dss.Circuit.SetActiveElement("storage.BESS")
print(f"State: {dss.CktElement.Name()}")
print(f"Powers: {dss.CktElement.Powers()}")
print(f"Voltages: {dss.CktElement.VoltagesMagAng()[::2]}")

print("\n--- Test 3: Idle Mode ---")
dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')
dss.Command("edit load.Load0 kw=5")
dss.Command("edit generator.PV0 kw=3")
dss.Command("edit storage.BESS state=idling")
dss.Solution.Solve()

dss.Circuit.SetActiveElement("storage.BESS")
print(f"State: {dss.CktElement.Name()}")
print(f"Powers: {dss.CktElement.Powers()}")
print(f"Voltages: {dss.CktElement.VoltagesMagAng()[::2]}")

print("\n" + "="*70 + "\n")
