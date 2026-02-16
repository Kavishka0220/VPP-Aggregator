"""
Test with all DERs disabled
"""
import opendssdirect as dss
from pathlib import Path

dss_path = Path(r"D:\UoM\FYP\VPP-Aggregator\openDSS\feeder_houses.dss")

print("\n"  + "="*70)
print("Test with DERs Disabled")
print("="*70)

dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')

# Disable all PV
for i in [0, 1, 2, 4, 6, 8]:
    dss.Command(f"edit generator.PV{i} kw=0")

# Disable all storage
for name in ["Batt0", "Batt2", "BESS"]:
    dss.Command(f"edit storage.{name} state=idling")

# Set loads to 5 kW each
for i in range(10):
    dss.Command(f"edit load.Load{i} kw=5 pf=0.95")

print("\nConfiguration: 10 loads × 5 kW = 50 kW total")
print("All PV = 0, All storage = idle")

dss.Solution.Solve()

# Get source power (use different method)
total_kw = dss.Circuit.TotalPower()[0]  # Index 0 is kW, 1 is kvar
print(f"\nTotal Circuit Power: {total_kw:.2f} kW")

# Method 2: Use substation power
dss.Meters.First()
if dss.Meters.Name():
    reg_kw = dss.Meters.RegisterValues()[0]
    print(f"Meter reading: {reg_kw:.2f} kWh")
    
# Method 3: Sum all load powers
total_load_kw = sum([dss.Loads.kW() for _ in range(dss.Loads.Count()) if dss.Loads.Next()])
print(f"Sum of load kW: {total_load_kw:.2f} kW")

print("="*70 + "\n")
