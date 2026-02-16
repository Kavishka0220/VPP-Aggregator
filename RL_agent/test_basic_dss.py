"""
Basic OpenDSS functionality test
"""
import opendssdirect as dss
from pathlib import Path

dss_path = Path(r"D:\UoM\FYP\VPP-Aggregator\openDSS\feeder_houses.dss")

print("\n" + "="*70)
print("Basic OpenDSS Test")
print("="*70)

dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')

# Check what elements exist
print("\nLoads:", len(dss.Loads.AllNames()), "->", dss.Loads.AllNames()[:3])
print("Generators:", len(dss.Generators.AllNames()), "->", dss.Generators.AllNames())
print("Storage:", len(dss.Storages.AllNames()), "->", dss.Storages.AllNames())

# Set one load
dss.Loads.First()
dss.Loads.kW(10)
print(f"\nFirst load set to: {dss.Loads.kW()} kW")

# Solve
dss.Solution.Solve()
print(f"Converged: {dss.Solution.Converged()}")

# Check losses
losses = dss.Circuit.Losses()
print(f"Losses: {losses[0]/1000:.2f} kW, {losses[1]/1000:.2f} kvar")

# Check total power
dss.Circuit.SetActiveElement("vsource.source")
powers = dss.CktElement.Powers()
print(f"Source powers (per phase, kW): {[p/1000 for p in powers[::2]]}")
print(f"Total source kW: {sum(powers[::2])/1000:.2f}")

print("="*70 + "\n")
