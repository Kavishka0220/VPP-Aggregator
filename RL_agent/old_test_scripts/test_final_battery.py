"""
Final test: Do batteries affect voltages?
Using correct OpenDSS power sign conventions
"""
import opendssdirect as dss
from pathlib import Path

dss_path = Path(r"D:\UoM\FYP\VPP-Aggregator\openDSS\feeder_houses.dss")

print("\n" + "="*70)
print("FINAL BATTERY VOLTAGE IMPACT TEST")
print("="*70)

# Test 1: Baseline
print("\n---  1: BASELINE (No Battery Action) ---")
dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')
for i in range(10):
    dss.Command(f"edit load.Load{i} kw=5 pf=0.95")
for i in [0, 1, 2, 4, 6, 8]:
    dss.Command(f"edit generator.PV{i} kw=0")
for name in ["Batt0", "Batt2", "BESS"]:
    dss.Command(f"edit storage.{name} state=idling")
dss.Solution.Solve()

power_base = dss.Circuit.TotalPower()[0]
dss.Circuit.SetActiveBus("NBESS")
v_base = min(dss.Bus.puVmagAngle()[::2])  # min phase voltage

print(f"Total power: {power_base:.2f} kW (negative = consuming)")
print(f"NBESS voltage: {v_base:.6f} p.u.")

# Test 2: Discharge
print("\n--- 2: DISCHARGE (BESS injecting 40 kW) ---")
dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')
for i in range(10):
    dss.Command(f"edit load.Load{i} kw=5 pf=0.95")
for i in [0, 1, 2, 4, 6, 8]:
    dss.Command(f"edit generator.PV{i} kw=0")
dss.Command("edit storage.Batt0 state=idling")
dss.Command("edit storage.Batt2 state=idling")
dss.Command("edit storage.BESS state=discharging kw=40")
dss.Solution.Solve()

power_discharge = dss.Circuit.TotalPower()[0]
dss.Circuit.SetActiveBus("NBESS")
v_discharge = min(dss.Bus.puVmagAngle()[::2])

print(f"Total power: {power_discharge:.2f} kW (should be ~10 kW less negative)")
print(f"NBESS voltage: {v_discharge:.6f} p.u.")
print(f"Power change: {power_discharge - power_base:.2f} kW (expect +40 kW)")

# Test 3: Charge
print("\n--- 3: CHARGING (BESS consuming 40 kW) ---")
dss.Basic.ClearAll()
dss.Command(f'compile "{dss_path}"')
for i in range(10):
    dss.Command(f"edit load.Load{i} kw=5 pf=0.95")
for i in [0, 1, 2, 4, 6, 8]:
    dss.Command(f"edit generator.PV{i} kw=0")
dss.Command("edit storage.Batt0 state=idling")
dss.Command("edit storage.Batt2 state=idling")
dss.Command("edit storage.BESS state=charging kw=40")
dss.Solution.Solve()

power_charge = dss.Circuit.TotalPower()[0]
dss.Circuit.SetActiveBus("NBESS")
v_charge = min(dss.Bus.puVmagAngle()[::2])

print(f"Total power: {power_charge:.2f} kW (should be ~40 kW more negative)")
print(f"NBESS voltage: {v_charge:.6f} p.u.")
print(f"Power change: {power_charge - power_base:.2f} kW (expect -40 kW)")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

print(f"\nPower Flow:")
print(f"  Discharge vs Base: {(power_discharge - power_base):+.2f} kW (expect +40)")
print(f"  Charge vs Base:    {(power_charge - power_base):+.2f} kW (expect -40)")

print(f"\nVoltages at NBESS:")
print(f"  Baseline:   {v_base:.6f} p.u.")
print(f"  Discharge:  {v_discharge:.6f} p.u.  ({(v_discharge-v_base)*1000:+.2f} mV)")
print(f"  Charge:     {v_charge:.6f} p.u.  ({(v_charge-v_base)*1000:+.2f} mV)")

power_ok = abs((power_discharge - power_base) - 40) < 5 and abs((power_charge - power_base) + 40) < 5
voltage_dis_ok = v_discharge > v_base
voltage_chg_ok = v_charge < v_base

print("\n" + "="*70)
if power_ok and voltage_dis_ok and voltage_chg_ok:
    print("✅ YES - BATTERIES PROPERLY AFFECT VOLTAGES!")
    print("   • Discharge (inject power) → INCREASES voltage ✓")
    print("   • Charge (consume power) → DECREASES voltage ✓")
    print("   • Power flow matches expected values ✓")
elif power_ok:
    print("⚠️  Power flow is correct, but voltage behavior unexpected")
elif voltage_dis_ok or voltage_chg_ok:
    print("⚠️  Voltage changes detected, but power flow doesn't match")  
else:
    print("❌ Batteries do NOT affect voltages or power flow correctly")
print("="*70 + "\n")
