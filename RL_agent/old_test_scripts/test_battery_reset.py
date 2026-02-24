"""
Battery voltage impact test with proper circuit reset between scenarios
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openDSS.run_opendss import VPPDSSRunner
from pathlib import Path

dss_path = Path(r"D:\UoM\FYP\VPP-Aggregator\openDSS\feeder_houses.dss")

# Fixed scenario
loads_kw = [5.0] * 10
pv_kw = {0: 3.0, 1: 3.0, 2: 3.0, 4: 3.0, 6: 3.0, 8: 3.0}

print("\n" + "="*70)
print("BATTERY VOLTAGE IMPACT TEST (With Circuit Resets)")
print("="*70)

# Test 1: Baseline
print("\n--- SCENARIO 1: Baseline (No Batteries) ---")
runner1 = VPPDSSRunner(dss_path)  # Fresh instance
runner1.compile()
result1 = runner1.step(loads_kw, pv_kw, batt_home0_kw=0.0, batt_home2_kw=0.0, bess_kw=0.0, auto_compile=False)
print(f"Battery: Home0=0.0, Home2=0.0, BESS=0.0 kW")
print(f"Voltages: N0={result1.vmin_pu_by_bus[0]:.6f}, N2={result1.vmin_pu_by_bus[2]:.6f}, NBESS={result1.vmin_pu_by_bus[10]:.6f}")

# Test 2: Heavy Discharge  
print("\n--- SCENARIO 2: Heavy Discharge ---")
runner2 = VPPDSSRunner(dss_path)  # Fresh instance
runner2.compile()
result2 = runner2.step(loads_kw, pv_kw, batt_home0_kw=5.0, batt_home2_kw=5.0, bess_kw=40.0, auto_compile=False)
print(f"Battery: Home0=+5.0, Home2=+5.0, BESS=+40.0 kW (DISCHARGING)")
print(f"Voltages: N0={result2.vmin_pu_by_bus[0]:.6f}, N2={result2.vmin_pu_by_bus[2]:.6f}, NBESS={result2.vmin_pu_by_bus[10]:.6f}")

# Test 3: Heavy Charge
print("\n--- SCENARIO 3: Heavy Charging ---")
runner3 = VPPDSSRunner(dss_path)  # Fresh instance
runner3.compile()
result3 = runner3.step(loads_kw, pv_kw, batt_home0_kw=-5.0, batt_home2_kw=-5.0, bess_kw=-40.0, auto_compile=False)
print(f"Battery: Home0=-5.0, Home2=-5.0, BESS=-40.0 kW (CHARGING)")
print(f"Voltages: N0={result3.vmin_pu_by_bus[0]:.6f}, N2={result3.vmin_pu_by_bus[2]:.6f}, NBESS={result3.vmin_pu_by_bus[10]:.6f}")

# Analysis
print("\n" + "="*70)
print("VOLTAGE IMPACT SUMMARY")
print("="*70)

v0_base = result1.vmin_pu_by_bus[0]
v2_base = result1.vmin_pu_by_bus[2]
vbess_base = result1.vmin_pu_by_bus[10]

v0_dis = result2.vmin_pu_by_bus[0]
v2_dis = result2.vmin_pu_by_bus[2]
vbess_dis = result2.vmin_pu_by_bus[10]

v0_chg = result3.vmin_pu_by_bus[0]
v2_chg = result3.vmin_pu_by_bus[2]
vbess_chg = result3.vmin_pu_by_bus[10]

print(f"\nN0 (Home Battery 0):")
print(f"  Baseline:   {v0_base:.6f} p.u.")
print(f"  Discharge:  {v0_dis:.6f} p.u.  ({(v0_dis-v0_base)*1000:+.2f} mV)")
print(f"  Charge:     {v0_chg:.6f} p.u.  ({(v0_chg-v0_base)*1000:+.2f} mV)")

print(f"\nN2 (Home Battery 2):")
print(f"  Baseline:   {v2_base:.6f} p.u.")
print(f"  Discharge:  {v2_dis:.6f} p.u.  ({(v2_dis-v2_base)*1000:+.2f} mV)")
print(f"  Charge:     {v2_chg:.6f} p.u.  ({(v2_chg-v2_base)*1000:+.2f} mV)")

print(f"\nNBESS (BESS):")
print(f"  Baseline:   {vbess_base:.6f} p.u.")
print(f"  Discharge:  {vbess_dis:.6f} p.u.  ({(vbess_dis-vbess_base)*1000:+.2f} mV)")
print(f"  Charge:     {vbess_chg:.6f} p.u.  ({(vbess_chg-vbess_base)*1000:+.2f} mV)")

# Physics check
print("\n" + "="*70)
dis_increases = vbess_dis > vbess_base
chg_decreases = vbess_chg < vbess_base

if dis_increases and chg_decreases:
    print("✅ YES - BATTERIES PROPERLY AFFECT VOLTAGES:")
    print("   • Discharge (inject power) → INCREASES voltages ✓")
    print("   • Charge (consume power) → DECREASES voltages ✓")
    print("\n   This is correct distributed energy resource behavior!")
elif dis_increases:
    print("⚠️  PARTIAL - Discharge increases voltage (correct)")
    print("   But charge doesn't decrease voltage as expected")
elif chg_decreases:
    print("⚠️  PARTIAL - Charge decreases voltage (correct)")
    print("   But discharge doesn't increase voltage as expected")
else:
    print("❌ ISSUE - Batteries don't affect voltages correctly")
    print("   Check OpenDSS storage configuration")

print("="*70 + "\n")
