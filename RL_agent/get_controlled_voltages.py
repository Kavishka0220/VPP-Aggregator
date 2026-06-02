"""
Get controlled feeder voltages WITH BESS discharge data.

This script:
1. Loads BESS power profile from RL agent results
2. Applies actual BESS discharge to the feeder
3. Calculates resulting voltages
4. Compares with baseline (no BESS)

Use this to see the ACTUAL impact of BESS control on voltage profiles.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add parent to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from openDSS.run_opendss import VPPDSSRunner


# ============================================================================
# ⚙️  CONFIGURATION - Change scenario here
# ============================================================================
SCENARIO = "night_peak"  # Options: "night_peak", "daytime_peak", "solar_unavailable_day", "low_load_weekend"
# ============================================================================


def get_controlled_voltages(scenario_name=SCENARIO, num_steps=96):
    """
    Run power flow WITH BESS discharge and capture voltages.
    
    Loads BESS power from detailed_simulation_results.csv
    
    Args:
        scenario_name: Scenario folder (e.g., "night_peak")
        num_steps: How many 15-min steps to simulate (96 = full day)
    
    Returns:
        DataFrame with columns:
        - 'time': Time of day (HH:MM)
        - 'N0' to 'N20': Voltage at each house (p.u.)
        - 'NBESS': Voltage at BESS (p.u.)
        - 'min_voltage': Minimum across all nodes
        - 'max_voltage': Maximum across all nodes
        - 'bess_power': BESS discharge power (kW)
    """
    
    # Load scenario data
    data_path = Path(current_dir) / ".." / "data" / "forecast_scenarios"
    load_file = data_path / f"load_{scenario_name}.csv"
    solar_file = data_path / f"solar_{scenario_name}.csv"
    
    if not load_file.exists() or not solar_file.exists():
        print(f"❌ Scenario files not found in {data_path}")
        return None
    
    # Load BESS power data from RL results
    results_dir = Path(parent_dir) / "results_plots" / scenario_name
    bess_file = results_dir / "detailed_simulation_results.csv"
    
    if not bess_file.exists():
        print(f"❌ BESS data file not found: {bess_file}")
        print(f"   Please run the RL agent first to generate: detailed_simulation_results.csv")
        return None
    
    print(f"\n{'='*70}")
    print(f"CONTROLLED VOLTAGE ANALYSIS - {scenario_name.upper()}")
    print(f"{'='*70}")
    print(f"📂 Load file: {load_file.name}")
    print(f"📂 Solar file: {solar_file.name}")
    print(f"📂 BESS data: {bess_file.name}")
    
    # Read data
    load_data = pd.read_csv(load_file)
    solar_data = pd.read_csv(solar_file)
    bess_data = pd.read_csv(bess_file)
    
    print(f"✓ Loaded {len(load_data)} timesteps (loads)")
    print(f"✓ Loaded {len(bess_data)} timesteps (BESS power)")
    
    # Initialize OpenDSS
    dss_file = Path(parent_dir) / "openDSS" / "feeder_houses.dss"
    dss_runner = VPPDSSRunner(dss_file)
    dss_runner.compile()
    print(f"✓ OpenDSS compiled")
    
    # Storage for results
    results = []
    
    print(f"\n🔄 Simulating {num_steps} timesteps (WITH BESS control)...\n")
    
    for step in range(num_steps):
        if step >= len(load_data) or step >= len(bess_data):
            print(f"⚠️  Reached end of data at step {step}")
            break
        
        # Extract loads (21 values, kW)
        # Handle both formats: some files have Timestamp column (22 cols), others don't (21 cols)
        if load_data.shape[1] == 22:
            loads = load_data.iloc[step, 1:22].values.astype(float)
        else:
            loads = load_data.iloc[step, :21].values.astype(float)
        
        # Extract solar (11 nodes with PV)
        # Similar handling for solar data format
        solar_indices = [3, 5, 7, 10, 11, 13, 15, 17, 18, 19, 20]
        pv_dict = {}
        solar_col_offset = 1 if solar_data.shape[1] == 22 else 0
        for pv_idx in solar_indices:
            col_idx = pv_idx + solar_col_offset
            if col_idx < len(solar_data.columns):
                pv_dict[pv_idx] = solar_data.iloc[step, col_idx]
        
        # WITH CONTROL: Get BESS power from RL results
        bess_power = bess_data.iloc[step]['BESS_Power_kW']  # Can be + (discharge) or - (charge)
        hb1_power = bess_data.iloc[step].get('HB1_Power_kW', 0.0)  # Home battery 1
        hb2_power = bess_data.iloc[step].get('HB2_Power_kW', 0.0)  # Home battery 2
        
        # Set actual controlled powers
        dss_runner.set_loads(loads.tolist())
        dss_runner.set_pv_kw(pv_dict)
        dss_runner.set_storage_kw(hb1_power, hb2_power, bess_power)
        
        # Solve power flow
        converged = dss_runner.solve()
        
        if not converged:
            print(f"⚠️  Step {step}: Power flow did not converge!")
            continue
        
        # Extract voltages at all nodes
        voltages_pu = []
        node_names = [f"N{i}" for i in range(21)] + ["NBESS"]
        for node_name in node_names:
            vmin, vabc = dss_runner.get_bus_v_pu(node_name)
            voltages_pu.append(vmin)
        
        # Calculate time of day
        hour = (step % 96) * 0.25
        minutes = int((hour % 1) * 60)
        hours = int(hour)
        time_str = f"{hours:02d}:{minutes:02d}"
        
        # Build row
        row = {"time": time_str}
        for node_idx, node_name in enumerate(node_names):
            row[node_name] = voltages_pu[node_idx]
        
        row["min_voltage"] = min(voltages_pu)
        row["max_voltage"] = max(voltages_pu)
        row["bess_power"] = bess_power
        row["hb1_power"] = hb1_power
        row["hb2_power"] = hb2_power
        
        # Flag violations
        has_violation = row["min_voltage"] < 0.94 or row["max_voltage"] > 1.06
        violation_marker = "❌ VIOLATION" if has_violation else "✓"
        
        results.append(row)
        
        # Print progress
        if (step + 1) % 6 == 0:
            print(f"  Step {step+1:3d} ({time_str}): V_min={row['min_voltage']:.4f}p.u., "
                  f"BESS={bess_power:+6.1f}kW  {violation_marker}")
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total timesteps: {len(df_results)}")
    print(f"Min voltage (overall): {df_results['min_voltage'].min():.4f} p.u.")
    print(f"Max voltage (overall): {df_results['max_voltage'].max():.4f} p.u.")
    
    # Count violations
    violations = (df_results['min_voltage'] < 0.94) | (df_results['max_voltage'] > 1.06)
    num_violations = violations.sum()
    print(f"Timesteps with violations: {num_violations} / {len(df_results)} ({100*num_violations/len(df_results):.1f}%)")
    
    if num_violations > 0:
        # Find worst violations
        worst_idx = df_results['min_voltage'].idxmin()
        worst_row = df_results.iloc[worst_idx]
        print(f"\nWorst violation:")
        print(f"  Time: {worst_row['time']}")
        print(f"  Min voltage: {worst_row['min_voltage']:.4f} p.u.")
        print(f"  BESS power: {worst_row['bess_power']:+.1f} kW")
    else:
        print(f"\n✅ NO VIOLATIONS - BESS control successful!")
    
    # Save to CSV
    output_dir = Path(parent_dir) / "results_plots" / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"controlled_voltages_{scenario_name}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    
    return df_results


def plot_comparison(scenario_name=SCENARIO):
    """
    Plot baseline vs controlled voltages side-by-side.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not available")
        return
    
    output_dir = Path(parent_dir) / "results_plots" / scenario_name
    
    # Load both datasets
    baseline_file = output_dir / f"baseline_voltages_{scenario_name}.csv"
    controlled_file = output_dir / f"controlled_voltages_{scenario_name}.csv"
    
    if not baseline_file.exists():
        print(f"❌ Baseline file not found: {baseline_file}")
        return
    if not controlled_file.exists():
        print(f"❌ Controlled file not found: {controlled_file}")
        return
    
    df_baseline = pd.read_csv(baseline_file)
    df_controlled = pd.read_csv(controlled_file)
    
    # Convert to hours
    hours = np.arange(len(df_baseline)) * 0.25
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot BESS voltages
    ax.plot(hours, df_baseline["NBESS"], linewidth=2.5, label="Without BESS", 
            color='#CC5500', linestyle='-', alpha=0.7)
    ax.plot(hours, df_controlled["NBESS"], linewidth=2.5, label="With BESS - Controlled", 
            color='#0055CC', linestyle='-', alpha=0.7)
    
    # Add limit lines
    ax.axhline(1.06, color='black', linestyle='--', linewidth=2, label='Upper Limit (1.06 p.u.)')
    ax.axhline(1.00, color='gray', linestyle=':', linewidth=2, label='Nominal (1.00 p.u.)')
    ax.axhline(0.94, color='black', linestyle='--', linewidth=2, label='Lower Limit (0.94 p.u.)')
    
    # Shade violation zones
    #ax.axhspan(0.85, 0.94, alpha=0.1, color='red', label='Undervoltage zone')
    #ax.axhspan(1.06, 1.15, alpha=0.1, color='orange', label='Overvoltage zone')
    
    # Formatting
    ax.set_xlabel("Time (Hours)", fontsize=22)
    ax.set_ylabel("Voltage (p.u.)", fontsize=22)
    #ax.set_title(f"BESS Voltage Control Impact - {scenario_name}", fontsize=25, fontweight='bold')
    ax.set_title(f"BESS Voltage Control Impact", fontsize=25, fontweight='bold')
    ax.legend(loc='best',fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.15])
    ax.set_xlim([0, 24])
    
    # Save
    output_fig_png = output_dir / f"voltage_control_comparison_{scenario_name}.png"
    output_fig_pdf = output_dir / f"voltage_control_comparison_{scenario_name}.pdf"
    plt.tight_layout()
    plt.savefig(output_fig_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_fig_pdf, bbox_inches='tight')
    print(f"📊 Saved comparison plot to: {output_fig_png}")
    print(f"📊 Saved comparison plot to: {output_fig_pdf}")
    #plt.show()


def plot_bess_power_and_voltage(scenario_name=SCENARIO):
    """
    Plot BESS discharge power on top and voltage on bottom (with same x-axis).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not available")
        return
    
    output_dir = Path(parent_dir) / "results_plots" / scenario_name
    controlled_file = output_dir / f"controlled_voltages_{scenario_name}.csv"
    
    if not controlled_file.exists():
        print(f"❌ File not found: {controlled_file}")
        return
    
    df = pd.read_csv(controlled_file)
    hours = np.arange(len(df)) * 0.25
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: BESS Power
    ax1.bar(hours, df["bess_power"], width=0.2, color='#0055CC', alpha=0.7, label='BESS Power')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel("Power (kW)", fontsize=20)
    ax1.set_title(f"BESS Discharge Control - {scenario_name}", fontsize=20, fontweight='bold')
    ax1.legend(loc='best', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: BESS Voltage
    ax2.plot(hours, df["NBESS"], linewidth=2.5, label='NBESS Voltage', color='#0055CC', marker='o', markersize=4)
    ax2.axhline(1.06, color='black', linestyle='--', linewidth=2, label='Upper Limit')
    ax2.axhline(1.00, color='gray', linestyle=':', linewidth=2, label='Nominal')
    ax2.axhline(0.94, color='black', linestyle='--', linewidth=2, label='Lower Limit')
    #ax2.axhspan(0.85, 0.94, alpha=0.1, color='red')
    ax2.set_ylabel("Voltage (p.u.)", fontsize=20)
    ax2.set_xlabel("Time (Hours)", fontsize=20)
    ax2.legend(loc='best', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.85, 1.15])
    
    # Save
    output_fig_png = output_dir / f"bess_power_and_voltage_{scenario_name}.png"
    output_fig_pdf = output_dir / f"bess_power_and_voltage_{scenario_name}.pdf"
    plt.tight_layout()
    plt.savefig(output_fig_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_fig_pdf, dpi=300, bbox_inches='tight')
    print(f"📊 Saved power-voltage plot to: {output_fig_png}")
    print(f"📊 Saved power-voltage plot to: {output_fig_pdf}")
    #plt.show()


if __name__ == "__main__":
    print("\n🔧 CONTROLLED VOLTAGE ANALYSIS")
    print(f"   Scenario: {SCENARIO}")
    print(f"   Mode: WITH BESS discharge (from RL results)\n")
    
    # Get controlled voltages
    df = get_controlled_voltages(scenario_name=SCENARIO, num_steps=96)
    
    if df is not None:
        print("\n📊 Generating comparison plots...")
        plot_comparison(SCENARIO)
        plot_bess_power_and_voltage(SCENARIO)
