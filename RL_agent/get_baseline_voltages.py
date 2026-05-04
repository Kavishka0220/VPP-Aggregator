"""
Get baseline feeder voltages WITHOUT RL control.

This script captures the "natural" voltage profile when:
1. All batteries are IDLE (zero power)
2. Only load + solar generation exist
3. No RL agent control

Use this to:
- Compare against RL-controlled voltages
- See what violations exist in baseline
- Validate RL improvement
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
SCENARIO = "night_peak"  # Options: "night_peak", "daytime_peak", "solar_unavailable_day", "low_load_weekend", "cloudy_day", "intermittent_solar"
NUM_STEPS = 96           # 96 steps = 24 hours (15-min intervals)
START_INDEX = 0          # Start from timestep 0
# ============================================================================


def get_baseline_voltages(scenario_name=SCENARIO, start_idx=START_INDEX, num_steps=NUM_STEPS):
    """
    Run power flow without battery control and capture voltages.
    
    Args:
        scenario_name: Scenario folder (e.g., "night_peak", "daytime_peak")
        start_idx: Which timestep to start from (0 = 00:00)
        num_steps: How many 15-min steps to simulate (96 = full day)
    
    Returns:
        DataFrame with columns:
        - 'time': Time of day (HH:MM)
        - 'N0' to 'N20': Voltage at each house (p.u.)
        - 'NBESS': Voltage at BESS (p.u.)
        - 'min_voltage': Minimum across all nodes
        - 'max_voltage': Maximum across all nodes
    """
    
    # Load data
    data_path = Path(current_dir) / ".." / "data" / "forecast_scenarios"
    
    load_file = data_path / f"load_{scenario_name}.csv"
    solar_file = data_path / f"solar_{scenario_name}.csv"
    
    if not load_file.exists() or not solar_file.exists():
        print(f"❌ Scenario files not found in {data_path}")
        print(f"   Expected: {load_file.name}, {solar_file.name}")
        print(f"   Available files: {list(data_path.glob('*.csv'))}")
        return None
    
    print(f"\n{'='*70}")
    print(f"BASELINE VOLTAGE ANALYSIS - {scenario_name.upper()}")
    print(f"{'='*70}")
    print(f"📂 Load file: {load_file.name}")
    print(f"📂 Solar file: {solar_file.name}")
    
    # Read data
    load_data = pd.read_csv(load_file)
    solar_data = pd.read_csv(solar_file)
    
    print(f"✓ Loaded {len(load_data)} timesteps")
    
    # Initialize OpenDSS
    dss_file = Path(parent_dir) / "openDSS" / "feeder_houses.dss"
    dss_runner = VPPDSSRunner(dss_file)
    dss_runner.compile()
    print(f"✓ OpenDSS compiled")
    
    # Storage for results
    results = []
    
    print(f"\n🔄 Simulating {num_steps} timesteps (No battery control)...\n")
    
    for step in range(num_steps):
        actual_idx = start_idx + step
        if actual_idx >= len(load_data):
            print(f"⚠️  Reached end of data at step {step}")
            break
        
        # Extract loads (21 values, kW)
        # Handle both formats: some files have Timestamp column (22 cols), others don't (21 cols)
        if load_data.shape[1] == 22:
            loads = load_data.iloc[actual_idx, 1:22].values.astype(float)
        else:
            loads = load_data.iloc[actual_idx, :21].values.astype(float)
        
        # Extract solar (11 nodes with PV)
        # Similar handling for solar data format
        solar_indices = [3, 5, 7, 10, 11, 13, 15, 17, 18, 19, 20]
        pv_dict = {}
        solar_col_offset = 1 if solar_data.shape[1] == 22 else 0
        for pv_idx in solar_indices:
            col_idx = pv_idx + solar_col_offset
            if col_idx < len(solar_data.columns):
                pv_dict[pv_idx] = solar_data.iloc[actual_idx, col_idx]
        
        # BASELINE: All batteries IDLE (0 kW)
        dss_runner.set_loads(loads.tolist())
        dss_runner.set_pv_kw(pv_dict)
        dss_runner.set_storage_kw(0.0, 0.0, 0.0)  # Zero all batteries!
        
        # Solve power flow
        converged = dss_runner.solve()
        
        if not converged:
            print(f"⚠️  Step {step} ({actual_idx}): Power flow did not converge!")
            continue
        
        # Extract voltages at all nodes using get_bus_v_pu
        voltages_pu = []
        node_names = [f"N{i}" for i in range(21)] + ["NBESS"]
        for node_name in node_names:
            vmin, vabc = dss_runner.get_bus_v_pu(node_name)
            voltages_pu.append(vmin)
        
        # Calculate time of day
        hour = (actual_idx % 96) * 0.25  # Each step is 15 minutes
        minutes = int((hour % 1) * 60)
        hours = int(hour)
        time_str = f"{hours:02d}:{minutes:02d}"
        
        # Build row
        row = {"time": time_str}
        for node_idx, node_name in enumerate(node_names):
            row[node_name] = voltages_pu[node_idx]
        
        row["min_voltage"] = min(voltages_pu)
        row["max_voltage"] = max(voltages_pu)
        
        # Flag violations
        has_violation = row["min_voltage"] < 0.94 or row["max_voltage"] > 1.06
        violation_marker = "❌ VIOLATION" if has_violation else "✓"
        
        results.append(row)
        
        # Print progress every 6 steps (every 1.5 hours)
        if (step + 1) % 6 == 0:
            print(f"  Step {step+1:3d} ({time_str}): V_min={row['min_voltage']:.4f}p.u., "
                  f"V_max={row['max_voltage']:.4f}p.u.  {violation_marker}")
    
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
    
    # Find worst violations
    worst_idx = df_results['min_voltage'].idxmin()
    worst_row = df_results.iloc[worst_idx]
    print(f"\nWorst violation:")
    print(f"  Time: {worst_row['time']}")
    print(f"  Min voltage: {worst_row['min_voltage']:.4f} p.u.")
    
    # Find node with worst voltage
    node_names = [f"N{i}" for i in range(21)] + ["NBESS"]
    node_voltages = {node: worst_row[node] for node in node_names}
    worst_node = min(node_voltages, key=node_voltages.get)
    print(f"  Worst node: {worst_node} = {node_voltages[worst_node]:.4f} p.u.")
    
    # Save to CSV in results_plots/{scenario_name}/
    output_dir = Path(parent_dir) / "results_plots" / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"baseline_voltages_{scenario_name}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    
    return df_results


def plot_baseline_vs_scenario(scenario_name="night_peak"):
    """
    Plot baseline voltages with violation regions highlighted.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Get baseline
    df_baseline = get_baseline_voltages(scenario_name)
    if df_baseline is None:
        return
    
    # Extract node voltages
    node_names = [f"N{i}" for i in range(21)] + ["NBESS"]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot all nodes
    for node_name in node_names:
        ax.plot(df_baseline.index, df_baseline[node_name], alpha=0.6, label=node_name, linewidth=1)
    
    # Add violation bands
    ax.axhline(0.94, color='black', linestyle='--', linewidth=2, label='Min limit (0.94 p.u.)')
    ax.axhline(1.06, color='black', linestyle='--', linewidth=2, label='Max limit (1.06 p.u.)')
    
    # Plot min/max envelope
    #ax.fill_between(df_baseline.index, df_baseline['min_voltage'], df_baseline['max_voltage'], alpha=0.2, color='blue', label='Min-Max range')
    
    ax.set_xlabel("Timestep (15-min intervals)")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(f"Baseline Feeder Voltages - {scenario_name} (No Battery Control)")
    ax.legend(loc='best',bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.92, 1.08])
    
    # Save figure in results_plots/{scenario_name}/
    output_dir = Path(parent_dir) / "results_plots" / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_fig_png = output_dir / f"baseline_voltages_{scenario_name}.png"
    output_fig_pdf = output_dir / f"baseline_voltages_{scenario_name}.pdf"
    plt.tight_layout()
    plt.savefig(output_fig_png, dpi=150)
    plt.savefig(output_fig_pdf, bbox_inches='tight')
    print(f"\n📊 Saved plot to: {output_fig_png}")
    print(f"📊 Saved plot to: {output_fig_pdf}")
    plt.show()


def plot_node_comparison(scenario_name="night_peak", node1="NBESS", node2="N0"):
    """
    Plot comparison of two specific nodes with limit lines (like your image).
    
    Args:
        scenario_name: Scenario to analyze
        node1: First node to compare (e.g., "NBESS")
        node2: Second node to compare (e.g., "N0")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Load baseline
    output_dir = Path(parent_dir) / "results_plots" / scenario_name
    csv_file = output_dir / f"baseline_voltages_{scenario_name}.csv"
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # Convert timestep index to hours (each step is 15 minutes = 0.25 hours)
    hours = np.arange(len(df)) * 0.25
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the two nodes
    ax.plot(hours, df[node1], linewidth=2.5, label=f"Node 21 (BESS)", color='#0055CC')
    ax.plot(hours, df[node2], linewidth=2.5, label=f"Node 1", color='#5599FF')
    
    # Add limit lines
    ax.axhline(1.06, color='black', linestyle='--', linewidth=2, label='Upper Limit (1.06 p.u.)')
    ax.axhline(1.00, color='gray', linestyle=':', linewidth=2, label='Nominal (1.00 p.u.)')
    ax.axhline(0.94, color='black', linestyle='--', linewidth=2, label='Lower Limit (0.94 p.u.)')
    
    # Formatting
    ax.set_xlabel("Time (Hours)", fontsize=12)
    ax.set_ylabel("Voltage (p.u.)", fontsize=12)
    ax.set_title(f"Voltage Comparison: BESS and Node 1 - {scenario_name}", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.15])
    ax.set_xlim([0, 24])
    
    # Save figure in results_plots/{scenario_name}/
    output_fig_png = output_dir / f"voltage_comparison_{scenario_name}.png"
    output_fig_pdf = output_dir / f"voltage_comparison_{scenario_name}.pdf"
    plt.tight_layout()
    plt.savefig(output_fig_png, dpi=150, bbox_inches='tight')
    plt.savefig(output_fig_pdf, bbox_inches='tight')
    print(f"📊 Saved comparison plot to: {output_fig_png}")
    print(f"📊 Saved comparison plot to: {output_fig_pdf}")
    plt.show()


if __name__ == "__main__":
    # Use the SCENARIO variable defined at the top
    
    # Get baseline voltages
    df = get_baseline_voltages(scenario_name=SCENARIO, num_steps=NUM_STEPS)
    
    # Plot comparison with the style you want
    if df is not None:
        print("\n📊 Generating comparison plots...")
        plot_baseline_vs_scenario(SCENARIO)
        plot_node_comparison(scenario_name=SCENARIO, node1="NBESS", node2="N0")
