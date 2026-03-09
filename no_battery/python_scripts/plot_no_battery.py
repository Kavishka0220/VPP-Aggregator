"""
Visualization script for no-battery scenario
Generates plots similar to the RL agent results for comparison

Available Scenarios:
    - "cloudy_reduced_solar"        : Solar reduced with variability
    - "daytime_peak_load_day"       : Extra midday load bump
    - "evening_peak_load_day"       : Extra evening load bump
    - "heatwave_day"                : Higher daytime load + reduced solar
    - "intermittent_solar_dropouts" : Random solar dropouts
    - "load_higher_day"             : All loads scaled up
    - "Next_Day_Forecast_21"        : Load dip 00:00-02:00
    - "solar_shifted_late"          : Solar delayed (morning clouds)
    - "solar_unavailable_day"       : Solar set to 0 all day
    - "weekend_low_load"            : All loads scaled down

Usage:
    python plot_no_battery.py                      # Use default scenario
    python plot_no_battery.py Next_Day_Forecast_21 # Use specific scenario
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add RL_agent to path to import vpp_env
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'RL_agent'))
from vpp_env import UrbanVPPEnv

# Constants
VOLTAGE_LOWER_LIMIT = 0.9
VOLTAGE_UPPER_LIMIT = 1.1
TIME_STEP_HOURS = 0.25  # 15-minute intervals
NUM_NODES = 11
NUM_BATTERIES = 3
DEFAULT_NUM_STEPS = 96  # 24 hours

# Economic parameters
IMPORT_COST_RATE = 0.30  # $/kWh
EXPORT_REVENUE_RATE = 0.10  # $/kWh

# Plotting configuration
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 8

# Color scheme
COLORS = {
    'solar': '#FFA500',      # Orange
    'load': '#4169E1',       # Blue
    'grid_import': '#FF6B6B',  # Red
    'grid_export': '#51CF66',  # Green
    'voltage': '#9B59B6',    # Purple
}

def plot_no_battery_scenario(scenario_name=None, num_steps=DEFAULT_NUM_STEPS, save_plots=True, show_plots=True):
    """
    Run simulation without battery actions and create comprehensive plots
    
    Args:
        scenario_name: Name of the scenario to test (None for default forecast)
        num_steps: Number of simulation steps (default 96 = 24 hours)
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots interactively
    
    Returns:
        DataFrame containing simulation results
    """
    # Define output directories relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    plots_dir = os.path.join(parent_dir, 'plots')
    data_dir = os.path.join(parent_dir, 'data_output')
    docs_dir = os.path.join(parent_dir, 'documentation')
    
    # Get workspace root (two levels up from script)
    workspace_root = os.path.dirname(parent_dir)
    vpp_data_path = os.path.join(workspace_root, 'data')
    
    # Initialize environment with absolute path
    env = UrbanVPPEnv(data_path=vpp_data_path, scenario_name=scenario_name)
    obs, info = env.reset()
    
    # Storage for results
    results = {
        'step': [],
        'hour': [],
        'total_load': [],
        'total_solar': [],
        'net_demand': [],
        'grid_power': [],
        'grid_import': [],
        'grid_export': [],
        'voltage_violations': [],
        'min_voltage': [],
        'max_voltage': [],
        'reward': [],
    }
    
    # Add voltage columns for each node
    for i in range(NUM_NODES):
        results[f'voltage_node_{i}'] = []
    
    print("\n" + "="*80)
    scenario_title = f"{scenario_name}" if scenario_name else "Default Forecast"
    print(f"SIMULATING FEEDER WITHOUT BATTERIES - {scenario_title}")
    print("="*80)
    
    # Run simulation with zero battery actions
    for step in range(num_steps):
        # Zero action = no battery charging/discharging
        action = np.zeros(NUM_BATTERIES)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract data
        total_load = info.get('total_load', 0)
        total_solar = info.get('total_solar', 0)
        
        # Calculate grid power: Load - Solar (no batteries active)
        # Positive = importing from grid, Negative = exporting to grid
        grid_power = total_load - total_solar
        
        results['step'].append(step)
        results['hour'].append(step * TIME_STEP_HOURS)
        results['total_load'].append(total_load)
        results['total_solar'].append(total_solar)
        results['net_demand'].append(info.get('net_demand', 0))
        results['grid_power'].append(grid_power)
        results['grid_import'].append(max(0, grid_power))
        results['grid_export'].append(abs(min(0, grid_power)))
        results['reward'].append(reward)
        
        # Voltage information
        voltages = env.voltages
        results['min_voltage'].append(voltages.min())
        results['max_voltage'].append(voltages.max())
        
        # Count voltage violations
        violations = np.sum((voltages < VOLTAGE_LOWER_LIMIT) | (voltages > VOLTAGE_UPPER_LIMIT))
        results['voltage_violations'].append(violations)
        
        # Store individual node voltages
        for i in range(NUM_NODES):
            results[f'voltage_node_{i}'].append(voltages[i])
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directories
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    # Generate scenario suffix for filenames
    scenario_suffix = f"_{scenario_name}" if scenario_name else ""
    scenario_title_suffix = f" - {scenario_name}" if scenario_name else ""
    
    # Create the plots
    if save_plots or show_plots:
        _create_power_plots(df, scenario_title_suffix, plots_dir, scenario_suffix, save_plots)
        _create_voltage_plots(df, scenario_title_suffix, plots_dir, scenario_suffix, save_plots)
    
    # Print summary statistics
    _print_summary_statistics(df, scenario_name)
    
    # Save CSV
    csv_filename = f"no_battery_results{scenario_suffix}.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Data saved to: data_output/{csv_filename}")
    
    # Create README summary file
    _save_summary_file(df, scenario_name, scenario_suffix, docs_dir, 
                      f"no_battery_scenario{scenario_suffix}.png",
                      f"no_battery_voltages{scenario_suffix}.png",
                      csv_filename)
    
    print(f"\n{'='*80}")
    
    if show_plots:
        plt.show()
    
    return df


def _create_power_plots(df, scenario_title_suffix, plots_dir, scenario_suffix, save_plot):
    """Create and save power generation and grid exchange plots"""
    fig = plt.figure(figsize=(14, 6))
    
    # ==================== PLOT 1: Power Generation and Consumption ====================
    ax1 = plt.subplot(2, 1, 1)
    ax1.fill_between(df['hour'], 0, df['total_solar'], 
                     color=COLORS['solar'], alpha=0.6, label='Solar Generation')
    ax1.fill_between(df['hour'], 0, df['total_load'], 
                     color=COLORS['load'], alpha=0.4, label='Load Demand')
    ax1.plot(df['hour'], df['total_solar'], color=COLORS['solar'], linewidth=1.5, alpha=0.8)
    ax1.plot(df['hour'], df['total_load'], color=COLORS['load'], linewidth=1.5, alpha=0.8)
    
    ax1.set_ylabel('Power (kW)', fontsize=9, fontweight='bold')
    ax1.set_title(f'Power Generation and Consumption (No Battery){scenario_title_suffix}', 
                  fontsize=10, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    ax1.set_ylim(bottom=0)
    
    # ==================== PLOT 2: Grid Power Exchange ====================
    ax2 = plt.subplot(2, 1, 2)
    
    # Plot import (positive) and export (negative)
    import_mask = df['grid_power'] > 0
    export_mask = df['grid_power'] < 0
    
    ax2.fill_between(df['hour'], 0, df['grid_power'], 
                     where=import_mask, color=COLORS['grid_import'], 
                     alpha=0.6, label='Import from Grid', interpolate=True)
    ax2.fill_between(df['hour'], 0, df['grid_power'], 
                     where=export_mask, color=COLORS['grid_export'], 
                     alpha=0.6, label='Export to Grid', interpolate=True)
    ax2.plot(df['hour'], df['grid_power'], color='black', linewidth=1, alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax2.set_xlabel('Time (Hours)', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Power (kW)', fontsize=9, fontweight='bold')
    ax2.set_title(f'Grid Power Exchange (Positive = Import, Negative = Export){scenario_title_suffix}', 
                  fontsize=10, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 24)
    
    plt.subplots_adjust(left=0.058, bottom=0.094, right=0.86, top=0.92, wspace=0.2, hspace=0.353)
    
    # Save the plot
    if save_plot:
        plot_filename = f"no_battery_scenario{scenario_suffix}.png"
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: plots/{plot_filename}")


def _create_voltage_plots(df, scenario_title_suffix, plots_dir, scenario_suffix, save_plot):
    """Create and save voltage profile plots"""
    fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 5: All Node Voltages
    for i in range(NUM_NODES):
        ax5.plot(df['hour'], df[f'voltage_node_{i}'], 
                label=f'Node {i}', linewidth=1.5, alpha=0.7)
    
    ax5.axhline(y=VOLTAGE_UPPER_LIMIT, color='red', linestyle='--', linewidth=1.5, label='Upper Limit')
    ax5.axhline(y=VOLTAGE_LOWER_LIMIT, color='red', linestyle='--', linewidth=1.5, label='Lower Limit')
    ax5.axhline(y=1.0, color='green', linestyle=':', linewidth=1, alpha=0.5)
    
    ax5.set_ylabel('Voltage (p.u.)', fontsize=9, fontweight='bold')
    ax5.set_title(f'Voltage Profile - All Nodes (No Battery){scenario_title_suffix}', 
                  fontsize=10, fontweight='bold', pad=10)
    ax5.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1, framealpha=0.9, fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 24)
    ax5.set_ylim(0.85, 1.15)
    
    # Plot 6: Min/Max Voltage Range
    ax6.fill_between(df['hour'], df['min_voltage'], df['max_voltage'], 
                     color=COLORS['voltage'], alpha=0.3, label='Voltage Range')
    ax6.plot(df['hour'], df['min_voltage'], color=COLORS['voltage'], 
             linewidth=2, label='Minimum Voltage')
    ax6.plot(df['hour'], df['max_voltage'], color=COLORS['voltage'], 
             linewidth=2, linestyle='--', label='Maximum Voltage')
    
    ax6.axhline(y=VOLTAGE_UPPER_LIMIT, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax6.axhline(y=VOLTAGE_LOWER_LIMIT, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax6.axhline(y=1.0, color='green', linestyle=':', linewidth=1, alpha=0.5)
    
    ax6.set_xlabel('Time (Hours)', fontsize=9, fontweight='bold')
    ax6.set_ylabel('Voltage (p.u.)', fontsize=9, fontweight='bold')
    ax6.set_title(f'Voltage Range Across Network{scenario_title_suffix}', fontsize=10, fontweight='bold', pad=10)
    ax6.legend(loc='upper left', bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 24)
    ax6.set_ylim(0.85, 1.15)
    
    plt.subplots_adjust(left=0.058, bottom=0.094, right=0.865, top=0.92, wspace=0.2, hspace=0.353)
    
    if save_plot:
        voltage_plot_filename = f"no_battery_voltages{scenario_suffix}.png"
        voltage_plot_path = os.path.join(plots_dir, voltage_plot_filename)
        plt.savefig(voltage_plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Voltage plot saved to: plots/{voltage_plot_filename}")


def _print_summary_statistics(df, scenario_name=None):
    """Print comprehensive summary statistics"""
    print("\n" + "="*80)
    scenario_title = f" - {scenario_name}" if scenario_name else ""
    print(f"SUMMARY STATISTICS - No Battery Scenario{scenario_title}")
    print("="*80)
    
    # Calculate energy metrics
    total_load_energy = df['total_load'].sum() * TIME_STEP_HOURS
    total_solar_energy = df['total_solar'].sum() * TIME_STEP_HOURS
    grid_import_energy = df['grid_import'].sum() * TIME_STEP_HOURS
    grid_export_energy = df['grid_export'].sum() * TIME_STEP_HOURS
    net_grid_energy = df['grid_power'].sum() * TIME_STEP_HOURS
    
    print(f"\n📊 Energy Statistics:")
    print(f"  Total Load Energy:        {total_load_energy:.2f} kWh")
    print(f"  Total Solar Energy:       {total_solar_energy:.2f} kWh")
    print(f"  Grid Import Energy:       {grid_import_energy:.2f} kWh")
    print(f"  Grid Export Energy:       {grid_export_energy:.2f} kWh")
    print(f"  Net Grid Energy:          {net_grid_energy:.2f} kWh")
    
    # Calculate solar self-consumption safely
    if total_solar_energy > 0:
        solar_used = total_solar_energy - grid_export_energy
        self_consumption_pct = (solar_used / total_solar_energy) * 100
        print(f"  Solar Self-Consumption:   {self_consumption_pct:.1f}%")
    else:
        print(f"  Solar Self-Consumption:   N/A (no solar generation)")
    
    print(f"\n⚡ Power Statistics:")
    print(f"  Peak Load:                {df['total_load'].max():.2f} kW")
    print(f"  Peak Solar:               {df['total_solar'].max():.2f} kW")
    print(f"  Peak Grid Import:         {df['grid_import'].max():.2f} kW")
    print(f"  Peak Grid Export:         {df['grid_export'].max():.2f} kW")
    
    print(f"\n🔌 Voltage Statistics:")
    print(f"  Min Voltage:              {df['min_voltage'].min():.4f} p.u.")
    print(f"  Max Voltage:              {df['max_voltage'].max():.4f} p.u.")
    
    violation_steps = (df['voltage_violations'] > 0).sum()
    violation_pct = (violation_steps / len(df)) * 100
    print(f"  Voltage Violations:       {violation_steps} steps ({violation_pct:.1f}%)")
    
    undervoltage_count = (df['min_voltage'] < VOLTAGE_LOWER_LIMIT).sum()
    overvoltage_count = (df['max_voltage'] > VOLTAGE_UPPER_LIMIT).sum()
    
    if undervoltage_count > 0:
        print(f"  ⚠️ Undervoltage Events:    {undervoltage_count} steps")
    if overvoltage_count > 0:
        print(f"  ⚠️ Overvoltage Events:     {overvoltage_count} steps")
    
    print(f"\n💰 Economic Metrics (Estimated):")
    grid_cost = grid_import_energy * IMPORT_COST_RATE
    export_revenue = grid_export_energy * EXPORT_REVENUE_RATE
    net_cost = grid_cost - export_revenue
    print(f"  Grid Import Cost:         ${grid_cost:.2f} (@ ${IMPORT_COST_RATE}/kWh)")
    print(f"  Export Revenue:           ${export_revenue:.2f} (@ ${EXPORT_REVENUE_RATE}/kWh)")
    print(f"  Net Energy Cost:          ${net_cost:.2f}")


def _save_summary_file(df, scenario_name, scenario_suffix, docs_dir, plot_filename, voltage_plot_filename, csv_filename):
    """Save summary statistics to a text file"""
    # Calculate all metrics
    total_load_energy = df['total_load'].sum() * TIME_STEP_HOURS
    total_solar_energy = df['total_solar'].sum() * TIME_STEP_HOURS
    grid_import_energy = df['grid_import'].sum() * TIME_STEP_HOURS
    grid_export_energy = df['grid_export'].sum() * TIME_STEP_HOURS
    net_grid_energy = df['grid_power'].sum() * TIME_STEP_HOURS
    grid_cost = grid_import_energy * IMPORT_COST_RATE
    export_revenue = grid_export_energy * EXPORT_REVENUE_RATE
    net_cost = grid_cost - export_revenue
    violation_steps = (df['voltage_violations'] > 0).sum()
    violation_pct = (violation_steps / len(df)) * 100
    undervoltage_count = (df['min_voltage'] < VOLTAGE_LOWER_LIMIT).sum()
    overvoltage_count = (df['max_voltage'] > VOLTAGE_UPPER_LIMIT).sum()
    
    readme_path = os.path.join(docs_dir, f"SUMMARY{scenario_suffix}.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"NO-BATTERY SCENARIO ANALYSIS - {scenario_name if scenario_name else 'Default'}\n")
        f.write("="*80 + "\n\n")
        
        f.write("📊 ENERGY STATISTICS:\n")
        f.write(f"  Total Load Energy:        {total_load_energy:.2f} kWh\n")
        f.write(f"  Total Solar Energy:       {total_solar_energy:.2f} kWh\n")
        f.write(f"  Grid Import Energy:       {grid_import_energy:.2f} kWh\n")
        f.write(f"  Grid Export Energy:       {grid_export_energy:.2f} kWh\n")
        f.write(f"  Net Grid Energy:          {net_grid_energy:.2f} kWh\n")
        
        if total_solar_energy > 0:
            solar_used = total_solar_energy - grid_export_energy
            self_consumption_pct = (solar_used / total_solar_energy) * 100
            f.write(f"  Solar Self-Consumption:   {self_consumption_pct:.1f}%\n")
        else:
            f.write(f"  Solar Self-Consumption:   N/A (no solar generation)\n")
        
        f.write(f"\n⚡ POWER STATISTICS:\n")
        f.write(f"  Peak Load:                {df['total_load'].max():.2f} kW\n")
        f.write(f"  Peak Solar:               {df['total_solar'].max():.2f} kW\n")
        f.write(f"  Peak Grid Import:         {df['grid_import'].max():.2f} kW\n")
        f.write(f"  Peak Grid Export:         {df['grid_export'].max():.2f} kW\n")
        
        f.write(f"\n🔌 VOLTAGE STATISTICS:\n")
        f.write(f"  Min Voltage:              {df['min_voltage'].min():.4f} p.u.\n")
        f.write(f"  Max Voltage:              {df['max_voltage'].max():.4f} p.u.\n")
        f.write(f"  Voltage Violations:       {violation_steps} steps ({violation_pct:.1f}%)\n")
        
        if undervoltage_count > 0:
            f.write(f"  ⚠️ Undervoltage Events:    {undervoltage_count} steps\n")
        if overvoltage_count > 0:
            f.write(f"  ⚠️ Overvoltage Events:     {overvoltage_count} steps\n")
        
        f.write(f"\n💰 ECONOMIC METRICS (Estimated):\n")
        f.write(f"  Grid Import Cost:         ${grid_cost:.2f}\n")
        f.write(f"  Export Revenue:           ${export_revenue:.2f}\n")
        f.write(f"  Net Energy Cost:          ${net_cost:.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("\nFILES GENERATED:\n")
        f.write(f"  - {plot_filename}\n")
        f.write(f"  - {voltage_plot_filename}\n")
        f.write(f"  - {csv_filename}\n")
        f.write(f"  - SUMMARY{scenario_suffix}.txt (this file)\n")
    
    print(f"✓ Summary saved to: documentation/SUMMARY{scenario_suffix}.txt")


if __name__ == "__main__":
    # Default scenario (matches train.py)
    DEFAULT_SCENARIO = "weekend_low_load"
    
    # Allow command-line argument for scenario selection
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        print(f"Running scenario from command line: {scenario}")
    else:
        scenario = DEFAULT_SCENARIO
        print(f"Running default scenario: {scenario}")
        print("(Use: python plot_no_battery.py <scenario_name> to test other scenarios)")
    
    # Run the visualization
    df = plot_no_battery_scenario(scenario_name=scenario)
    
    print(f"\n{'='*80}")
    print("✓ Visualization completed successfully!")
    print(f"{'='*80}\n")
    