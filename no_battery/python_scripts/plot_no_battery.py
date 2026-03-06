"""
Visualization script for no-battery scenario
Generates plots similar to the RL agent results for comparison
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

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

def plot_no_battery_scenario(scenario_name=None, num_steps=96):
    """
    Run simulation without battery actions and create comprehensive plots
    """
    # Define output directories relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    plots_dir = os.path.join(parent_dir, 'plots')
    data_dir = os.path.join(parent_dir, 'data')
    docs_dir = os.path.join(parent_dir, 'documentation')
    
    # Initialize environment
    env = UrbanVPPEnv(data_path="../../data", scenario_name=scenario_name)
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
    for i in range(11):
        results[f'voltage_node_{i}'] = []
    
    print("\n" + "="*80)
    print(f"SIMULATING FEEDER WITHOUT BATTERIES - {scenario_name if scenario_name else 'Default'}")
    print("="*80)
    
    # Run simulation with zero battery actions
    for step in range(num_steps):
        # Zero action = no battery charging/discharging
        action = np.zeros(3)  # [0, 0, 0] for all batteries
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract data
        total_load = info.get('total_load', 0)
        total_solar = info.get('total_solar', 0)
        
        # Calculate grid power: Load - Solar (no batteries active)
        # Positive = importing from grid, Negative = exporting to grid
        grid_power = total_load - total_solar
        
        results['step'].append(step)
        results['hour'].append(step * 0.25)  # 15-min intervals
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
        results['voltage_violations'].append(
            np.sum((voltages < 0.9) | (voltages > 1.1))
        )
        
        for i in range(11):
            results[f'voltage_node_{i}'].append(voltages[i])
        
        if terminated or truncated:
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directories
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    # Create the plots
    fig = plt.figure(figsize=(14, 9))
    
    # Color scheme
    color_solar = '#FFA500'  # Orange
    color_load = '#4169E1'   # Blue
    color_grid_import = '#FF6B6B'  # Red
    color_grid_export = '#51CF66'  # Green
    color_voltage = '#9B59B6'  # Purple
    
    # ==================== PLOT 1: Power Generation and Consumption ====================
    ax1 = plt.subplot(3, 1, 1)
    ax1.fill_between(df['hour'], 0, df['total_solar'], 
                     color=color_solar, alpha=0.6, label='Solar Generation')
    ax1.fill_between(df['hour'], 0, df['total_load'], 
                     color=color_load, alpha=0.4, label='Load Demand')
    ax1.plot(df['hour'], df['total_solar'], color=color_solar, linewidth=1.5, alpha=0.8)
    ax1.plot(df['hour'], df['total_load'], color=color_load, linewidth=1.5, alpha=0.8)
    
    ax1.set_ylabel('Power (kW)', fontsize=11, fontweight='bold')
    ax1.set_title('Power Generation and Consumption (No Battery)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    ax1.set_ylim(bottom=0)
    
    # ==================== PLOT 2: Grid Power Exchange ====================
    ax2 = plt.subplot(3, 1, 2)
    
    # Plot import (positive) and export (negative)
    # Import: when grid_power > 0
    # Export: when grid_power < 0
    import_mask = df['grid_power'] > 0
    export_mask = df['grid_power'] < 0
    
    ax2.fill_between(df['hour'], 0, df['grid_power'], 
                     where=import_mask, color=color_grid_import, 
                     alpha=0.6, label='Import from Grid', interpolate=True)
    ax2.fill_between(df['hour'], 0, df['grid_power'], 
                     where=export_mask, color=color_grid_export, 
                     alpha=0.6, label='Export to Grid', interpolate=True)
    ax2.plot(df['hour'], df['grid_power'], color='black', linewidth=1, alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax2.set_ylabel('Power (kW)', fontsize=11, fontweight='bold')
    ax2.set_title('Grid Power Exchange', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 24)

    # ==================== PLOT 3: Net Demand ====================
    ax3 = plt.subplot(3, 1, 3)
    
    ax3.plot(df['hour'], df['net_demand'], color='#E74C3C', linewidth=2, label='Net Demand (Load - Solar)')
    ax3.fill_between(df['hour'], 0, df['net_demand'], 
                     where=(df['net_demand'] > 0), color='#E74C3C', alpha=0.3, interpolate=True)
    ax3.fill_between(df['hour'], 0, df['net_demand'], 
                     where=(df['net_demand'] < 0), color='#2ECC71', alpha=0.3, interpolate=True)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax3.set_xlabel('Time (Hours)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Power (kW)', fontsize=11, fontweight='bold')
    ax3.set_title('Net Demand (Positive = Deficit, Negative = Surplus)', fontsize=13, fontweight='bold', pad=15)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 24)
    
    plt.tight_layout()
    
    # Save the plot
    scenario_suffix = f"_{scenario_name}" if scenario_name else ""
    plot_filename = f"no_battery_scenario{scenario_suffix}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: plots/{plot_filename}")
    
    # ==================== Additional Plot: Voltage Profiles ====================
    fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 5: All Node Voltages
    for i in range(11):
        ax5.plot(df['hour'], df[f'voltage_node_{i}'], 
                label=f'Node {i}', linewidth=1.5, alpha=0.7)
    
    ax5.axhline(y=1.1, color='red', linestyle='--', linewidth=1.5, label='Upper Limit')
    ax5.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, label='Lower Limit')
    ax5.axhline(y=1.0, color='green', linestyle=':', linewidth=1, alpha=0.5)
    
    ax5.set_ylabel('Voltage (p.u.)', fontsize=11, fontweight='bold')
    ax5.set_title('Voltage Profile - All Nodes (No Battery)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax5.legend(loc='right', bbox_to_anchor=(1.12, 0.5), ncol=1, framealpha=0.9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 24)
    ax5.set_ylim(0.85, 1.15)
    
    # Plot 6: Min/Max Voltage Range
    ax6.fill_between(df['hour'], df['min_voltage'], df['max_voltage'], 
                     color=color_voltage, alpha=0.3, label='Voltage Range')
    ax6.plot(df['hour'], df['min_voltage'], color=color_voltage, 
             linewidth=2, label='Minimum Voltage')
    ax6.plot(df['hour'], df['max_voltage'], color=color_voltage, 
             linewidth=2, linestyle='--', label='Maximum Voltage')
    
    ax6.axhline(y=1.1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax6.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax6.axhline(y=1.0, color='green', linestyle=':', linewidth=1, alpha=0.5)
    
    ax6.set_xlabel('Time (Hours)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Voltage (p.u.)', fontsize=11, fontweight='bold')
    ax6.set_title('Voltage Range Across Network', fontsize=13, fontweight='bold', pad=15)
    ax6.legend(loc='upper right', framealpha=0.9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 24)
    ax6.set_ylim(0.85, 1.15)
    
    plt.tight_layout()
    
    voltage_plot_filename = f"no_battery_voltages{scenario_suffix}.png"
    voltage_plot_path = os.path.join(plots_dir, voltage_plot_filename)
    plt.savefig(voltage_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Voltage plot saved to: plots/{voltage_plot_filename}")
    
    # Print Summary Statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS - No Battery Scenario")
    print("="*80)
    print(f"\n📊 Energy Statistics:")
    print(f"  Total Load Energy:        {df['total_load'].sum() * 0.25:.2f} kWh")
    print(f"  Total Solar Energy:       {df['total_solar'].sum() * 0.25:.2f} kWh")
    print(f"  Grid Import Energy:       {df['grid_import'].sum() * 0.25:.2f} kWh")
    print(f"  Grid Export Energy:       {df['grid_export'].sum() * 0.25:.2f} kWh")
    print(f"  Net Grid Energy:          {df['grid_power'].sum() * 0.25:.2f} kWh")
    
    # Calculate solar self-consumption safely
    total_solar_energy = df['total_solar'].sum()
    if total_solar_energy > 0:
        solar_used = (total_solar_energy - df['grid_export'].sum())
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
    print(f"  Voltage Violations:       {(df['voltage_violations'] > 0).sum()} steps ({(df['voltage_violations'] > 0).sum()/len(df)*100:.1f}%)")
    
    undervoltage_count = (df['min_voltage'] < 0.9).sum()
    overvoltage_count = (df['max_voltage'] > 1.1).sum()
    
    if undervoltage_count > 0:
        print(f"  ⚠️ Undervoltage Events:    {undervoltage_count} steps")
    if overvoltage_count > 0:
        print(f"  ⚠️ Overvoltage Events:     {overvoltage_count} steps")
    
    print(f"\n💰 Economic Metrics (Estimated):")
    grid_cost = df['grid_import'].sum() * 0.25 * 0.30  # Assume $0.30/kWh import
    export_revenue = df['grid_export'].sum() * 0.25 * 0.10  # Assume $0.10/kWh export
    net_cost = grid_cost - export_revenue
    print(f"  Grid Import Cost:         ${grid_cost:.2f}")
    print(f"  Export Revenue:           ${export_revenue:.2f}")
    print(f"  Net Energy Cost:          ${net_cost:.2f}")
    
    # Save CSV
    csv_filename = f"no_battery_results{scenario_suffix}.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Data saved to: data/{csv_filename}")
    
    # Create README summary file
    readme_path = os.path.join(docs_dir, f"SUMMARY{scenario_suffix}.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"NO-BATTERY SCENARIO ANALYSIS - {scenario_name if scenario_name else 'Default'}\n")
        f.write("="*80 + "\n\n")
        
        f.write("📊 ENERGY STATISTICS:\n")
        f.write(f"  Total Load Energy:        {df['total_load'].sum() * 0.25:.2f} kWh\n")
        f.write(f"  Total Solar Energy:       {df['total_solar'].sum() * 0.25:.2f} kWh\n")
        f.write(f"  Grid Import Energy:       {df['grid_import'].sum() * 0.25:.2f} kWh\n")
        f.write(f"  Grid Export Energy:       {df['grid_export'].sum() * 0.25:.2f} kWh\n")
        f.write(f"  Net Grid Energy:          {df['grid_power'].sum() * 0.25:.2f} kWh\n")
        
        total_solar_energy = df['total_solar'].sum()
        if total_solar_energy > 0:
            solar_used = (total_solar_energy - df['grid_export'].sum())
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
        f.write(f"  Voltage Violations:       {(df['voltage_violations'] > 0).sum()} steps ")
        f.write(f"({(df['voltage_violations'] > 0).sum()/len(df)*100:.1f}%)\n")
        
        undervoltage_count = (df['min_voltage'] < 0.9).sum()
        overvoltage_count = (df['max_voltage'] > 1.1).sum()
        
        if undervoltage_count > 0:
            f.write(f"  ⚠️ Undervoltage Events:    {undervoltage_count} steps\n")
        if overvoltage_count > 0:
            f.write(f"  ⚠️ Overvoltage Events:     {overvoltage_count} steps\n")
        
        f.write(f"\n💰 ECONOMIC METRICS (Estimated):\n")
        grid_cost = df['grid_import'].sum() * 0.25 * 0.30
        export_revenue = df['grid_export'].sum() * 0.25 * 0.10
        net_cost = grid_cost - export_revenue
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
    
    print("\n" + "="*80)
    
    plt.show()
    
    return df

if __name__ == "__main__":
    import sys
    
    # Allow command-line argument for scenario
    scenario = None
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        print(f"Running scenario: {scenario}")
    
    # Run the visualization
    df = plot_no_battery_scenario(scenario_name=scenario)
    
    print("\n📝 To run different scenarios, use:")
    print("   python plot_no_battery.py heatwave_day")
    print("   python plot_no_battery.py solar_unavailable_day")
    print("   python plot_no_battery.py weekend_low_load")
