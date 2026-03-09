"""
Test script to run the VPP environment without battery actions
This simulates the feeder behavior with only loads and solar generation

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
    python test_no_batteries.py                      # Use default scenario
    python test_no_batteries.py Next_Day_Forecast_21 # Use specific scenario
"""
import numpy as np
import pandas as pd
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

def test_feeder_without_batteries(scenario_name=None, num_steps=96, verbose=True):
    """
    Run the environment for a full day with zero battery actions
    to observe feeder behavior without storage
    
    Args:
        scenario_name: Name of the scenario to test (None for default forecast)
        num_steps: Number of simulation steps (default 96 = 24 hours)
        verbose: Whether to print progress updates
    
    Returns:
        DataFrame containing simulation results
    """
    # Define output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, 'data_output')
    os.makedirs(data_dir, exist_ok=True)
    
    # Get absolute path to data directory
    workspace_root = os.path.dirname(parent_dir)
    vpp_data_path = os.path.join(workspace_root, 'data')
    
    # Initialize environment
    env = UrbanVPPEnv(data_path=vpp_data_path, scenario_name=scenario_name)
    
    # Reset environment
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
    
    if verbose:
        print("\n" + "="*80)
        scenario_title = f"{scenario_name}" if scenario_name else "Default Forecast"
        print(f"TESTING FEEDER WITHOUT BATTERIES - {scenario_title}")
        print("="*80)
    
    # Run simulation with zero battery actions
    for step in range(num_steps):
        # Zero action = no battery charging/discharging
        action = np.zeros(NUM_BATTERIES)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract data from info
        total_load = info.get('total_load', 0)
        total_solar = info.get('total_solar', 0)
        grid_power = info.get('grid_power', 0)
        
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
        
        # Print periodic updates
        if verbose and step % 24 == 0:  # Every 6 hours
            print(f"\nTime Step {step} (Hour {step * TIME_STEP_HOURS:.1f}):")
            print(f"  Load: {total_load:.2f} kW")
            print(f"  Solar: {total_solar:.2f} kW")
            print(f"  Grid Power: {grid_power:.2f} kW")
            print(f"  Min Voltage: {voltages.min():.3f} p.u.")
            print(f"  Max Voltage: {voltages.max():.3f} p.u.")
            if (voltages < VOLTAGE_LOWER_LIMIT).any():
                print(f"  ⚠️ Undervoltage detected!")
            if (voltages > VOLTAGE_UPPER_LIMIT).any():
                print(f"  ⚠️ Overvoltage detected!")
        
        if terminated or truncated:
            if verbose:
                print(f"\nEpisode ended at step {step}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if verbose:
        _print_summary_statistics(df, scenario_name)
    
    # Save results
    output_file = f"no_battery_test_results{'_' + scenario_name if scenario_name else ''}.csv"
    output_path = os.path.join(data_dir, output_file)
    df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n✓ Results saved to: data_output/{output_file}")
    
    return df


def _print_summary_statistics(df, scenario_name=None):
    """Print comprehensive summary statistics"""
    print("\n" + "="*80)
    scenario_title = f" - {scenario_name}" if scenario_name else ""
    print(f"SUMMARY - Feeder Performance WITHOUT Batteries{scenario_title}")
    print("="*80)
    
    # Simulation statistics
    print(f"\n📊 Simulation Info:")
    print(f"  Total Time Steps:         {len(df)}")
    print(f"  Simulation Duration:      {len(df) * TIME_STEP_HOURS:.1f} hours")
    
    # Energy statistics
    print(f"\n⚡ Energy Statistics:")
    total_load_energy = df['total_load'].sum() * TIME_STEP_HOURS
    total_solar_energy = df['total_solar'].sum() * TIME_STEP_HOURS
    grid_import_energy = df['grid_import'].sum() * TIME_STEP_HOURS
    grid_export_energy = df['grid_export'].sum() * TIME_STEP_HOURS
    net_grid_energy = df['grid_power'].sum() * TIME_STEP_HOURS
    
    print(f"  Total Load Energy:        {total_load_energy:.2f} kWh")
    print(f"  Total Solar Energy:       {total_solar_energy:.2f} kWh")
    print(f"  Grid Import Energy:       {grid_import_energy:.2f} kWh")
    print(f"  Grid Export Energy:       {grid_export_energy:.2f} kWh")
    print(f"  Net Grid Energy:          {net_grid_energy:.2f} kWh")
    
    # Solar self-consumption
    if total_solar_energy > 0:
        solar_self_consumed = total_solar_energy - grid_export_energy
        self_consumption_rate = (solar_self_consumed / total_solar_energy) * 100
        print(f"  Solar Self-Consumption:   {self_consumption_rate:.1f}%")
    else:
        print(f"  Solar Self-Consumption:   N/A (no solar generation)")
    
    # Power statistics
    print(f"\n📈 Power Statistics:")
    print(f"  Average Load:             {df['total_load'].mean():.2f} kW")
    print(f"  Peak Load:                {df['total_load'].max():.2f} kW")
    print(f"  Min Load:                 {df['total_load'].min():.2f} kW")
    print(f"  Average Solar:            {df['total_solar'].mean():.2f} kW")
    print(f"  Peak Solar:               {df['total_solar'].max():.2f} kW")
    print(f"  Peak Grid Import:         {df['grid_import'].max():.2f} kW")
    print(f"  Peak Grid Export:         {df['grid_export'].max():.2f} kW")
    
    # Voltage statistics
    print(f"\n🔌 Voltage Statistics:")
    print(f"  Min Voltage:              {df['min_voltage'].min():.4f} p.u.")
    print(f"  Max Voltage:              {df['max_voltage'].max():.4f} p.u.")
    
    violation_steps = (df['voltage_violations'] > 0).sum()
    violation_pct = (violation_steps / len(df)) * 100
    print(f"  Voltage Violations:       {violation_steps} steps ({violation_pct:.1f}%)")
    
    undervoltage_count = (df['min_voltage'] < VOLTAGE_LOWER_LIMIT).sum()
    overvoltage_count = (df['max_voltage'] > VOLTAGE_UPPER_LIMIT).sum()
    
    if undervoltage_count > 0:
        print(f"  ⚠️ Undervoltage Events:    {undervoltage_count} steps ({undervoltage_count/len(df)*100:.1f}%)")
        print(f"     Minimum Voltage:       {df['min_voltage'].min():.4f} p.u.")
    
    if overvoltage_count > 0:
        print(f"  ⚠️ Overvoltage Events:     {overvoltage_count} steps ({overvoltage_count/len(df)*100:.1f}%)")
        print(f"     Maximum Voltage:       {df['max_voltage'].max():.4f} p.u.")
    
    # Economic metrics (estimated)
    print(f"\n💰 Economic Metrics (Estimated):")
    import_cost_rate = 0.30  # $/kWh
    export_revenue_rate = 0.10  # $/kWh
    
    grid_import_cost = grid_import_energy * import_cost_rate
    grid_export_revenue = grid_export_energy * export_revenue_rate
    net_cost = grid_import_cost - grid_export_revenue
    
    print(f"  Grid Import Cost:         ${grid_import_cost:.2f} (@ ${import_cost_rate}/kWh)")
    print(f"  Export Revenue:           ${grid_export_revenue:.2f} (@ ${export_revenue_rate}/kWh)")
    print(f"  Net Energy Cost:          ${net_cost:.2f}")
    
    # Performance summary
    print(f"\n📝 Performance Summary:")
    if violation_steps == 0:
        print(f"  ✅ No voltage violations detected")
    else:
        print(f"  ⚠️ Voltage violations present - battery control recommended")
    
    if grid_export_energy > 0:
        print(f"  ⚡ {grid_export_energy:.2f} kWh of solar energy exported (could be stored)")
    
    print("="*80)



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
        print("(Use: python test_no_batteries.py <scenario_name> to test other scenarios)")
    
    # Run the test
    df = test_feeder_without_batteries(scenario_name=scenario)
    
    print(f"\n{'='*80}")
    print("✓ Test completed successfully!")
    print(f"{'='*80}\n")

