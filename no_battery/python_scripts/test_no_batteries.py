"""
Test script to run the VPP environment without battery actions
This simulates the feeder behavior with only loads and solar generation
"""
import numpy as np
import pandas as pd
import sys
import os

# Add RL_agent to path to import vpp_env
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'RL_agent'))
from vpp_env import UrbanVPPEnv

def test_feeder_without_batteries(scenario_name=None, num_steps=96):
    """
    Run the environment for a full day with zero battery actions
    to observe feeder behavior without storage
    """
    # Define output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize environment
    env = UrbanVPPEnv(data_path="../../data", scenario_name=scenario_name)
    
    # Reset environment
    obs, info = env.reset()
    
    # Storage for results
    results = {
        'step': [],
        'total_load': [],
        'total_solar': [],
        'net_demand': [],
        'grid_power': [],
        'voltage_violations': [],
        'min_voltage': [],
        'max_voltage': [],
    }
    
    # Add voltage columns for each node
    for i in range(11):
        results[f'voltage_node_{i}'] = []
    
    print("\n" + "="*60)
    print("TESTING FEEDER WITHOUT BATTERIES")
    print("="*60)
    
    # Run simulation with zero battery actions
    for step in range(num_steps):
        # Zero action = no battery charging/discharging
        action = np.zeros(3)  # [0, 0, 0] for all batteries
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract data from info
        results['step'].append(step)
        results['total_load'].append(info.get('total_load', 0))
        results['total_solar'].append(info.get('total_solar', 0))
        results['net_demand'].append(info.get('net_demand', 0))
        results['grid_power'].append(info.get('grid_power', 0))
        
        # Voltage information
        voltages = env.voltages
        results['min_voltage'].append(voltages.min())
        results['max_voltage'].append(voltages.max())
        results['voltage_violations'].append(
            np.sum((voltages < 0.9) | (voltages > 1.1))
        )
        
        for i in range(11):
            results[f'voltage_node_{i}'].append(voltages[i])
        
        # Print periodic updates
        if step % 24 == 0:  # Every 6 hours
            print(f"\nTime Step {step} (Hour {step/4:.1f}):")
            print(f"  Load: {info.get('total_load', 0):.2f} kW")
            print(f"  Solar: {info.get('total_solar', 0):.2f} kW")
            print(f"  Grid Power: {info.get('grid_power', 0):.2f} kW")
            print(f"  Min Voltage: {voltages.min():.3f} p.u.")
            print(f"  Max Voltage: {voltages.max():.3f} p.u.")
            if (voltages < 0.9).any():
                print(f"  ⚠️ Undervoltage detected!")
            if (voltages > 1.1).any():
                print(f"  ⚠️ Overvoltage detected!")
        
        if terminated or truncated:
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY - Feeder Performance WITHOUT Batteries")
    print("="*60)
    print(f"\nTotal Time Steps: {len(df)}")
    print(f"\nLoad Statistics:")
    print(f"  Average Load: {df['total_load'].mean():.2f} kW")
    print(f"  Peak Load: {df['total_load'].max():.2f} kW")
    print(f"  Min Load: {df['total_load'].min():.2f} kW")
    
    print(f"\nSolar Statistics:")
    print(f"  Average Solar: {df['total_solar'].mean():.2f} kW")
    print(f"  Peak Solar: {df['total_solar'].max():.2f} kW")
    
    print(f"\nGrid Power Statistics:")
    print(f"  Average Grid Power: {df['grid_power'].mean():.2f} kW")
    print(f"  Peak Grid Import: {df['grid_power'].max():.2f} kW")
    print(f"  Peak Grid Export: {df['grid_power'].min():.2f} kW")
    print(f"  Total Energy from Grid: {df['grid_power'].sum() * 0.25:.2f} kWh")
    
    print(f"\nVoltage Statistics:")
    print(f"  Min Voltage: {df['min_voltage'].min():.3f} p.u.")
    print(f"  Max Voltage: {df['max_voltage'].max():.3f} p.u.")
    print(f"  Steps with Voltage Violations: {(df['voltage_violations'] > 0).sum()}")
    print(f"  Percentage Time with Violations: {(df['voltage_violations'] > 0).sum()/len(df)*100:.1f}%")
    
    # Check voltage limits
    if df['min_voltage'].min() < 0.9:
        print(f"\n⚠️ WARNING: Undervoltage violations detected!")
        print(f"   Minimum voltage reached: {df['min_voltage'].min():.3f} p.u.")
    
    if df['max_voltage'].max() > 1.1:
        print(f"\n⚠️ WARNING: Overvoltage violations detected!")
        print(f"   Maximum voltage reached: {df['max_voltage'].max():.3f} p.u.")
    
    # Save results
    output_file = f"no_battery_test_results{'_' + scenario_name if scenario_name else ''}.csv"
    output_path = os.path.join(data_dir, output_file)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: data/{output_file}")
    
    return df

if __name__ == "__main__":
    # Test with default data
    print("Running test with default forecast data...")
    df = test_feeder_without_batteries()
    
    # Optionally test with a specific scenario
    # Uncomment to test specific scenarios:
    # df = test_feeder_without_batteries(scenario_name="heatwave_day")
    # df = test_feeder_without_batteries(scenario_name="solar_unavailable_day")
