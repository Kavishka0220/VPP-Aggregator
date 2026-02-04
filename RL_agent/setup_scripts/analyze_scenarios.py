import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Set style
plt.style.use('ggplot')
# plt.rcParams['figure.figsize'] = (12, 6)

def load_scenarios(data_dir):
    """
    Parses files in the directory and organizes them by scenario.
    Returns a dictionary: scenarios[scenario_name] = {'load': df, 'solar': df}
    """
    scenarios = {}
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    print(f"Searching for scenarios in: {os.path.abspath(data_dir)}")
    print(f"Found {len(files)} files.")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # files are named like 'load_scenario_name.csv' or 'solar_scenario_name.csv'
        if filename.startswith('load_'):
            type_key = 'load'
            # Only remove the prefix 'load_' (length 5)
            scenario_name = filename[5:].replace('.csv', '')
        elif filename.startswith('solar_'):
            type_key = 'solar'
            # Only remove the prefix 'solar_' (length 6)
            scenario_name = filename[6:].replace('.csv', '')
        else:
            continue
            
        # Read Data
        # Assuming no header for time, and 10 columns for houses
        df = pd.read_csv(file_path)
        
        # Create 15-min time index for 24 hours (96 steps)
        # Using a dummy date to make plotting easier
        times = pd.date_range(start='2024-01-01', periods=96, freq='15min')
        df.index = times
        
        if scenario_name not in scenarios:
            scenarios[scenario_name] = {}
        
        scenarios[scenario_name][type_key] = df
        
    return scenarios

def main():
    # Define path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 2 levels: RL_agent/setup_scripts -> RL_agent -> VPP-Aggregator
    # Then down into data/forecast_scenarios
    data_dir = os.path.join(script_dir, '..', '..', 'data', 'forecast_scenarios')

    scenarios_data = load_scenarios(data_dir)
    print(f"Loaded {len(scenarios_data)} scenarios: {list(scenarios_data.keys())}")

    # Calculate Aggregates (Sum of all houses)
    scenario_stats = []

    for name, data in scenarios_data.items():
        if 'load' in data and 'solar' in data:
            load_agg = data['load'].sum(axis=1) # Sum across columns (houses)
            solar_agg = data['solar'].sum(axis=1)
            net_load = load_agg - solar_agg
            
            scenarios_data[name]['load_agg'] = load_agg
            scenarios_data[name]['solar_agg'] = solar_agg
            scenarios_data[name]['net_load'] = net_load
            
            scenario_stats.append({
                'Scenario': name,
                'Total Daily Load (kWh)': load_agg.sum() / 4, # divide by 4 for 15-min intervals
                'Total Daily Solar (kWh)': solar_agg.sum() / 4,
                'Peak Load (kW)': load_agg.max(),
                'Peak Solar (kW)': solar_agg.max()
            })

    stats_df = pd.DataFrame(scenario_stats).set_index('Scenario')
    print("\n--- Scenario Statistics ---")
    print(stats_df)
    
    # 1. Ten Individual Figures (One per Scenario)
    # "give 10 figures for 10 scenarios"
    for name, data in sorted(scenarios_data.items()):
        if 'load_agg' in data and 'solar_agg' in data:
            plt.figure(figsize=(10, 6))
            plt.plot(data['load_agg'], label='Load', color='blue')
            plt.plot(data['solar_agg'], label='Solar', color='orange')
            plt.plot(data['net_load'], label='Net Load', color='green', linestyle='--')
            
            plt.title(f'Scenario Analysis: {name}')
            plt.ylabel('Power (kW)')
            plt.xlabel('Time')
            plt.legend()
            plt.tight_layout()

    # 2. One Figure for All Loads and Solar (Two Subplots)
    # "one figure for all loads and solar in 2 plots"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Subplot 1: Loads
    for name in scenarios_data:
        if 'load_agg' in scenarios_data[name]:
            ax1.plot(scenarios_data[name]['load_agg'], label=name, alpha=0.7)
    
    ax1.set_title('Aggregate Load Profiles (All Scenarios)')
    ax1.set_ylabel('Power (kW)')
    #ax1.set_xlabel('Time')
    # Smaller legend font and adjust position
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', borderaxespad=0.)

    # Subplot 2: Solar
    for name in scenarios_data:
        if 'solar_agg' in scenarios_data[name]:
            ax2.plot(scenarios_data[name]['solar_agg'], label=name, alpha=0.7)

    ax2.set_title('Aggregate Solar Profiles (All Scenarios)')
    ax2.set_ylabel('Power (kW)')
    ax2.set_xlabel('Time')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', borderaxespad=0.)
    plt.subplots_adjust(top=0.95, bottom=0.07, left=0.07, right=0.83, hspace=0.25)

    # Adjust layout to make room for legend on the right (rect=[left, bottom, right, top])

    plt.show()

    if len(scenarios_data) == 0:
        print("No complete scenarios found (requiring both load and solar files).")

if __name__ == "__main__":
    main()
