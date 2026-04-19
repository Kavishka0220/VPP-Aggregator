"""
Print Economic Benefits Analysis
Tests the trained VPP agent and displays detailed economic breakdown
Separates economics between aggregator and individual prosumer/consumer levels
"""

import numpy as np
import sys
import os
from datetime import datetime
from stable_baselines3 import PPO

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from vpp_env import UrbanVPPEnv

# ===== ECONOMIC PARAMETERS =====
BESS_CAPITAL_COST_LKR = 10_000_000  # 1 crore LKR
BESS_LIFESPAN_YEARS = 10
HOME_BATTERY_CAPITAL_COST_LKR = 70_000  # Per home battery (13.5 kWh @ ~5,000-10,000 per kWh) in LKR
HOME_BATTERY_LIFESPAN_YEARS = 10

# ===== TIME PERIOD DEFINITIONS =====
TIME_PERIODS = {
    'daytime': (6, 18),    # 6am-6pm
    'peak': (18, 23),      # 6pm-11pm
    'offpeak': (23, 6),    # 11pm-6am (next day)
}


def _print_energy_analysis(episode_metrics, home_batt_indices):
    """Print detailed energy analysis by time periods and return lines for report."""
    lines = [
        "=" * 100,
        "DETAILED ENERGY ANALYSIS BY TIME PERIODS".center(100),
        "=" * 100,
        "",
    ]
    
    print("\n".join(lines))
    
    # Total Solar Generation
    solar_lines = [
        "TOTAL SOLAR GENERATION:",
        f"   Daytime (6am-6pm):      {episode_metrics['solar_daytime']:>10.2f} kWh",
        f"   Peak (6pm-11pm):        {episode_metrics['solar_peak']:>10.2f} kWh",
        f"   Off-Peak (11pm-6am):    {episode_metrics['solar_offpeak']:>10.2f} kWh",
    ]
    total_solar = episode_metrics['solar_daytime'] + episode_metrics['solar_peak'] + episode_metrics['solar_offpeak']
    solar_lines.append(f"   TOTAL:                  {total_solar:>10.2f} kWh")
    solar_lines.append("")
    
    # Total Load Consumption
    load_lines = [
        "TOTAL LOAD CONSUMPTION:",
        f"   Daytime (6am-6pm):      {episode_metrics['load_daytime']:>10.2f} kWh",
        f"   Peak (6pm-11pm):        {episode_metrics['load_peak']:>10.2f} kWh",
        f"   Off-Peak (11pm-6am):    {episode_metrics['load_offpeak']:>10.2f} kWh",
    ]
    total_load = episode_metrics['load_daytime'] + episode_metrics['load_peak'] + episode_metrics['load_offpeak']
    load_lines.append(f"   TOTAL:                  {total_load:>10.2f} kWh")
    load_lines.append("")
    
    # Grid Import/Export by Time Period
    grid_lines = [
        "GRID IMPORT/EXPORT BY TIME PERIOD:",
        "   DAYTIME (6am-6pm):",
        f"      Grid Import:        {episode_metrics['grid_import_daytime']:>10.2f} kWh",
        f"      Grid Export:        {episode_metrics['grid_export_daytime']:>10.2f} kWh",
        "   PEAK (6pm-11pm):",
        f"      Grid Import:        {episode_metrics['grid_import_peak']:>10.2f} kWh",
        f"      Grid Export:        {episode_metrics['grid_export_peak']:>10.2f} kWh",
        "   OFF-PEAK (11pm-6am):",
        f"      Grid Import:        {episode_metrics['grid_import_offpeak']:>10.2f} kWh",
        f"      Grid Export:        {episode_metrics['grid_export_offpeak']:>10.2f} kWh",
        "",
    ]
    
    # BESS Operations by Time Period
    bess_lines = [
        "BESS (Central Battery) BY TIME PERIOD:",
        "   DAYTIME (6am-6pm):",
        f"      Discharge:          {episode_metrics['bess_discharge_daytime']:>10.2f} kWh",
        f"      Charge:             {episode_metrics['bess_charge_daytime']:>10.2f} kWh",
        "   PEAK (6pm-11pm):",
        f"      Discharge:          {episode_metrics['bess_discharge_peak']:>10.2f} kWh",
        f"      Charge:             {episode_metrics['bess_charge_peak']:>10.2f} kWh",
        "   OFF-PEAK (11pm-6am):",
        f"      Discharge:          {episode_metrics['bess_discharge_offpeak']:>10.2f} kWh",
        f"      Charge:             {episode_metrics['bess_charge_offpeak']:>10.2f} kWh",
        "",
    ]
    
    # Home Batteries by Time Period and Node
    home_batt_lines = [
        "HOME BATTERIES BY TIME PERIOD AND NODE:",
    ]
    for idx, home_idx in enumerate(home_batt_indices):
        home_batt_lines.append(f"   HOME {home_idx} BATTERY:")
        home_batt_lines.append(f"      DAYTIME: Discharge={episode_metrics[f'home{home_idx}_discharge_daytime']:>8.2f} kWh | Charge={episode_metrics[f'home{home_idx}_charge_daytime']:>8.2f} kWh")
        home_batt_lines.append(f"      PEAK:    Discharge={episode_metrics[f'home{home_idx}_discharge_peak']:>8.2f} kWh | Charge={episode_metrics[f'home{home_idx}_charge_peak']:>8.2f} kWh")
        home_batt_lines.append(f"      OFFPEAK: Discharge={episode_metrics[f'home{home_idx}_discharge_offpeak']:>8.2f} kWh | Charge={episode_metrics[f'home{home_idx}_charge_offpeak']:>8.2f} kWh")
    home_batt_lines.append("")
    home_batt_lines.append("")
    
    all_lines = solar_lines + load_lines + grid_lines + bess_lines + home_batt_lines
    print("\n".join(all_lines))
    
    return lines + all_lines


def _print_detailed_episode_summary(episode_metrics, episode_num, total_episodes, step_count,
                                   home_batt_indices, solar_indices):
    """Print detailed episode summary with aggregator and prosumer economics and return lines for report."""
    
    report_lines = [f"Completed {step_count} steps\n"]
    print(f"Completed {step_count} steps\n")
    
    # ===== AGGREGATOR ECONOMICS =====
    print("=" * 100)
    print("AGGREGATOR ECONOMICS".center(100))
    print("=" * 100)
    print()
    
    report_lines.extend([
        "=" * 100,
        "AGGREGATOR ECONOMICS".center(100),
        "=" * 100,
        "",
    ])
    
    # BESS Operations
    bess_net = (episode_metrics['agg_bess_discharge_revenue'] - 
                episode_metrics['agg_bess_charge_cost'])
    daily_bess_amortized_cost = BESS_CAPITAL_COST_LKR / (BESS_LIFESPAN_YEARS * 365)
    
    bess_section = [
        "BESS OPERATIONS (Central Battery Energy Storage):",
        f"   Energy Discharged:     {episode_metrics['agg_bess_energy_discharged']:>10.2f} kWh",
        f"   Energy Charged:        {episode_metrics['agg_bess_energy_charged']:>10.2f} kWh",
        f"   Discharge Revenue:     {episode_metrics['agg_bess_discharge_revenue']:>10.2f} LKR",
        f"   Charge Cost:          -{episode_metrics['agg_bess_charge_cost']:>10.2f} LKR",
        f"   Net BESS Profit:       {bess_net:>10.2f} LKR",
        f"   Amortized BESS Cost*:  -{daily_bess_amortized_cost:>10.2f} LKR/day",
        "",
    ]
    
    for line in bess_section:
        print(line)
    report_lines.extend(bess_section)
    
    # Grid Operations
    grid_net = (episode_metrics['agg_grid_export_revenue'] - 
                episode_metrics['agg_grid_import_cost'])
    
    grid_section = [
        "GRID ARBITRAGE OPERATIONS:",
        f"   Export Revenue:        {episode_metrics['agg_grid_export_revenue']:>10.2f} LKR",
        f"   Import Cost:          -{episode_metrics['agg_grid_import_cost']:>10.2f} LKR",
        f"   Net Grid Profit:       {grid_net:>10.2f} LKR",
        "",
    ]
    
    for line in grid_section:
        print(line)
    report_lines.extend(grid_section)
    
    # Aggregator Total
    agg_total = bess_net + grid_net
    
    agg_total_section = [
        "AGGREGATOR TOTAL (Before Capital Costs):",
        f"   Net Profit:            {agg_total:>10.2f} LKR",
        "",
    ]
    
    for line in agg_total_section:
        print(line)
    report_lines.extend(agg_total_section)
    
    # ===== PROSUMER/CONSUMER ECONOMICS =====
    print("=" * 100)
    print("PROSUMER/CONSUMER ECONOMICS (Individual Homes)".center(100))
    print("=" * 100)
    print()
    
    report_lines.extend([
        "=" * 100,
        "PROSUMER/CONSUMER ECONOMICS (Individual Homes)".center(100),
        "=" * 100,
        "",
    ])
    
    # Summary statistics
    total_home_solar = np.sum(episode_metrics['home_solar_generation'])
    total_home_load = np.sum(episode_metrics['home_load_consumption'])
    total_home_battery_discharge = np.sum(episode_metrics['home_batt_energy_discharged'])
    total_home_battery_charge = np.sum(episode_metrics['home_batt_energy_charged'])
    total_home_grid_export = np.sum(episode_metrics['home_grid_export'])
    total_home_grid_import = np.sum(episode_metrics['home_grid_import'])
    total_home_export_revenue = np.sum(episode_metrics['home_grid_export_revenue'])
    total_home_import_cost = np.sum(episode_metrics['home_grid_import_cost'])
    
    system_agg_section = [
        "SYSTEM-WIDE AGGREGATES:",
        f"   Total Solar Generation:   {total_home_solar:>10.2f} kWh",
        f"   Total Load Consumption:   {total_home_load:>10.2f} kWh",
        f"   Total Home Battery Discharge: {total_home_battery_discharge:>10.2f} kWh",
        f"   Total Home Battery Charge:    {total_home_battery_charge:>10.2f} kWh",
        f"   Total Grid Export:        {total_home_grid_export:>10.2f} kWh",
        f"   Total Grid Import:        {total_home_grid_import:>10.2f} kWh",
        "",
    ]
    
    for line in system_agg_section:
        print(line)
    report_lines.extend(system_agg_section)
    
    grid_benefits_section = [
        "GRID ECONOMIC BENEFITS:",
        f"   Total Export Revenue:     {total_home_export_revenue:>10.2f} LKR",
        f"   Total Import Cost:       -{total_home_import_cost:>10.2f} LKR",
    ]
    
    prosumer_grid_net = total_home_export_revenue - total_home_import_cost
    grid_benefits_section.append(f"   Net Grid Profit:          {prosumer_grid_net:>10.2f} LKR")
    grid_benefits_section.append("")
    
    for line in grid_benefits_section:
        print(line)
    report_lines.extend(grid_benefits_section)
    
    # Prosumers with batteries
    home_battery_amortized = HOME_BATTERY_CAPITAL_COST_LKR / (HOME_BATTERY_LIFESPAN_YEARS * 365)
    
    home_batt_header = [
        f"HOME BATTERIES (Nodes 3, 5 - Amortized Cost: {home_battery_amortized:.2f} LKR/day each):",
    ]
    print(f"HOME BATTERIES (Nodes 3, 5 - Amortized Cost: {home_battery_amortized:.2f} LKR/day each):")
    report_lines.extend(home_batt_header)
    
    for idx, home_idx in enumerate(home_batt_indices):
        batt_net = (episode_metrics['home_grid_export_revenue'][home_idx] - 
                   episode_metrics['home_grid_import_cost'][home_idx])
        line = (f"   Home {home_idx}: Battery Discharge={episode_metrics['home_batt_energy_discharged'][home_idx]:>8.2f} kWh  |  "
               f"Charge={episode_metrics['home_batt_energy_charged'][home_idx]:>8.2f} kWh")
        print(line)
        report_lines.append(line)
    print()
    report_lines.append("")
    
    # ===== TOP PERFORMERS =====
    print("=" * 100)
    print("TOP PERFORMERS".center(100))
    print("=" * 100)
    print()
    
    report_lines.extend([
        "=" * 100,
        "TOP PERFORMERS".center(100),
        "=" * 100,
        "",
    ])
    
    # Best solar generators
    top_solar_idx = np.argsort(episode_metrics['home_solar_generation'])[-3:][::-1]
    
    print("Top 3 Solar Generators:")
    report_lines.append("Top 3 Solar Generators:")
    
    for rank, idx in enumerate(top_solar_idx, 1):
        if episode_metrics['home_solar_generation'][idx] > 0:
            solar_gen = episode_metrics['home_solar_generation'][idx]
            solar_rev = episode_metrics['home_grid_export_revenue'][idx]
            line = f"   {rank}. Home {idx:2d}: {solar_gen:>8.2f} kWh  (Export Revenue: {solar_rev:>10.2f} LKR)"
            print(line)
            report_lines.append(line)
    
    print()
    report_lines.append("")
    
    # Best net beneficiaries
    home_net_benefits = (episode_metrics['home_grid_export_revenue'] - 
                        episode_metrics['home_grid_import_cost'])
    top_benefit_idx = np.argsort(home_net_benefits)[-3:][::-1]
    
    print("Top 3 Net Beneficiaries (Net Grid Profit):")
    report_lines.append("Top 3 Net Beneficiaries (Net Grid Profit):")
    
    for rank, idx in enumerate(top_benefit_idx, 1):
        net_ben = home_net_benefits[idx]
        if net_ben > 0:
            solar_gen = episode_metrics['home_solar_generation'][idx]
            line = f"   {rank}. Home {idx:2d}: {net_ben:>10.2f} LKR  (Solar Gen: {solar_gen:>8.2f} kWh)"
            print(line)
            report_lines.append(line)
    
    print()
    report_lines.append("")
    
    return report_lines


def _build_detailed_episode_report(episode_metrics, episode_num, total_episodes, step_count,
                                   home_batt_indices, solar_indices):
    """Build text report lines for detailed episode summary."""
    
    bess_net = (episode_metrics['agg_bess_discharge_revenue'] - 
                episode_metrics['agg_bess_charge_cost'])
    grid_net = (episode_metrics['agg_grid_export_revenue'] - 
                episode_metrics['agg_grid_import_cost'])
    agg_total = bess_net + grid_net
    
    daily_bess_amortized_cost = BESS_CAPITAL_COST_LKR / (BESS_LIFESPAN_YEARS * 365)
    home_battery_amortized = HOME_BATTERY_CAPITAL_COST_LKR / (HOME_BATTERY_LIFESPAN_YEARS * 365)
    
    lines = [
        "",
        "─" * 100,
        f"EPISODE {episode_num}/{total_episodes}",
        "─" * 100,
        "",
        f"Completed {step_count} steps",
        "",
        "=" * 100,
        "AGGREGATOR ECONOMICS",
        "=" * 100,
        f"BESS Discharge Revenue:  {episode_metrics['agg_bess_discharge_revenue']:>10.2f} LKR",
        f"BESS Charge Cost:       -{episode_metrics['agg_bess_charge_cost']:>10.2f} LKR",
        f"Net BESS Profit:         {bess_net:>10.2f} LKR",
        f"Amortized BESS Cost:    -{daily_bess_amortized_cost:>10.2f} LKR/day",
        "",
        f"Grid Export Revenue:     {episode_metrics['agg_grid_export_revenue']:>10.2f} LKR",
        f"Grid Import Cost:       -{episode_metrics['agg_grid_import_cost']:>10.2f} LKR",
        f"Net Grid Profit:         {grid_net:>10.2f} LKR",
        "",
        f"Aggregator Total (Before Capital Costs): {agg_total:>10.2f} LKR",
        "",
        "=" * 100,
        "PROSUMER/CONSUMER ECONOMICS",
        "=" * 100,
        f"Total Solar Generation:  {np.sum(episode_metrics['home_solar_generation']):>10.2f} kWh",
        f"Total Load Consumption:  {np.sum(episode_metrics['home_load_consumption']):>10.2f} kWh",
        f"Total Grid Export:       {np.sum(episode_metrics['home_grid_export']):>10.2f} kWh",
        f"Total Grid Import:       {np.sum(episode_metrics['home_grid_import']):>10.2f} kWh",
        f"Total Export Revenue:    {np.sum(episode_metrics['home_grid_export_revenue']):>10.2f} LKR",
        f"Total Import Cost:      -{np.sum(episode_metrics['home_grid_import_cost']):>10.2f} LKR",
        f"Net Prosumer Profit:     {np.sum(episode_metrics['home_grid_export_revenue']) - np.sum(episode_metrics['home_grid_import_cost']):>10.2f} LKR",
        "",
        f"Home Battery Amortized Cost (each): {home_battery_amortized:.2f} LKR/day",
        "",
    ]
    
    return lines


def _write_summary_report(report_lines, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def print_economics(model_path="checkpoints/best_model/best_model", num_episodes=1, scenario_name="night_peak"):
    """
    Test the trained model and print economic benefits breakdown.
    Separates benefits between aggregator and individual prosumers/consumers.
    
    Args:
        model_path: Path to trained model (without .zip extension)
        num_episodes: Number of episodes to run
        scenario_name: Scenario to use for the environment
    """
    
    print("=" * 100)
    print("VPP AGGREGATOR - COMPREHENSIVE ECONOMIC BENEFITS ANALYSIS".center(100))
    print("(Aggregator vs. Individual Prosumer/Consumer Economics)".center(100))
    print("=" * 100)

    output_dir = os.path.join(os.path.dirname(current_dir), "results_plots")
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "economics_summary_detailed.txt")
    report_lines = [
        "=" * 100,
        "VPP AGGREGATOR - COMPREHENSIVE ECONOMIC BENEFITS ANALYSIS".center(100),
        "(Aggregator vs. Individual Prosumer/Consumer Economics)".center(100),
        "=" * 100,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model path: {model_path}",
        f"Scenario: {scenario_name}",
        f"Episodes: {num_episodes}",
        f"BESS Capital Cost (1 core rupees): {BESS_CAPITAL_COST_LKR:,.0f} LKR",
        f"BESS Lifespan: {BESS_LIFESPAN_YEARS} years",
        f"Home Battery Capital Cost: {HOME_BATTERY_CAPITAL_COST_LKR:,.0f} LKR (per home)",
        f"Home Battery Lifespan: {HOME_BATTERY_LIFESPAN_YEARS} years",
    ]
    
    # Load environment and model
    try:
        data_path = os.path.join(os.path.dirname(current_dir), "data")
        env = UrbanVPPEnv(data_path=data_path, scenario_name=scenario_name)
        
        # Resolve model path to absolute
        if not os.path.isabs(model_path):
            model_path = os.path.join(current_dir, model_path)
        
        model = PPO.load(model_path)
        print(f"[OK] Loaded model from: {model_path}\n")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        report_lines.append(f"[ERROR] Failed to load model: {e}")
        _write_summary_report(report_lines, summary_file)
        return
    
    # System configuration
    home_batt_indices = [3, 5]
    bess_index = 21
    solar_indices = [3, 5, 7, 10, 11, 13, 15, 17, 18, 19, 20]
    n_homes = 21
    
    # Run episodes and collect economics
    total_metrics = {
        # Aggregator metrics
        "agg_bess_discharge_revenue": 0.0,
        "agg_bess_charge_cost": 0.0,
        "agg_grid_export_revenue": 0.0,
        "agg_grid_import_cost": 0.0,
        "agg_bess_energy_discharged": 0.0,
        "agg_bess_energy_charged": 0.0,
        
        # Per-home prosumer metrics (21 homes)
        "home_solar_generation": np.zeros(n_homes),
        "home_load_consumption": np.zeros(n_homes),
        "home_batt_energy_discharged": np.zeros(n_homes),
        "home_batt_energy_charged": np.zeros(n_homes),
        "home_grid_export": np.zeros(n_homes),
        "home_grid_import": np.zeros(n_homes),
        
        # System metrics
        "total_reward": 0.0,
        "voltage_violations": 0,
    }
    
    for episode in range(num_episodes):
        print(f"\n{'=' * 100}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'=' * 100}\n")
        
        obs, _ = env.reset()
        done = False
        step_count = 0
        
        # Episode metrics
        episode_metrics = {
            # Aggregator level
            "agg_bess_discharge_revenue": 0.0,
            "agg_bess_charge_cost": 0.0,
            "agg_grid_export_revenue": 0.0,
            "agg_grid_import_cost": 0.0,
            "agg_bess_energy_discharged": 0.0,
            "agg_bess_energy_charged": 0.0,
            "agg_home_battery_revenue": 0.0,
            "agg_home_battery_cost": 0.0,
            
            # Per-home prosumer metrics (21 homes)
            "home_solar_generation": np.zeros(n_homes),
            "home_load_consumption": np.zeros(n_homes),
            "home_batt_energy_discharged": np.zeros(n_homes),
            "home_batt_energy_charged": np.zeros(n_homes),
            "home_grid_export": np.zeros(n_homes),
            "home_grid_import": np.zeros(n_homes),
            "home_grid_export_revenue": np.zeros(n_homes),
            "home_grid_import_cost": np.zeros(n_homes),
            
            # Time-period based energy tracking
            # Solar by period
            "solar_daytime": 0.0,
            "solar_peak": 0.0,
            "solar_offpeak": 0.0,
            
            # Load by period
            "load_daytime": 0.0,
            "load_peak": 0.0,
            "load_offpeak": 0.0,
            
            # Grid import/export by period
            "grid_import_daytime": 0.0,
            "grid_export_daytime": 0.0,
            "grid_import_peak": 0.0,
            "grid_export_peak": 0.0,
            "grid_import_offpeak": 0.0,
            "grid_export_offpeak": 0.0,
            
            # BESS operations by period
            "bess_discharge_daytime": 0.0,
            "bess_charge_daytime": 0.0,
            "bess_discharge_peak": 0.0,
            "bess_charge_peak": 0.0,
            "bess_discharge_offpeak": 0.0,
            "bess_charge_offpeak": 0.0,
            
            # Home batteries by period and node
            "home3_discharge_daytime": 0.0,
            "home3_charge_daytime": 0.0,
            "home3_discharge_peak": 0.0,
            "home3_charge_peak": 0.0,
            "home3_discharge_offpeak": 0.0,
            "home3_charge_offpeak": 0.0,
            
            "home5_discharge_daytime": 0.0,
            "home5_charge_daytime": 0.0,
            "home5_discharge_peak": 0.0,
            "home5_charge_peak": 0.0,
            "home5_discharge_offpeak": 0.0,
            "home5_charge_offpeak": 0.0,
            
            # System metrics
            "total_reward": 0.0,
            "voltage_violations": 0,
        }
        
        # Run episode and collect per-node data
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            # ===== EXTRACT PER-NODE DATA FROM ENVIRONMENT =====
            # Access environment's internal data
            total_solar = env.solar_episode[env.current_step - 1] if env.current_step > 0 else np.zeros(21)
            total_load = env.load_episode[env.current_step - 1] if env.current_step > 0 else np.zeros(21)
            batt_power = env.node_battery_power_kw  # Array of 22 values [home batteries + BESS]
            
            # Determine pricing for this step
            hour = info.get("hour", 12)
            if 6 <= hour < 18:         # Daytime
                buy_price, sell_price = 35, 19
                period = 'daytime'
            elif 18 <= hour < 23:      # Peak
                buy_price, sell_price = 67, 45
                period = 'peak'
            else:                      # Off-peak
                buy_price, sell_price = 21, 0
                period = 'offpeak'
            
            # ===== TRACK ENERGY BY TIME PERIOD =====
            # Total solar and load by period
            total_solar_step = np.sum(total_solar) * 0.25
            total_load_step = np.sum(total_load) * 0.25
            episode_metrics[f'solar_{period}'] += total_solar_step
            episode_metrics[f'load_{period}'] += total_load_step
            
            # Grid import/export by period
            grid_import_step = info.get("grid_import_kwh", 0.0) if "grid_import_kwh" in info else 0.0
            grid_export_step = info.get("grid_export_kwh", 0.0) if "grid_export_kwh" in info else 0.0
            
            # Calculate from net demand if not in info
            remaining_demand = info.get("remaining_demand", 0.0)
            if remaining_demand > 0:
                grid_import_step = remaining_demand * 0.25
            else:
                grid_export_step = abs(remaining_demand) * 0.25
            
            episode_metrics[f'grid_import_{period}'] += grid_import_step
            episode_metrics[f'grid_export_{period}'] += grid_export_step
            
            # BESS operations by period
            bess_power_val = info.get("bess_power", 0.0)
            if bess_power_val > 0:
                episode_metrics[f'bess_discharge_{period}'] += bess_power_val * 0.25
            else:
                episode_metrics[f'bess_charge_{period}'] += abs(bess_power_val) * 0.25
            
            # Home batteries by period and node
            for home_idx in home_batt_indices:
                battery_idx = home_batt_indices.index(home_idx)
                home_battery_power = batt_power[battery_idx]
                if home_battery_power > 0:
                    episode_metrics[f'home{home_idx}_discharge_{period}'] += home_battery_power * 0.25
                else:
                    episode_metrics[f'home{home_idx}_charge_{period}'] += abs(home_battery_power) * 0.25
            
            # ===== ACCUMULATE AGGREGATOR METRICS =====
            episode_metrics["agg_bess_discharge_revenue"] += info.get("bess_discharge_revenue", 0.0)
            episode_metrics["agg_bess_charge_cost"] += info.get("bess_charge_cost", 0.0)
            episode_metrics["agg_grid_export_revenue"] += info.get("grid_export_revenue", 0.0)
            episode_metrics["agg_grid_import_cost"] += info.get("grid_import_cost", 0.0)
            episode_metrics["total_reward"] += reward
            
            # BESS energy flows (15-min timestep = 0.25 hour)
            bess_power_val = info.get("bess_power", 0.0)
            if bess_power_val > 0:
                episode_metrics["agg_bess_energy_discharged"] += bess_power_val * 0.25
            else:
                episode_metrics["agg_bess_energy_charged"] += abs(bess_power_val) * 0.25
            
            if info.get("violation", 0) > 0:
                episode_metrics["voltage_violations"] += 1
            
            # ===== ACCUMULATE PER-HOME PROSUMER METRICS =====
            for node_idx in range(n_homes):
                # Solar generation
                solar_gen = total_solar[node_idx] * 0.25  # Convert to kWh
                episode_metrics["home_solar_generation"][node_idx] += solar_gen
                
                # Load consumption
                load_cons = total_load[node_idx] * 0.25  # Convert to kWh
                episode_metrics["home_load_consumption"][node_idx] += load_cons
                
                # Home battery operations (nodes 3 and 5)
                if node_idx in home_batt_indices:
                    battery_idx = home_batt_indices.index(node_idx)
                    home_battery_power = batt_power[battery_idx]
                    if home_battery_power > 0:
                        episode_metrics["home_batt_energy_discharged"][node_idx] += home_battery_power * 0.25
                    else:
                        episode_metrics["home_batt_energy_charged"][node_idx] += abs(home_battery_power) * 0.25
                
                # Home grid import/export
                # Net power = solar + battery - load
                net_power = solar_gen/0.25 + (batt_power[home_batt_indices.index(node_idx)] if node_idx in home_batt_indices else 0) - load_cons/0.25
                
                if net_power > 0:
                    export_kwh = net_power * 0.25
                    episode_metrics["home_grid_export"][node_idx] += export_kwh
                    episode_metrics["home_grid_export_revenue"][node_idx] += export_kwh * sell_price
                else:
                    import_kwh = abs(net_power) * 0.25
                    episode_metrics["home_grid_import"][node_idx] += import_kwh
                    episode_metrics["home_grid_import_cost"][node_idx] += import_kwh * buy_price
        
        # ===== PRINT EPISODE RESULTS =====
        energy_report = _print_energy_analysis(episode_metrics, home_batt_indices)
        report_lines.extend(energy_report)
        
        episode_report = _print_detailed_episode_summary(
            episode_metrics, episode + 1, num_episodes, step_count,
            home_batt_indices, solar_indices
        )
        report_lines.extend(episode_report)
        
        # Accumulate for multi-episode average
        total_metrics["agg_bess_discharge_revenue"] += episode_metrics["agg_bess_discharge_revenue"]
        total_metrics["agg_bess_charge_cost"] += episode_metrics["agg_bess_charge_cost"]
        total_metrics["agg_grid_export_revenue"] += episode_metrics["agg_grid_export_revenue"]
        total_metrics["agg_grid_import_cost"] += episode_metrics["agg_grid_import_cost"]
        total_metrics["agg_bess_energy_discharged"] += episode_metrics["agg_bess_energy_discharged"]
        total_metrics["agg_bess_energy_charged"] += episode_metrics["agg_bess_energy_charged"]
        total_metrics["home_solar_generation"] += episode_metrics["home_solar_generation"]
        total_metrics["home_load_consumption"] += episode_metrics["home_load_consumption"]
        total_metrics["home_batt_energy_discharged"] += episode_metrics["home_batt_energy_discharged"]
        total_metrics["home_batt_energy_charged"] += episode_metrics["home_batt_energy_charged"]
        total_metrics["home_grid_export"] += episode_metrics["home_grid_export"]
        total_metrics["home_grid_import"] += episode_metrics["home_grid_import"]
        total_metrics["total_reward"] += episode_metrics["total_reward"]
        total_metrics["voltage_violations"] += episode_metrics["voltage_violations"]
    
    # Print and save summary
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("   ANALYSIS COMPLETE")
    report_lines.append("=" * 100)
    _write_summary_report(report_lines, summary_file)
    print(f"[OK] Saved detailed summary to: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Print VPP Economic Benefits")
    parser.add_argument("--model", type=str, default="checkpoints/best_model/best_model",
                       help="Path to trained model (without .zip)")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to run")
    parser.add_argument("--scenario", type=str, default="night_peak",
                       help="Scenario name to use")
    
    args = parser.parse_args()
    
    print_economics(model_path=args.model, num_episodes=args.episodes, scenario_name=args.scenario)
