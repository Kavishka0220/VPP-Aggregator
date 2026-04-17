"""
Print Economic Benefits Analysis
Tests the trained VPP agent and displays detailed economic breakdown
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


def _box_line(text=""):
    text = str(text)
    if len(text) > 76:
        text = text[:76]
    return f"│ {text:<76}│"


def _episode_summary_lines(episode_metrics, episode_num, total_episodes, step_count):
    """Build pretty text summary lines for one episode."""
    self_consumption = episode_metrics["total_load_kwh"] - episode_metrics["grid_import_kwh"]
    self_consumption_rate = (
        (self_consumption / episode_metrics["total_load_kwh"] * 100)
        if episode_metrics["total_load_kwh"] > 0
        else 0
    )
    solar_utilization_rate = (
        ((episode_metrics["total_solar_kwh"] - episode_metrics["grid_export_kwh"]) / episode_metrics["total_solar_kwh"] * 100)
        if episode_metrics["total_solar_kwh"] > 0
        else 0
    )

    total_revenue = episode_metrics["grid_export_revenue"] + episode_metrics["bess_discharge_revenue"]
    total_cost = episode_metrics["grid_import_cost"] + episode_metrics["bess_charge_cost"]
    net_profit = total_revenue - total_cost
    annual_projection = net_profit * 365

    grid_net = episode_metrics["grid_export_revenue"] - episode_metrics["grid_import_cost"]
    bess_net = episode_metrics["bess_discharge_revenue"] - episode_metrics["bess_charge_cost"]
    net_bess_energy = episode_metrics["bess_discharge_kwh"] - episode_metrics["bess_charge_kwh"]

    lines = [
        "",
        "─" * 80,
        f"EPISODE {episode_num}/{total_episodes}",
        "─" * 80,
        "",
        f"Completed {step_count} steps",
        "",
        "┌" + "─" * 78 + "┐",
        _box_line("POWER & ENERGY SUMMARY".center(76)),
        "├" + "─" * 78 + "┤",
        _box_line("ENERGY FLOWS (kWh):"),
        _box_line(f"  Solar Generation:      {episode_metrics['total_solar_kwh']:>10.2f} kWh"),
        _box_line(f"  Load Consumption:      {episode_metrics['total_load_kwh']:>10.2f} kWh"),
        _box_line(f"  Grid Import:           {episode_metrics['grid_import_kwh']:>10.2f} kWh"),
        _box_line(f"  Grid Export:           {episode_metrics['grid_export_kwh']:>10.2f} kWh"),
        _box_line(),
        _box_line("BATTERY OPERATIONS (kWh):"),
        _box_line(f"  BESS Charged:          {episode_metrics['bess_charge_kwh']:>10.2f} kWh"),
        _box_line(f"  BESS Discharged:       {episode_metrics['bess_discharge_kwh']:>10.2f} kWh"),
        _box_line(f"  Net BESS Energy:       {net_bess_energy:>10.2f} kWh"),
        _box_line(f"  Solar Surplus:         {episode_metrics['solar_surplus_kwh']:>10.2f} kWh"),
        _box_line(),
        _box_line("SYSTEM EFFICIENCY:"),
        _box_line(f"  Self-Consumption:      {self_consumption_rate:>10.1f} %"),
        _box_line(f"  Solar Utilization:     {solar_utilization_rate:>10.1f} %"),
        "└" + "─" * 78 + "┘",
        "",
        "┌" + "─" * 78 + "┐",
        _box_line("ECONOMIC BREAKDOWN (Sri Lankan Rupees)".center(76)),
        "├" + "─" * 78 + "┤",
        _box_line("GRID OPERATIONS:"),
        _box_line(f"  Export Revenue:        {episode_metrics['grid_export_revenue']:>10.2f} LKR"),
        _box_line(f"  Import Cost:          -{episode_metrics['grid_import_cost']:>10.2f} LKR"),
        _box_line(f"  Net Grid Profit:       {grid_net:>10.2f} LKR"),
        _box_line(),
        _box_line("BESS OPERATIONS:"),
        _box_line(f"  Discharge Revenue:     {episode_metrics['bess_discharge_revenue']:>10.2f} LKR"),
        _box_line(f"  Charge Cost:          -{episode_metrics['bess_charge_cost']:>10.2f} LKR"),
        _box_line(f"  Net BESS Profit:       {bess_net:>10.2f} LKR"),
        _box_line(),
        _box_line("TOTAL SYSTEM:"),
        _box_line(f"  Total Revenue:         {total_revenue:>10.2f} LKR"),
        _box_line(f"  Total Cost:           -{total_cost:>10.2f} LKR"),
        _box_line(f"  Net Profit:            {net_profit:>10.2f} LKR"),
        _box_line(),
        _box_line(f"  Daily Profit:          {net_profit:>10.2f} LKR"),
        _box_line(f"  Annual Projection:     {annual_projection:>10.2f} LKR"),
        _box_line(),
        _box_line("PERFORMANCE METRICS:"),
        _box_line(f"  Total Reward:          {episode_metrics['total_reward']:>10.2f}"),
        _box_line(f"  Voltage Violations:    {int(episode_metrics['voltage_violations']):>10d} steps"),
        "└" + "─" * 78 + "┘",
        "",
    ]
    return lines


def _write_summary_report(report_lines, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

def print_economics(model_path="checkpoints/best_model/best_model", num_episodes=1, scenario_name="night_peak"):
    """
    Test the trained model and print economic benefits breakdown.
    
    Args:
        model_path: Path to trained model (without .zip extension)
        num_episodes: Number of episodes to run
        scenario_name: Scenario to use for the environment
    """
    
    print("="*80)
    print("   VPP AGGREGATOR - ECONOMIC BENEFITS ANALYSIS")
    print("="*80)

    output_dir = os.path.join(os.path.dirname(current_dir), "results_plots")
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "economics_summary.txt")
    report_lines = [
        "=" * 80,
        "   VPP AGGREGATOR - ECONOMIC BENEFITS ANALYSIS",
        "=" * 80,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model path: {model_path}",
        f"Scenario: {scenario_name}",
        f"Episodes: {num_episodes}",
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
    
    # Run episodes and collect economics
    total_metrics = {
        "grid_export_revenue": 0.0,
        "grid_import_cost": 0.0,
        "bess_discharge_revenue": 0.0,
        "bess_charge_cost": 0.0,
        "total_reward": 0.0,
        "voltage_violations": 0,
        # Power/Energy metrics
        "total_solar_kwh": 0.0,
        "total_load_kwh": 0.0,
        "grid_export_kwh": 0.0,
        "grid_import_kwh": 0.0,
        "bess_discharge_kwh": 0.0,
        "bess_charge_kwh": 0.0,
        "solar_surplus_kwh": 0.0,
    }
    
    for episode in range(num_episodes):
        print(f"\n{'─'*80}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'─'*80}\n")
        
        obs, _ = env.reset()
        done = False
        step_count = 0
        
        episode_metrics = {
            "grid_export_revenue": 0.0,
            "grid_import_cost": 0.0,
            "bess_discharge_revenue": 0.0,
            "bess_charge_cost": 0.0,
            "total_reward": 0.0,
            "voltage_violations": 0,
            # Power/Energy metrics
            "total_solar_kwh": 0.0,
            "total_load_kwh": 0.0,
            "grid_export_kwh": 0.0,
            "grid_import_kwh": 0.0,
            "bess_discharge_kwh": 0.0,
            "bess_charge_kwh": 0.0,
            "solar_surplus_kwh": 0.0,
        }
        
        # Run episode
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            # Accumulate metrics
            episode_metrics["grid_export_revenue"] += info.get("grid_export_revenue", 0.0)
            episode_metrics["grid_import_cost"] += info.get("grid_import_cost", 0.0)
            episode_metrics["bess_discharge_revenue"] += info.get("bess_discharge_revenue", 0.0)
            episode_metrics["bess_charge_cost"] += info.get("bess_charge_cost", 0.0)
            episode_metrics["total_reward"] += reward
            
            # Accumulate energy metrics (15-min timestep = 0.25 hour)
            episode_metrics["total_solar_kwh"] += info.get("total_solar", 0.0) * 0.25
            episode_metrics["total_load_kwh"] += info.get("total_load", 0.0) * 0.25
            
            # Calculate grid import/export from net_demand
            net_demand = info.get("remaining_demand", 0.0)
            if net_demand > 0:
                episode_metrics["grid_import_kwh"] += net_demand * 0.25
            else:
                episode_metrics["grid_export_kwh"] += abs(net_demand) * 0.25
            
            # BESS energy flows
            bess_power = info.get("bess_power", 0.0)
            if bess_power > 0:
                episode_metrics["bess_discharge_kwh"] += bess_power * 0.25
            else:
                episode_metrics["bess_charge_kwh"] += abs(bess_power) * 0.25
            
            episode_metrics["solar_surplus_kwh"] += max(0, info.get("solar_surplus", 0.0)) * 0.25
            
            if info.get("violation", 0) > 0:
                episode_metrics["voltage_violations"] += 1
        
        # Print episode results
        print(f"Completed {step_count} steps\n")
        
        # Calculate derived metrics
        self_consumption = episode_metrics["total_load_kwh"] - episode_metrics["grid_import_kwh"]
        self_consumption_rate = (self_consumption / episode_metrics["total_load_kwh"] * 100) if episode_metrics["total_load_kwh"] > 0 else 0
        solar_utilization_rate = ((episode_metrics["total_solar_kwh"] - episode_metrics["grid_export_kwh"]) / episode_metrics["total_solar_kwh"] * 100) if episode_metrics["total_solar_kwh"] > 0 else 0
        
        print("┌" + "─"*78 + "┐")
        print("│" + " "*27 + "POWER & ENERGY SUMMARY" + " "*29 + "│")
        print("├" + "─"*78 + "┤")
        
        # Energy Generation & Consumption
        print("│ ENERGY FLOWS (kWh):" + " "*57 + "│")
        print(f"│   Solar Generation:      {episode_metrics['total_solar_kwh']:>10.2f} kWh" + " "*38 + "│")
        print(f"│   Load Consumption:      {episode_metrics['total_load_kwh']:>10.2f} kWh" + " "*38 + "│")
        print(f"│   Grid Import:           {episode_metrics['grid_import_kwh']:>10.2f} kWh" + " "*38 + "│")
        print(f"│   Grid Export:           {episode_metrics['grid_export_kwh']:>10.2f} kWh" + " "*38 + "│")
        print("│" + " "*78 + "│")
        
        # Battery Operations
        print("│ BATTERY OPERATIONS (kWh):" + " "*51 + "│")
        print(f"│   BESS Charged:          {episode_metrics['bess_charge_kwh']:>10.2f} kWh" + " "*38 + "│")
        print(f"│   BESS Discharged:       {episode_metrics['bess_discharge_kwh']:>10.2f} kWh" + " "*38 + "│")
        net_bess_energy = episode_metrics['bess_discharge_kwh'] - episode_metrics['bess_charge_kwh']
        print(f"│   Net BESS Energy:       {net_bess_energy:>10.2f} kWh" + " "*38 + "│")
        print(f"│   Solar Surplus:         {episode_metrics['solar_surplus_kwh']:>10.2f} kWh" + " "*38 + "│")
        print("│" + " "*78 + "│")
        
        # System Efficiency
        print("│ SYSTEM EFFICIENCY:" + " "*58 + "│")
        print(f"│   Self-Consumption:      {self_consumption_rate:>10.1f} %" + " "*40 + "│")
        print(f"│   Solar Utilization:     {solar_utilization_rate:>10.1f} %" + " "*40 + "│")
        
        print("└" + "─"*78 + "┘")

        report_lines.extend(_episode_summary_lines(episode_metrics, episode + 1, num_episodes, step_count))
        print()
        print("┌" + "─"*78 + "┐")
        print("│" + " "*23 + "ECONOMIC BREAKDOWN (Sri Lankan Rupees)" + " "*17 + "│")
        print("├" + "─"*78 + "┤")
        
        # Grid Economics
        print("│ GRID OPERATIONS:" + " "*61 + "│")
        print(f"│   Export Revenue:        {episode_metrics['grid_export_revenue']:>10.2f} LKR" + " "*39 + "│")
        print(f"│   Import Cost:          -{episode_metrics['grid_import_cost']:>10.2f} LKR" + " "*39 + "│")
        grid_net = episode_metrics['grid_export_revenue'] - episode_metrics['grid_import_cost']
        print(f"│   Net Grid Profit:       {grid_net:>10.2f} LKR" + " "*39 + "│")
        
        print("│" + " "*78 + "│")
        
        # BESS Economics
        print("│ BESS OPERATIONS:" + " "*61 + "│")
        print(f"│   Discharge Revenue:     {episode_metrics['bess_discharge_revenue']:>10.2f} LKR" + " "*39 + "│")
        print(f"│   Charge Cost:          -{episode_metrics['bess_charge_cost']:>10.2f} LKR" + " "*39 + "│")
        bess_net = episode_metrics['bess_discharge_revenue'] - episode_metrics['bess_charge_cost']
        print(f"│   Net BESS Profit:       {bess_net:>10.2f} LKR" + " "*39 + "│")
        
        print("│" + " "*78 + "│")
        
        # Total Economics
        total_revenue = episode_metrics['grid_export_revenue'] + episode_metrics['bess_discharge_revenue']
        total_cost = episode_metrics['grid_import_cost'] + episode_metrics['bess_charge_cost']
        net_profit = total_revenue - total_cost
        
        print("│ TOTAL SYSTEM:" + " "*64 + "│")
        print(f"│   Total Revenue:         {total_revenue:>10.2f} LKR" + " "*39 + "│")
        print(f"│   Total Cost:           -{total_cost:>10.2f} LKR" + " "*39 + "│")
        print(f"│   Net Profit:            {net_profit:>10.2f} LKR" + " "*39 + "│")
        
        print("│" + " "*78 + "│")
        
        # Daily/annual projections in LKR
        daily_profit = net_profit  # Already in LKR per 15-min timestep
        annual_profit = daily_profit * 365
        
        print(f"│   Daily Profit:          {daily_profit:>10.2f} LKR" + " "*41 + "│")
        print(f"│   Annual Projection:     {annual_profit:>10.2f} LKR" + " "*41 + "│")
        
        print("│" + " "*78 + "│")
        
        # Performance Metrics
        print("│ PERFORMANCE METRICS:" + " "*57 + "│")
        print(f"│   Total Reward:          {episode_metrics['total_reward']:>10.2f}" + " "*47 + "│")
        print(f"│   Voltage Violations:    {episode_metrics['voltage_violations']:>10d} steps" + " "*43 + "│")
        
        print("└" + "─"*78 + "┘")
        
        # Accumulate for multi-episode average
        for key in total_metrics:
            total_metrics[key] += episode_metrics[key]
    
    # Print average if multiple episodes
    if num_episodes > 1:
        print(f"\n{'='*80}")
        print(f"AVERAGE OVER {num_episodes} EPISODES")
        print(f"{'='*80}\n")
        
        avg_metrics = {k: v/num_episodes for k, v in total_metrics.items()}
        
        # Calculate derived metrics
        avg_self_consumption = avg_metrics["total_load_kwh"] - avg_metrics["grid_import_kwh"]
        avg_self_consumption_rate = (avg_self_consumption / avg_metrics["total_load_kwh"] * 100) if avg_metrics["total_load_kwh"] > 0 else 0
        avg_solar_utilization_rate = ((avg_metrics["total_solar_kwh"] - avg_metrics["grid_export_kwh"]) / avg_metrics["total_solar_kwh"] * 100) if avg_metrics["total_solar_kwh"] > 0 else 0
        
        print("ENERGY METRICS:")
        print(f"  Solar Generation:       {avg_metrics['total_solar_kwh']:>10.2f} kWh")
        print(f"  Load Consumption:       {avg_metrics['total_load_kwh']:>10.2f} kWh")
        print(f"  Grid Import:            {avg_metrics['grid_import_kwh']:>10.2f} kWh")
        print(f"  Grid Export:            {avg_metrics['grid_export_kwh']:>10.2f} kWh")
        print(f"  BESS Charge/Discharge:  {avg_metrics['bess_charge_kwh']:>10.2f} / {avg_metrics['bess_discharge_kwh']:>10.2f} kWh")
        print(f"  Self-Consumption:       {avg_self_consumption_rate:>10.1f} %")
        print(f"  Solar Utilization:      {avg_solar_utilization_rate:>10.1f} %")
        print()
        
        avg_net_profit = (avg_metrics['grid_export_revenue'] + avg_metrics['bess_discharge_revenue'] 
                         - avg_metrics['grid_import_cost'] - avg_metrics['bess_charge_cost'])
        
        print("ECONOMIC METRICS:")
        print(f"  Average Net Profit:     {avg_net_profit:>10.2f} LKR")
        print(f"  Average Daily Profit:   {avg_net_profit:>10.2f} LKR")
        print(f"  Average Annual Profit:  {avg_net_profit*365:>10.2f} LKR")
        print(f"  Average Total Reward:   {avg_metrics['total_reward']:>10.2f}")
        print(f"  Average Violations:     {avg_metrics['voltage_violations']:>10.1f} steps")

        report_lines.extend([
            "",
            "=" * 80,
            f"AVERAGE OVER {num_episodes} EPISODES",
            "=" * 80,
            f"Average net profit: {avg_net_profit:.2f} LKR",
            f"Average annual profit: {avg_net_profit * 365:.2f} LKR",
            f"Average self-consumption: {avg_self_consumption_rate:.1f} %",
            f"Average solar utilization: {avg_solar_utilization_rate:.1f} %",
            f"Average total reward: {avg_metrics['total_reward']:.2f}",
            f"Average voltage violations: {avg_metrics['voltage_violations']:.1f} steps",
        ])

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("   ANALYSIS COMPLETE")
    report_lines.append("=" * 80)
    _write_summary_report(report_lines, summary_file)
    print(f"[OK] Saved summary to: {summary_file}")
    
    print("\n" + "="*80)
    print("   ANALYSIS COMPLETE")
    print("="*80)


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
