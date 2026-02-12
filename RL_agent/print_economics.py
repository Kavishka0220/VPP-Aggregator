"""
Print Economic Benefits Analysis
Tests the trained VPP agent and displays detailed economic breakdown
"""

import numpy as np
import sys
import os
from stable_baselines3 import PPO

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from vpp_env import UrbanVPPEnv

def print_economics(model_path="checkpoints/best_model/best_model", num_episodes=1):
    """
    Test the trained model and print economic benefits breakdown.
    
    Args:
        model_path: Path to trained model (without .zip extension)
        num_episodes: Number of episodes to run
    """
    
    print("="*80)
    print("   VPP AGGREGATOR - ECONOMIC BENEFITS ANALYSIS")
    print("="*80)
    
    # Load environment and model
    try:
        data_path = os.path.join(os.path.dirname(current_dir), "data")
        env = UrbanVPPEnv(data_path=data_path)
        
        # Resolve model path to absolute
        if not os.path.isabs(model_path):
            model_path = os.path.join(current_dir, model_path)
        
        model = PPO.load(model_path)
        print(f"[OK] Loaded model from: {model_path}\n")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Run episodes and collect economics
    total_metrics = {
        "grid_export_revenue": 0.0,
        "grid_import_cost": 0.0,
        "bess_discharge_revenue": 0.0,
        "bess_charge_cost": 0.0,
        "total_reward": 0.0,
        "voltage_violations": 0,
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
            
            if info.get("violation", 0) > 0:
                episode_metrics["voltage_violations"] += 1
        
        # Print episode results
        print(f"Completed {step_count} steps\n")
        
        print("┌" + "─"*78 + "┐")
        print("│" + " "*25 + "ECONOMIC BREAKDOWN (cents)" + " "*27 + "│")
        print("├" + "─"*78 + "┤")
        
        # Grid Economics
        print("│ GRID OPERATIONS:" + " "*61 + "│")
        print(f"│   Export Revenue:        {episode_metrics['grid_export_revenue']:>10.2f} cents" + " "*37 + "│")
        print(f"│   Import Cost:          -{episode_metrics['grid_import_cost']:>10.2f} cents" + " "*37 + "│")
        grid_net = episode_metrics['grid_export_revenue'] - episode_metrics['grid_import_cost']
        print(f"│   Net Grid Profit:       {grid_net:>10.2f} cents" + " "*37 + "│")
        
        print("│" + " "*78 + "│")
        
        # BESS Economics
        print("│ BESS OPERATIONS:" + " "*61 + "│")
        print(f"│   Discharge Revenue:     {episode_metrics['bess_discharge_revenue']:>10.2f} cents" + " "*37 + "│")
        print(f"│   Charge Cost:          -{episode_metrics['bess_charge_cost']:>10.2f} cents" + " "*37 + "│")
        bess_net = episode_metrics['bess_discharge_revenue'] - episode_metrics['bess_charge_cost']
        print(f"│   Net BESS Profit:       {bess_net:>10.2f} cents" + " "*37 + "│")
        
        print("│" + " "*78 + "│")
        
        # Total Economics
        total_revenue = episode_metrics['grid_export_revenue'] + episode_metrics['bess_discharge_revenue']
        total_cost = episode_metrics['grid_import_cost'] + episode_metrics['bess_charge_cost']
        net_profit = total_revenue - total_cost
        
        print("│ TOTAL SYSTEM:" + " "*64 + "│")
        print(f"│   Total Revenue:         {total_revenue:>10.2f} cents" + " "*37 + "│")
        print(f"│   Total Cost:           -{total_cost:>10.2f} cents" + " "*37 + "│")
        print(f"│   Net Profit:            {net_profit:>10.2f} cents  (${net_profit/100:.2f})" + " "*26 + "│")
        
        print("│" + " "*78 + "│")
        
        # Convert to dollars for daily/annual projections
        daily_profit = net_profit / 100  # cents to dollars
        annual_profit = daily_profit * 365
        
        print(f"│   Daily Profit:          ${daily_profit:>10.2f}" + " "*47 + "│")
        print(f"│   Annual Projection:     ${annual_profit:>10.2f}" + " "*47 + "│")
        
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
        
        avg_net_profit = (avg_metrics['grid_export_revenue'] + avg_metrics['bess_discharge_revenue'] 
                         - avg_metrics['grid_import_cost'] - avg_metrics['bess_charge_cost'])
        
        print(f"Average Net Profit:       {avg_net_profit:>10.2f} cents (${avg_net_profit/100:.2f})")
        print(f"Average Daily Profit:     ${avg_net_profit/100:>10.2f}")
        print(f"Average Annual Profit:    ${avg_net_profit/100*365:>10.2f}")
        print(f"Average Total Reward:     {avg_metrics['total_reward']:>10.2f}")
        print(f"Average Violations:       {avg_metrics['voltage_violations']:>10.1f} steps")
    
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
    
    args = parser.parse_args()
    
    print_economics(model_path=args.model, num_episodes=args.episodes)
