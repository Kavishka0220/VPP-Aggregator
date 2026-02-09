import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Get script directory for absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))

def parse_tensorboard_logs(log_dir, tags=['rollout/ep_rew_mean', 'rollout/ep_len_mean', 'train/loss']):
    """
    Parse TensorBoard event files and extract training metrics.
    
    Args:
        log_dir: Directory containing tensorboard logs (e.g., './tensorboard_logs/PPO_42')
        tags: List of metric tags to extract
        
    Returns:
        Dictionary with metric data: {tag: {'steps': [], 'values': []}}
    """
    # Find the latest PPO run directory
    if os.path.isdir(log_dir):
        # Find event file in the directory
        event_files = list(Path(log_dir).rglob('events.out.tfevents.*'))
        if not event_files:
            print(f"[WARNING] No event files found in {log_dir}")
            return None
        
        # Use the most recent event file
        event_file = max(event_files, key=lambda p: p.stat().st_mtime)
        print(f"[INFO] Reading: {event_file}")
    else:
        print(f"[ERROR] Directory not found: {log_dir}")
        return None
    
    # Load event file
    ea = event_accumulator.EventAccumulator(str(event_file.parent))
    ea.Reload()
    
    # Extract data
    data = {}
    available_tags = ea.Tags()['scalars']
    print(f"[INFO] Available metrics: {available_tags}")
    
    for tag in tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = {'steps': steps, 'values': values}
            print(f"[OK] Extracted '{tag}': {len(values)} data points")
        else:
            print(f"[WARNING] Tag '{tag}' not found in logs")
    
    return data


def find_latest_run(base_dir):
    """Find the latest PPO training run directory."""
    ppo_dirs = sorted([d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith('PPO_')],
                      key=lambda p: int(p.name.split('_')[1]))
    if ppo_dirs:
        latest = ppo_dirs[-1]
        print(f"[INFO] Found latest training run: {latest}")
        return str(latest)
    return None


def plot_training_rewards(data, output_dir='./results_plots'):
    """
    Plot training metrics from parsed TensorBoard data.
    
    Args:
        data: Dictionary from parse_tensorboard_logs()
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which metrics are available
    has_reward = 'rollout/ep_rew_mean' in data
    has_length = 'rollout/ep_len_mean' in data
    has_loss = 'train/loss' in data
    
    # Create figure with subplots based on available data
    n_plots = sum([has_reward, has_length, has_loss])
    if n_plots == 0:
        print("[ERROR] No metrics to plot!")
        return
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot 1: Episode Reward Mean
    if has_reward:
        reward_data = data['rollout/ep_rew_mean']
        steps = np.array(reward_data['steps'])
        rewards = np.array(reward_data['values'])
        
        axes[plot_idx].plot(steps, rewards, color='#2E8B57', linewidth=1.5, alpha=0.8)
        axes[plot_idx].set_title('Mean Episode Reward During Training', fontsize=12, fontweight='bold')
        axes[plot_idx].set_ylabel('Mean Reward', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Add trend line
        if len(steps) > 10:
            z = np.polyfit(steps, rewards, 3)
            p = np.poly1d(z)
            axes[plot_idx].plot(steps, p(steps), "--", color='red', linewidth=2, alpha=0.7, label='Trend')
            axes[plot_idx].legend()
        
        # Add statistics
        max_reward = np.max(rewards)
        final_reward = rewards[-1]
        axes[plot_idx].axhline(y=max_reward, color='green', linestyle=':', linewidth=1, alpha=0.5)
        axes[plot_idx].text(0.02, 0.98, f'Max: {max_reward:.2f}\nFinal: {final_reward:.2f}', 
                           transform=axes[plot_idx].transAxes, fontsize=9,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plot_idx += 1
    
    # Plot 2: Episode Length
    if has_length:
        length_data = data['rollout/ep_len_mean']
        steps = np.array(length_data['steps'])
        lengths = np.array(length_data['values'])
        
        axes[plot_idx].plot(steps, lengths, color='#4169E1', linewidth=1.5, alpha=0.8)
        axes[plot_idx].set_title('Mean Episode Length During Training', fontsize=12, fontweight='bold')
        axes[plot_idx].set_ylabel('Episode Length (steps)', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Plot 3: Training Loss
    if has_loss:
        loss_data = data['train/loss']
        steps = np.array(loss_data['steps'])
        losses = np.array(loss_data['values'])
        
        axes[plot_idx].plot(steps, losses, color='#FF6347', linewidth=1.5, alpha=0.8)
        axes[plot_idx].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[plot_idx].set_ylabel('Loss', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Set x-label on bottom plot
    axes[-1].set_xlabel('Training Steps (Timesteps)', fontweight='bold')
    
    plt.tight_layout()
    output_file = f"{output_dir}/training_progress.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved training plot to '{output_file}'")
    
    return fig


def plot_multiple_runs(base_dir='./tensorboard_logs', output_dir='./results_plots', max_runs=5):
    """
    Plot and compare multiple training runs.
    
    Args:
        base_dir: Directory containing PPO_X folders
        output_dir: Where to save plots
        max_runs: Maximum number of recent runs to compare
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PPO runs
    ppo_dirs = sorted([d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith('PPO_')],
                      key=lambda p: int(p.name.split('_')[1]))
    
    if not ppo_dirs:
        print(f"[ERROR] No PPO training directories found in {base_dir}")
        return
    
    # Select recent runs
    selected_runs = ppo_dirs[-max_runs:] if len(ppo_dirs) > max_runs else ppo_dirs
    print(f"[INFO] Comparing {len(selected_runs)} training runs")
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_runs)))
    
    for idx, run_dir in enumerate(selected_runs):
        data = parse_tensorboard_logs(str(run_dir), tags=['rollout/ep_rew_mean'])
        if data and 'rollout/ep_rew_mean' in data:
            reward_data = data['rollout/ep_rew_mean']
            steps = np.array(reward_data['steps'])
            rewards = np.array(reward_data['values'])
            
            ax.plot(steps, rewards, color=colors[idx], linewidth=1.5, alpha=0.7, label=run_dir.name)
    
    ax.set_title('Training Reward Comparison: Multiple Runs', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Steps (Timesteps)', fontweight='bold')
    ax.set_ylabel('Mean Episode Reward', fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f"{output_dir}/training_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved comparison plot to '{output_file}'")
    
    plt.show()


def main():
    """Main function to plot training rewards."""
    print("="*60)
    print("   VPP AGGREGATOR - TRAINING REWARD PLOTTER")
    print("="*60)
    
    # Set up directories
    tensorboard_dir = os.path.join(script_dir, 'tensorboard_logs')
    output_dir = os.path.join(os.path.dirname(script_dir), 'results_plots')
    
    # Find latest training run
    latest_run = find_latest_run(tensorboard_dir)
    
    if not latest_run:
        print(f"[ERROR] No training runs found in '{tensorboard_dir}'")
        print("        Run train.py first to generate training data.")
        return
    
    # Parse tensorboard logs
    print(f"\n[INFO] Parsing TensorBoard logs from: {latest_run}")
    data = parse_tensorboard_logs(latest_run)
    
    if not data:
        print("[ERROR] Failed to parse tensorboard logs")
        return
    
    # Plot training metrics
    print("\n[INFO] Generating training plots...")
    plot_training_rewards(data, output_dir=output_dir)
    
    # Optional: Compare multiple runs
    print("\n[INFO] Generating comparison plot for recent runs...")
    plot_multiple_runs(base_dir=tensorboard_dir, output_dir=output_dir, max_runs=5)
    
    print("\n" + "="*60)
    print(f"   Plotting complete! Check {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
