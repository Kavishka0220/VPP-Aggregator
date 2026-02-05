import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Path to the evaluation log file
# Resolve paths relative to this script file to avoid CWD issues
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

EVAL_LOG_PATH = os.path.join(project_root, "eval_logs", "evaluations.npz")
OUTPUT_DIR = os.path.join(project_root, "results_plots")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "training_reward_curve.png")

def plot_training_curve():
    if not os.path.exists(EVAL_LOG_PATH):
        print(f"[ERROR] file not found: {EVAL_LOG_PATH}")
        print("Make sure you have trained the model and `eval_logs/evaluations.npz` exists.")
        return

    print(f"[INFO] Loading training logs from {EVAL_LOG_PATH}...")
    data = np.load(EVAL_LOG_PATH)
    
    # Extract data
    timesteps = data['timesteps']
    results = data['results']
    
    print(f"[DEBUG] Loaded data shape: timesteps={timesteps.shape}, results={results.shape}")
    print(f"[DEBUG] Max timestep found: {np.max(timesteps)}")

    # Detect multiple training runs (where timesteps reset)
    # Find indices where the next timestep is smaller than the current one
    reset_indices = np.where(timesteps[1:] < timesteps[:-1])[0] + 1
    
    # Split data into sessions
    split_timesteps = np.split(timesteps, reset_indices)
    split_results = np.split(results, reset_indices)
    
    print(f"[INFO] Found {len(split_timesteps)} separate training sessions in logs.")
    
    # Select the longest session (most likely the 1M run)
    longest_idx = np.argmax([len(arr) for arr in split_timesteps])
    
    timesteps = split_timesteps[longest_idx]
    results = split_results[longest_idx]
    
    print(f"[INFO] Plotting session {longest_idx+1} (Length: {len(timesteps)} checkouts, Max Step: {np.max(timesteps)})")

    # Calculate statistics
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, color='#1f77b4', linewidth=2, label=f'Mean Reward (Run {longest_idx+1})')
    plt.fill_between(timesteps, 
                     mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, 
                     color='#1f77b4', alpha=0.2, label='Std Dev')
    
    plt.title("Training Progress: Reward over Iterations", fontsize=14, fontweight='bold')
    plt.xlabel("Timesteps (Iterations)", fontsize=12, fontweight='bold')
    plt.ylabel("Average Episode Reward", fontsize=12, fontweight='bold')
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"[OK] Training plot saved to '{OUTPUT_FILE}'")
    plt.show()

if __name__ == "__main__":
    plot_training_curve()
