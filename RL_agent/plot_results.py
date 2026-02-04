from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vpp_env import UrbanVPPEnv
import os
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 6

# --- CONFIGURATION ---
MODEL_PATH = os.path.abspath("./checkpoints/best_model/best_model")  # Use best model
STATS_PATH = os.path.abspath("./checkpoints/best_model/vecnormalize.pkl")  # Normalization stats
OUTPUT_DIR = os.path.abspath("./results_plots")  # Where to save plots

steps_to_plot = 96  # One day (15 min intervals)
SCENARIO_NAME = None # Set to same as train.py (e.g. "heatwave_day") or None for default

# Node configuration
SOLAR_NODE_INDICES = [0, 1, 2, 4, 6, 8]
LOAD_ONLY_NODE_INDICES = [3, 5, 7, 9]  
HOME_BATTERY_INDICES = [0, 2]  # Which nodes have home batteries
BESS_NODE_INDEX = 10

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load the Environment and Model
print("[INFO] Loading environment and model...")
def make_env():
    return UrbanVPPEnv(data_path="./data", scenario_name=SCENARIO_NAME, start_index=0)

# Recreate env
env = DummyVecEnv([make_env])

try:
    env = VecNormalize.load(STATS_PATH, env)
    env.training = False 
    env.norm_reward = False
    print(f"[OK] Normalization stats loaded from '{STATS_PATH}'")
except FileNotFoundError:
    print(f"[WARNING] {STATS_PATH} not found. Trying fallback path...")
    try:
        STATS_PATH_FALLBACK = "./checkpoints/ppo_vpp_aggregator_vecnormalize.pkl"
        env = VecNormalize.load(STATS_PATH_FALLBACK, env)
        env.training = False
        env.norm_reward = False
        print(f"[OK] Loaded from fallback: {STATS_PATH_FALLBACK}")
    except FileNotFoundError:
        print(f"[ERROR] No normalization stats found. Proceeding without normalization.")

real_env = env.envs[0]
real_env.max_steps = 96
obs = env.reset()

try:
    model = PPO.load(MODEL_PATH)
    print(f"[OK] Model loaded: '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"[ERROR] Model '{MODEL_PATH}.zip' not found!")
    print("   Trying fallback: './checkpoints/ppo_vpp_aggregator'")
    try:
        model = PPO.load("./checkpoints/ppo_vpp_aggregator")
        print("[OK] Loaded fallback model")
    except:
        print("[ERROR] No model found. Run train.py first!")
        exit()

# 2. Run the Simulation for One Day
history = {
    "soc_bess": [], "soc_hb1": [], "soc_hb2": [],
    "solar": [], "load": [],
    "bess_power": [], "hb1_power": [], "hb2_power": [],
    "all_voltages": [],
    "rewards": [],  # Track rewards
    "grid_import": [],  # Track grid interactions
    "grid_export": [],
    "net_power": []  # Net power flow
}

print("[INFO] Running simulation for 1 day (96 steps)...")
for step in range(steps_to_plot):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = env.step(action)
    
    reward_val = reward[0]
    done = dones[0]
    
    # Collect data
    history["soc_bess"].append(real_env.soc[real_env.storage_map.index(BESS_NODE_INDEX)])
    history["soc_hb1"].append(real_env.soc[real_env.storage_map.index(HOME_BATTERY_INDICES[0])])
    history["soc_hb2"].append(real_env.soc[real_env.storage_map.index(HOME_BATTERY_INDICES[1])])
    
    t = max(real_env.current_step-1, 0)
    solar_total = np.sum(real_env.solar_episode[t])
    load_total = np.sum(real_env.load_episode[t])
    history["solar"].append(solar_total)
    history["load"].append(load_total)
    
    history["hb1_power"].append(real_env.node_battery_power_kw[HOME_BATTERY_INDICES[0]])
    history["hb2_power"].append(real_env.node_battery_power_kw[HOME_BATTERY_INDICES[1]])
    history["bess_power"].append(real_env.node_battery_power_kw[BESS_NODE_INDEX])
    
    history["all_voltages"].append(real_env.voltages.copy())
    history["rewards"].append(reward_val)
    
    # Calculate grid interactions
    net_injection = np.sum(real_env.net_injection)
    history["net_power"].append(net_injection)
    history["grid_export"].append(max(0, net_injection))
    history["grid_import"].append(max(0, -net_injection))
    
    if done: break

print(f"[OK] Simulation complete ({len(history['rewards'])} steps)")
# Calculate statistics
total_reward = sum(history["rewards"])
total_export = sum(history["grid_export"]) * 0.25  # kWh
total_import = sum(history["grid_import"]) * 0.25  # kWh
voltage_violations = sum(1 for v in history["all_voltages"] if np.any((np.array(v) > 1.1) | (np.array(v) < 0.9)))

print("\n=== Performance Summary ===")
print(f"Total Reward: {total_reward:.2f}")
print(f"Energy Exported: {total_export:.2f} kWh")
print(f"Energy Imported: {total_import:.2f} kWh")
print(f"Voltage Violations: {voltage_violations}/{len(history['all_voltages'])} timesteps")
print(f"Final SoC - BESS: {history['soc_bess'][-1]:.2%}, HB1: {history['soc_hb1'][-1]:.2%}, HB2: {history['soc_hb2'][-1]:.2%}")
print()

# Convert to arrays
voltage_matrix = np.array(history["all_voltages"])
time_axis = np.arange(len(history["solar"])) * 15 / 60 

# ==========================================
# FIGURE 1: Simple Thesis Plot (2 Subplots)
# ==========================================
print("[INFO] Generating Figure 1 (Simple Thesis Plot)...")
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot 1: Power Balance
ax1.set_title("Feeder Power Balance", fontsize=10, fontweight='bold')
ax1.plot(time_axis, history["solar"], color='orange', label='Solar Gen', linewidth=1.5, alpha=0.7)
ax1.plot(time_axis, history["load"], color='blue', label='Load Demand', linewidth=1.5, alpha=0.7)
ax1.bar(time_axis, history["bess_power"], color='green', width=0.2, label='BESS Power', alpha=0.7)
ax1.plot(time_axis, history["bess_power"], color='green', label='BESS Power', alpha=0.5)
ax1.plot(time_axis, history["hb1_power"], color='purple', linestyle='--', linewidth=1.2, label='Home Battery 1')
ax1.plot(time_axis, history["hb2_power"], color='magenta', linestyle=':', linewidth=1.2, label='Home Battery 2')
ax1.set_ylabel("Power (kW)", fontsize=8, fontweight='bold')
ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6)
ax1.grid(True, alpha=0.3)

# Plot 2: SoC
ax2.set_title("Battery State of Charge", fontsize=10, fontweight='bold')
ax2.plot(time_axis, history["soc_bess"], color='green', linewidth=1.8, label='Central BESS')
ax2.plot(time_axis, history["soc_hb1"], color='purple', linestyle='--', linewidth=1.5, label='Home Battery 1')
ax2.plot(time_axis, history["soc_hb2"], color='magenta', linestyle=':', linewidth=1.5, label='Home Battery 2')
ax2.set_ylabel("SoC (0-1)", fontsize=8, fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.set_xlabel("Time (Hours)", fontsize=8, fontweight='bold')
ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(np.arange(0, 25, 4))

plt.subplots_adjust(left=0.05, bottom=0.08, right=0.89, top=0.94, hspace=0.22)
output_file_1a = f"{OUTPUT_DIR}/thesis_result_plot.png"
plt.savefig(output_file_1a, dpi=300, bbox_inches='tight')
print(f"[OK] Saved '{output_file_1a}'")

# ==========================================
# FIGURE 2: Comprehensive Power & Economics
# ==========================================
print("[INFO] Generating Figure 2 (Detailed Power & Economics)...")
fig2, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

# Subplot 1: Power Balance
axes[0].set_title("Power Generation and Consumption", fontsize=9, fontweight='bold')
axes[0].plot(time_axis, history["solar"], color='#FF8C00', label='Solar Generation', linewidth=1.5, alpha=0.8)
axes[0].plot(time_axis, history["load"], color='#1E90FF', label='Load Demand', linewidth=1.5, alpha=0.8)
axes[0].fill_between(time_axis, 0, history["solar"], color='#FF8C00', alpha=0.2)
axes[0].fill_between(time_axis, 0, history["load"], color='#1E90FF', alpha=0.2)
axes[0].set_ylabel("Power (kW)", fontweight='bold')
axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=6)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Battery Operations
axes[1].set_title("Battery Power (Negative = Charging, Positive = Discharging)", fontsize=9, fontweight='bold')
axes[1].plot(time_axis, history["bess_power"], color='#32CD32', linewidth=1.5, label='BESS (50kW)', alpha=0.8)
axes[1].plot(time_axis, history["hb1_power"], color='#A680F1', linestyle='--', linewidth=1.2, label='Home Battery 1 (5kW)')
axes[1].plot(time_axis, history["hb2_power"], color='#BC14FF', linestyle=':', linewidth=1.2, label='Home Battery 2 (5kW)')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
axes[1].fill_between(time_axis, 0, history["bess_power"], where=np.array(history["bess_power"])>0, color='#32CD32', alpha=0.2, label='Discharge')
axes[1].fill_between(time_axis, 0, history["bess_power"], where=np.array(history["bess_power"])<0, color='#32CD32', alpha=0.4, label='Charge')
axes[1].set_ylabel("Power (kW)", fontweight='bold')
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=6)
axes[1].grid(True, alpha=0.3)

# Subplot 3: Battery State of Charge
axes[2].set_title("Battery State of Charge", fontsize=9, fontweight='bold')
axes[2].plot(time_axis, history["soc_bess"], color='#32CD32', linewidth=1.8, label='BESS (100kWh)')
axes[2].plot(time_axis, history["soc_hb1"], color="#A680F1", linestyle='--', linewidth=1.5, label='Home Battery 1 (13.5kWh)')
axes[2].plot(time_axis, history["soc_hb2"], color="#BC14FF", linestyle=':', linewidth=1.5, label='Home Battery 2 (13.5kWh)')
axes[2].axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Upper Limit')
axes[2].axhline(y=0.2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Lower Limit')
axes[2].fill_between(time_axis, 0.2, 0.8, color='green', alpha=0.1, label='Safe Zone')
axes[2].set_ylabel("SoC (0-1)", fontweight='bold')
axes[2].set_ylim(0, 1.05)
axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=6)
axes[2].grid(True, alpha=0.3)

# Subplot 4: Grid Interaction
axes[3].set_title("Grid Power Exchange", fontsize=9, fontweight='bold')
axes[3].fill_between(time_axis, 0, history["grid_export"], color='#32CD32', alpha=0.6, label='Export to Grid')
axes[3].fill_between(time_axis, 0, [-x for x in history["grid_import"]], color='#FF6347', alpha=0.6, label='Import from Grid')
axes[3].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[3].set_ylabel("Power (kW)", fontweight='bold')
axes[3].set_xlabel("Time (Hours)", fontweight='bold')
axes[3].legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=6)
axes[3].grid(True, alpha=0.3)

# Format x-axis
for ax in axes:
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(0, 24)

plt.subplots_adjust(left=0.06, bottom=0.08, right=0.85, top=0.95, hspace=0.3)
output_file_2 = f"{OUTPUT_DIR}/1_power_economics_detailed.png"
plt.savefig(output_file_2, dpi=300, bbox_inches='tight')
print(f"[OK] Saved '{output_file_2}'")

# ==========================================
# FIGURE 3: Voltage Profiles (All Nodes)
# ==========================================
print("[INFO] Generating Figure 3 (Voltage Profiles)...")
fig3, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# Plot 1: BESS Node Voltage
axes[0].set_title("Voltage at BESS Connection Point", fontsize=10, fontweight='bold')
axes[0].plot(time_axis, voltage_matrix[:, BESS_NODE_INDEX], color='#FF4500', linewidth=1.5, label=f'Node {BESS_NODE_INDEX} (BESS)')
axes[0].axhline(y=1.10, color='black', linestyle='--', linewidth=1.5, label='Upper Limit (1.10 p.u.)')
axes[0].axhline(y=1.00, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Nominal (1.00 p.u.)')
axes[0].axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, label='Lower Limit (0.90 p.u.)')
axes[0].fill_between(time_axis, 0.90, 1.10, color='green', alpha=0.1, label='Safe Zone')
axes[0].set_ylabel("Voltage (p.u.)", fontweight='bold')
axes[0].set_ylim(0.85, 1.15)
axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=6)
axes[0].grid(True, alpha=0.3)

# Plot 2: Solar Nodes
axes[1].set_title("Voltage Profiles at Solar-Connected Nodes", fontsize=10, fontweight='bold')
colors_solar = plt.cm.tab10(np.linspace(0, 1, len(SOLAR_NODE_INDICES)))
for i, node_idx in enumerate(SOLAR_NODE_INDICES):
    axes[1].plot(time_axis, voltage_matrix[:, node_idx], label=f'Node {node_idx}', 
                color=colors_solar[i], linewidth=1.2, alpha=0.8)
axes[1].axhline(y=1.10, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].axhline(y=1.00, color='gray', linestyle=':', linewidth=1, alpha=0.5)
axes[1].fill_between(time_axis, 0.90, 1.10, color='green', alpha=0.1)
axes[1].set_ylabel("Voltage (p.u.)", fontweight='bold')
axes[1].set_ylim(0.85, 1.15)
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=2, framealpha=0.9, fontsize=6)
axes[1].grid(True, alpha=0.3)

# Plot 3: Load-Only Nodes
axes[2].set_title("Voltage Profiles at Load-Only Nodes", fontsize=10, fontweight='bold')
colors_load = plt.cm.tab10(np.linspace(0, 1, len(LOAD_ONLY_NODE_INDICES)))
for i, node_idx in enumerate(LOAD_ONLY_NODE_INDICES):
    axes[2].plot(time_axis, voltage_matrix[:, node_idx], label=f'Node {node_idx}', 
                color=colors_load[i], linewidth=1.2, alpha=0.8)
axes[2].axhline(y=1.10, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Limits')
axes[2].axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[2].axhline(y=1.00, color='gray', linestyle=':', linewidth=1, alpha=0.5)
axes[2].fill_between(time_axis, 0.90, 1.10, color='green', alpha=0.1)
axes[2].set_ylabel("Voltage (p.u.)", fontweight='bold')
axes[2].set_xlabel("Time (Hours)", fontweight='bold')
axes[2].set_ylim(0.85, 1.15)
axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=2, framealpha=0.9, fontsize=6)
axes[2].grid(True, alpha=0.3)

# Format x-axis
for ax in axes:
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(0, 24)

plt.subplots_adjust(left=0.05, bottom=0.08, right=0.85, top=0.94, hspace=0.22)
output_file_3 = f"{OUTPUT_DIR}/2_voltage_profiles.png"
plt.savefig(output_file_3, dpi=300, bbox_inches='tight')
print(f"[OK] Saved '{output_file_3}'")

# ==========================================
# FIGURE 4: Economic Performance & Rewards
# ==========================================
print("[INFO] Generating Figure 4 (Economics & Rewards)...")
fig4, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Plot 1: Cumulative Reward
cumulative_rewards = np.cumsum(history["rewards"])
axes[0].set_title("Cumulative Reward Over Time", fontsize=10, fontweight='bold')
axes[0].plot(time_axis, cumulative_rewards, color='#2E8B57', linewidth=1.5)
axes[0].fill_between(time_axis, 0, cumulative_rewards, color='#2E8B57', alpha=0.3)
axes[0].set_ylabel("Cumulative Reward", fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Plot 2: Instantaneous Reward
axes[1].set_title("Instantaneous Reward per Timestep", fontsize=10, fontweight='bold')
axes[1].plot(time_axis, history["rewards"], color='#4169E1', linewidth=1.2)
axes[1].fill_between(time_axis, 0, history["rewards"], where=np.array(history["rewards"])>0, 
                     color='#32CD32', alpha=0.4, label='Positive Reward')
axes[1].fill_between(time_axis, 0, history["rewards"], where=np.array(history["rewards"])<0, 
                     color='#FF6347', alpha=0.4, label='Negative Reward')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_ylabel("Reward", fontweight='bold')
axes[1].set_xlabel("Time (Hours)", fontweight='bold')
axes[1].legend(loc="upper right", framealpha=0.9, fontsize=6)
axes[1].grid(True, alpha=0.3)

# Format x-axis
for ax in axes:
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(0, 24)

plt.subplots_adjust(left=0.06, bottom=0.08, right=0.97, top=0.95, hspace=0.2)
output_file_4 = f"{OUTPUT_DIR}/3_rewards.png"
plt.savefig(output_file_4, dpi=300, bbox_inches='tight')
print(f"[OK] Saved '{output_file_4}'")

# ==========================================
# EXPORT DATA
# ==========================================
print("[INFO] Exporting Simulation Data to CSV...")
results_df = pd.DataFrame({
    "Time_Hour": time_axis,
    "Solar_kW": history["solar"],
    "Load_kW": history["load"],
    "BESS_Power_kW": history["bess_power"],
    "HB1_Power_kW": history["hb1_power"],
    "HB2_Power_kW": history["hb2_power"],
    "BESS_SoC": history["soc_bess"],
    "HB1_SoC": history["soc_hb1"],
    "HB2_SoC": history["soc_hb2"],
    "Net_Grid_Power_kW": history["net_power"],
    "Grid_Export_kW": history["grid_export"],
    "Grid_Import_kW": history["grid_import"],
    "Instant_Reward": history["rewards"]
})

csv_file_path = f"{OUTPUT_DIR}/detailed_simulation_results.csv"
results_df.to_csv(csv_file_path, index=False)
print(f"[OK] Saved detailed results to '{csv_file_path}'")

print("\n" + "="*50)
print("All plots saved successfully!")
print(f"Output directory: {OUTPUT_DIR}/")
print("="*50)

plt.show()