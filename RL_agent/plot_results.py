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
plt.rcParams['font.size'] = 19

# --- CONFIGURATION ---
# Get script directory for absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "checkpoints", "best_model", "best_model")  # Use best model
STATS_PATH = os.path.join(script_dir, "checkpoints", "best_model", "vecnormalize.pkl")  # Normalization stats
OUTPUT_DIR = os.path.join(os.path.dirname(script_dir), "results_plots")  # Where to save plots

steps_to_plot = 96  # One day (15 min intervals)
SCENARIO_NAME = "intermittent_solar" # Set to same as train.py (e.g. "heatwave_day") or None for default
 
# Node configuration
SOLAR_NODE_INDICES = [3, 5, 7, 10, 11, 13, 15, 17, 18, 19, 20]
LOAD_ONLY_NODE_INDICES = [0, 1, 2, 4, 6, 8, 9, 12, 14, 16]  
HOME_BATTERY_INDICES = [3, 5]  # Which nodes have home batteries
BESS_NODE_INDEX = 21

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create scenario-specific subdirectory
SCENARIO_FOLDER = os.path.join(OUTPUT_DIR, SCENARIO_NAME) if SCENARIO_NAME else os.path.join(OUTPUT_DIR, "default")
os.makedirs(SCENARIO_FOLDER, exist_ok=True)
print(f"[INFO] Saving outputs to scenario folder: {SCENARIO_FOLDER}")

# 1. Load the Environment and Model
print("[INFO] Loading environment and model...")
# Get absolute path to data directory
data_path = os.path.join(os.path.dirname(script_dir), "data")

def make_env():
    return UrbanVPPEnv(data_path=data_path, scenario_name=SCENARIO_NAME, start_index=0)

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
        STATS_PATH_FALLBACK = os.path.join(script_dir, "checkpoints", "ppo_vpp_aggregator_vecnormalize.pkl")
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
    MODEL_PATH_FALLBACK = os.path.join(script_dir, "checkpoints", "ppo_vpp_aggregator")
    print(f"   Trying fallback: '{MODEL_PATH_FALLBACK}'")
    try:
        model = PPO.load(MODEL_PATH_FALLBACK)
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
    "net_power": [],  # Net power flow
    # Per-node data for homes
    "hb1_load": [], "hb1_solar": [],
    "hb2_load": [], "hb2_solar": []
}

print("[INFO] Running simulation for 1 day (96 steps)...")

# Check which home batteries actually exist in storage_map
HOME_BATTERY_INDICES_AVAILABLE = [idx for idx in HOME_BATTERY_INDICES if idx in real_env.storage_map]
print(f"[INFO] Available home batteries: {HOME_BATTERY_INDICES_AVAILABLE}")
if len(HOME_BATTERY_INDICES_AVAILABLE) < len(HOME_BATTERY_INDICES):
    print(f"[WARNING] Not all home batteries available. Using default 0 values for missing ones.")

for step in range(steps_to_plot):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = env.step(action)
    
    reward_val = reward[0]
    done = dones[0]
    
    # Collect data - handle missing home batteries
    history["soc_bess"].append(real_env.soc[real_env.storage_map.index(BESS_NODE_INDEX)])
    
    # Home Battery 1 (Node 3)
    if HOME_BATTERY_INDICES[0] in real_env.storage_map:
        history["soc_hb1"].append(real_env.soc[real_env.storage_map.index(HOME_BATTERY_INDICES[0])])
    else:
        history["soc_hb1"].append(0.0)  # Default to 0 if not present
    
    # Home Battery 2 (Node 5)
    if HOME_BATTERY_INDICES[1] in real_env.storage_map:
        history["soc_hb2"].append(real_env.soc[real_env.storage_map.index(HOME_BATTERY_INDICES[1])])
    else:
        history["soc_hb2"].append(0.0)  # Default to 0 if not present
    
    t = max(real_env.current_step-1, 0)
    solar_total = np.sum(real_env.solar_episode[t])
    load_total = np.sum(real_env.load_episode[t])
    history["solar"].append(solar_total)
    history["load"].append(load_total)
    
    # Per-node data for homes
    history["hb1_load"].append(real_env.load_episode[t][HOME_BATTERY_INDICES[0]])
    history["hb1_solar"].append(real_env.solar_episode[t][HOME_BATTERY_INDICES[0]])
    history["hb2_load"].append(real_env.load_episode[t][HOME_BATTERY_INDICES[1]])
    history["hb2_solar"].append(real_env.solar_episode[t][HOME_BATTERY_INDICES[1]])
    
    # Handle missing home batteries for power data
    if HOME_BATTERY_INDICES[0] in real_env.storage_map:
        history["hb1_power"].append(real_env.node_battery_power_kw[HOME_BATTERY_INDICES[0]])
    else:
        history["hb1_power"].append(0.0)  # Default to 0 if not present
    
    if HOME_BATTERY_INDICES[1] in real_env.storage_map:
        history["hb2_power"].append(real_env.node_battery_power_kw[HOME_BATTERY_INDICES[1]])
    else:
        history["hb2_power"].append(0.0)  # Default to 0 if not present
    
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
voltage_violations = sum(1 for v in history["all_voltages"] if np.any((np.array(v) > 1.06) | (np.array(v) < 0.94)))

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
ax1.set_title("Feeder Power Balance", fontsize=23, fontweight='bold')
ax1.plot(time_axis, history["solar"], color='orange', label='Solar Gen', linewidth=1.5, alpha=0.7)
ax1.plot(time_axis, history["load"], color='blue', label='Load Demand', linewidth=1.5, alpha=0.7)
ax1.bar(time_axis, history["bess_power"], color='green', width=0.2, label='BESS Power', alpha=0.7)
ax1.plot(time_axis, history["bess_power"], color='green', label='BESS Power', alpha=0.5)

# Only plot home batteries if they exist
if HOME_BATTERY_INDICES[0] in real_env.storage_map:
    ax1.plot(time_axis, history["hb1_power"], color='purple', linestyle='--', linewidth=1.2, label='Home Battery 1')
if HOME_BATTERY_INDICES[1] in real_env.storage_map:
    ax1.plot(time_axis, history["hb2_power"], color='magenta', linestyle=':', linewidth=1.2, label='Home Battery 2')

ax1.set_ylabel("Power (kW)", fontsize=21, fontweight='bold')
ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=19)
ax1.grid(True, alpha=0.3)

# Plot 2: SoC
ax2.set_title("Battery State of Charge", fontsize=23, fontweight='bold')
ax2.plot(time_axis, history["soc_bess"], color='green', linewidth=1.8, label='Central BESS')

# Only plot home batteries if they exist
if HOME_BATTERY_INDICES[0] in real_env.storage_map:
    ax2.plot(time_axis, history["soc_hb1"], color='purple', linestyle='--', linewidth=1.5, label='Home Battery 1')
if HOME_BATTERY_INDICES[1] in real_env.storage_map:
    ax2.plot(time_axis, history["soc_hb2"], color='magenta', linestyle=':', linewidth=1.5, label='Home Battery 2')

ax2.set_ylabel("SoC (0-1)", fontsize=21, fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.set_xlabel("Time (Hours)", fontsize=21, fontweight='bold')
ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=19)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(np.arange(0, 25, 4))

plt.subplots_adjust(left=0.05, bottom=0.08, right=0.89, top=0.94, hspace=0.22)
output_file_1a = f"{SCENARIO_FOLDER}/thesis_result_plot.png"
plt.savefig(output_file_1a, dpi=300, bbox_inches='tight')
print(f"[OK] Saved '{output_file_1a}'")

# ==========================================
# FIGURE 2: Comprehensive Power & Economics
# ==========================================
print("[INFO] Generating Figure 2 (Detailed Power & Economics)...")
fig2, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# Subplot 1: Power Balance
axes[0].set_title("Power Generation and Consumption", fontsize=22, fontweight='bold')
axes[0].plot(time_axis, history["solar"], color='#FF8C00', label='Solar Generation', linewidth=1.5, alpha=0.8)
axes[0].plot(time_axis, history["load"], color='#1E90FF', label='Load Demand', linewidth=1.5, alpha=0.8)
axes[0].fill_between(time_axis, 0, history["solar"], color='#FF8C00', alpha=0.2)
axes[0].fill_between(time_axis, 0, history["load"], color='#1E90FF', alpha=0.2)
axes[0].set_ylabel("Power (kW)", fontweight='bold')
axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=19)
axes[0].grid(True, alpha=0.3)

# Subplot 2: BESS Charge/Discharge Curve + SOC (Dual Axis)
axes[1].set_title("BESS Charge/Discharge Power & State of Charge(SoC)", fontsize=22, fontweight='bold')
# Left y-axis: Power
axes[1].plot(time_axis, history["bess_power"], color='#32CD32', linewidth=1.8, label='BESS Power', alpha=0.9)
axes[1].fill_between(time_axis, 0, history["bess_power"], where=np.array(history["bess_power"])>0, color='#32CD32', alpha=0.25, label='Discharge')
axes[1].fill_between(time_axis, 0, history["bess_power"], where=np.array(history["bess_power"])<0, color='#FF6347', alpha=0.25, label='Charge')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
axes[1].set_ylabel("BESS Power (kW)", fontweight='bold', color='#32CD32')
axes[1].tick_params(axis='y', labelcolor='#32CD32')
# Right y-axis: SOC
ax1_right = axes[1].twinx()
ax1_right.plot(time_axis, history["soc_bess"], color='#1E90FF', linewidth=1.5, linestyle='--', label='BESS SOC', alpha=0.9)
ax1_right.axhline(y=0.8, color='black', linestyle=':', linewidth=1, alpha=0.5)
ax1_right.axhline(y=0.2, color='black', linestyle=':', linewidth=1, alpha=0.5)
ax1_right.fill_between(time_axis, 0.2, 0.8, color='green', alpha=0.05)
ax1_right.set_ylabel("BESS SOC (0-1)", fontweight='bold', color='#1E90FF')
ax1_right.tick_params(axis='y', labelcolor='#1E90FF')
ax1_right.set_ylim(0, 1.05)
# Combined legend
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax1_right.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.07, 1), framealpha=0.9, fontsize=19)
axes[1].grid(True, alpha=0.3)

# Subplot 3: Grid Interaction
axes[2].set_title("Grid Power Exchange", fontsize=22, fontweight='bold')
axes[2].fill_between(time_axis, 0, history["grid_export"], color='#32CD32', alpha=0.6, label='Export to Grid')
axes[2].fill_between(time_axis, 0, [-x for x in history["grid_import"]], color='#FF6347', alpha=0.6, label='Import from Grid')
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[2].set_ylabel("Power (kW)", fontweight='bold')
axes[2].set_xlabel("Time (Hours)", fontweight='bold')
axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=19)
axes[2].grid(True, alpha=0.3)

# Format x-axis
for ax in axes:
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(0, 24)

plt.subplots_adjust(left=0.06, bottom=0.08, right=0.87, top=0.95, hspace=0.25)
output_file_2 = f"{SCENARIO_FOLDER}/1_power_exchange_detailed.png"
plt.savefig(output_file_2, dpi=300, bbox_inches='tight')
print(f"[OK] Saved '{output_file_2}'")
output_file_2_pdf = f"{SCENARIO_FOLDER}/1_power_exchange_detailed.pdf"
plt.savefig(output_file_2_pdf, format='pdf', bbox_inches='tight')
print(f"[OK] Saved '{output_file_2_pdf}'")

# ==========================================
# FIGURE 3: BESS Power Exchange & Voltage (Combined - Dual Axis)
# ==========================================
print("[INFO] Generating Figure 3 (BESS Power & Voltage Combined)...")
fig_bess = plt.figure(figsize=(18, 9))
ax_power = fig_bess.add_subplot(111)

# Plot 1: BESS Power (Left Axis)
ax_power.set_title("BESS Power Exchange and Voltage at BESS Node", fontsize=28, fontweight='bold', pad=20)
line_power = ax_power.plot(time_axis, history["bess_power"], color='#32CD32', linewidth=3.5, label='BESS Power', alpha=0.85, zorder=3)
ax_power.fill_between(time_axis, 0, history["bess_power"], where=np.array(history["bess_power"])>0, 
                      color='#32CD32', alpha=0.25, label='Discharge (Export)', zorder=2)
ax_power.fill_between(time_axis, 0, history["bess_power"], where=np.array(history["bess_power"])<0, 
                      color='#FF6347', alpha=0.25, label='Charge (Import)', zorder=2)
ax_power.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.6, zorder=1)
ax_power.set_ylabel("BESS Power (kW)", fontweight='bold', color='#32CD32', fontsize=26)
ax_power.tick_params(axis='y', labelcolor='#32CD32', labelsize=23)
ax_power.tick_params(axis='x', labelsize=23)
ax_power.set_xlabel("Time (Hours)", fontweight='bold', fontsize=26)
ax_power.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
ax_power.set_xticks(np.arange(0, 25, 2))
ax_power.set_xlim(0, 24)

# Plot 2: BESS Voltage (Right Axis - Dual Axis)
ax_voltage = ax_power.twinx()
line_voltage = ax_voltage.plot(time_axis, voltage_matrix[:, BESS_NODE_INDEX], color='#FF4500', linewidth=3.5, label='BESS Node Voltage', alpha=0.85, zorder=4, linestyle='-')
ax_voltage.axhline(y=1.06, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Upper Limit (1.06 p.u.)', zorder=1)
ax_voltage.axhline(y=1.00, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Nominal (1.00 p.u.)', zorder=1)
ax_voltage.axhline(y=0.94, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Lower Limit (0.94 p.u.)', zorder=1)
#ax_voltage.fill_between(time_axis, 0.94, 1.06, color='green', alpha=0.08, label='Safe Operating Zone', zorder=0)
ax_voltage.set_ylabel("Voltage (p.u.)", fontweight='bold', color='#FF4500', fontsize=26)
ax_voltage.tick_params(axis='y', labelcolor='#FF4500', labelsize=23)
ax_voltage.set_ylim(0.88, 1.14)

# Combined Legend
lines_power = line_power + [ax_power.fill_between([], [], [], color='#32CD32', alpha=0.25),
                            ax_power.fill_between([], [], [], color='#FF6347', alpha=0.25)]
lines_voltage = line_voltage + [ax_voltage.plot([], [], color='red', linestyle='--', linewidth=2.5)[0],
                                ax_voltage.plot([], [], color='gray', linestyle=':', linewidth=2)[0],
                                ax_voltage.plot([], [], color='red', linestyle='--', linewidth=2.5)[0],
                                ax_voltage.fill_between([], [], [], color='green', alpha=0.08)]
labels_power = ['BESS Power', 'Discharge (Export)', 'Charge (Import)']
labels_voltage = ['BESS Node Voltage', 'Upper Limit (1.06 p.u.)', 'Nominal (1.00 p.u.)', 'Lower Limit (0.94 p.u.)', 'Safe Zone']

ax_power.legend(lines_power + lines_voltage, labels_power + labels_voltage, 
               loc='upper left', fontsize=22,bbox_to_anchor=(1.01, 1), framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

plt.tight_layout()
output_file_bess = f"{SCENARIO_FOLDER}/3_bess_power_voltage_combined.png"
plt.savefig(output_file_bess, dpi=300, bbox_inches='tight')
print(f"[OK] Saved '{output_file_bess}'")
output_file_bess_pdf = f"{SCENARIO_FOLDER}/3_bess_power_voltage_combined.pdf"
plt.savefig(output_file_bess_pdf, format='pdf', bbox_inches='tight')
print(f"[OK] Saved '{output_file_bess_pdf}'")
plt.close(fig_bess)

# ==========================================
# ==========================================
# FIGURE 4: Home Batteries - Detailed Per-Home Analysis (if available)
# ==========================================
# Only create this plot if at least one home battery exists
if len(HOME_BATTERY_INDICES_AVAILABLE) > 0:
    print("[INFO] Generating Figure 4 (Home Batteries Details)...")
    fig3, axes = plt.subplots(len(HOME_BATTERY_INDICES_AVAILABLE), 1, figsize=(16, 5 * len(HOME_BATTERY_INDICES_AVAILABLE)), sharex=True)
    
    # Ensure axes is always a list (in case of single subplot)
    if len(HOME_BATTERY_INDICES_AVAILABLE) == 1:
        axes = [axes]
    
    # Plot each available home battery
    for plot_idx, home_battery_idx in enumerate(HOME_BATTERY_INDICES_AVAILABLE):
        ax = axes[plot_idx]
        
        if home_battery_idx == HOME_BATTERY_INDICES[0]:
            hb_data = {"load": history["hb1_load"], "solar": history["hb1_solar"], "power": history["hb1_power"], "soc": history["soc_hb1"]}
        else:
            hb_data = {"load": history["hb2_load"], "solar": history["hb2_solar"], "power": history["hb2_power"], "soc": history["soc_hb2"]}
        
        ax.set_title(f"Home at Node {home_battery_idx} - Load, Solar, Battery Operations & State of Charge", fontsize=23, fontweight='bold')
        
        # Left y-axis: Power flows
        ax.plot(time_axis, hb_data["load"], color='#1E90FF', linewidth=1, label='Load Demand', alpha=0.8)
        ax.plot(time_axis, hb_data["solar"], color='#FF8C00', linewidth=1, label='Solar Generation', alpha=0.8)
        ax.fill_between(time_axis, 0, hb_data["load"], color='#1E90FF', alpha=0.2)
        ax.fill_between(time_axis, 0, hb_data["solar"], color='#FF8C00', alpha=0.2)
        ax.plot(time_axis, hb_data["power"], color='#32CD32', linewidth=1, label='Battery Power', alpha=0.8)
        ax.fill_between(time_axis, 0, hb_data["power"], where=np.array(hb_data["power"])>0, color='#32CD32', alpha=0.2, label='Discharge')
        ax.fill_between(time_axis, 0, hb_data["power"], where=np.array(hb_data["power"])<0, color='#FF6347', alpha=0.2, label='Charge')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Power (kW)", fontweight='bold', color='black')
        ax.tick_params(axis='y', labelcolor='black')
        
        # Right y-axis: SOC
        ax_right = ax.twinx()
        ax_right.plot(time_axis, hb_data["soc"], color='#1E90FF', linewidth=1.5, linestyle='--', label='Battery SOC', alpha=0.9)
        ax_right.axhline(y=0.8, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax_right.axhline(y=0.2, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax_right.set_ylabel("Battery SOC (0-1)", fontweight='bold', color='#1E90FF')
        ax_right.tick_params(axis='y', labelcolor='#1E90FF')
        ax_right.set_ylim(0, 1.05)
        
        # Combined legend
        lines, labels = ax.get_legend_handles_labels()
        lines_r, labels_r = ax_right.get_legend_handles_labels()
        ax.legend(lines + lines_r, labels + labels_r, loc="upper left", bbox_to_anchor=(1.07, 1), framealpha=0.9, fontsize=19)
        ax.grid(True, alpha=0.3)
        
        if plot_idx == len(HOME_BATTERY_INDICES_AVAILABLE) - 1:
            ax.set_xlabel("Time (Hours)", fontweight='bold')
    
    # Format x-axis
    for ax in axes:
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xlim(0, 24)
        ax.set_ylabel("Power (kW)", fontweight='bold')  # Missing in original, but adding formatting
    
    axes[-1].set_xlabel("Time (Hours)", fontweight='bold')
    
    plt.subplots_adjust(left=0.06, bottom=0.08, right=0.85, top=0.95, hspace=0.25)
    output_file_3 = f"{SCENARIO_FOLDER}/1_5_home_batteries.png"
    plt.savefig(output_file_3, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved '{output_file_3}'")
    output_file_3_pdf = f"{SCENARIO_FOLDER}/1_5_home_batteries.pdf"
    plt.savefig(output_file_3_pdf, format='pdf', bbox_inches='tight')
    print(f"[OK] Saved '{output_file_3_pdf}'")
else:
    print("[INFO] Skipping Figure 3 - No home batteries available")
    output_file_3 = None

# ==========================================
# FIGURE 5: Voltage Profiles (All Nodes)
# ==========================================
print("[INFO] Generating Figure 5 (Voltage Profiles)...")
fig4, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# Plot 1: BESS Node Voltage
axes[0].set_title("Voltage at BESS Connection Point", fontsize=23, fontweight='bold')
axes[0].plot(time_axis, voltage_matrix[:, BESS_NODE_INDEX], color='#FF4500', linewidth=1.5, label=f'Node {BESS_NODE_INDEX} (BESS)')
axes[0].axhline(y=1.06, color='black', linestyle='--', linewidth=1.5, label='Upper Limit (1.06 p.u.)')
axes[0].axhline(y=1.00, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Nominal (1.00 p.u.)')
axes[0].axhline(y=0.94, color='black', linestyle='--', linewidth=1.5, label='Lower Limit (0.94 p.u.)')
#axes[0].fill_between(time_axis, 0.94, 1.06, color='green', alpha=0.1, label='Safe Zone')
axes[0].set_ylabel("Voltage (p.u.)", fontweight='bold')
axes[0].set_ylim(0.85, 1.15)
axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=19)
axes[0].grid(True, alpha=0.3)

# Plot 2: Solar Nodes
axes[1].set_title("Voltage Profiles at Solar-Connected Nodes", fontsize=23, fontweight='bold')
colors_solar = plt.cm.tab10(np.linspace(0, 1, len(SOLAR_NODE_INDICES)))
for i, node_idx in enumerate(SOLAR_NODE_INDICES):
    axes[1].plot(time_axis, voltage_matrix[:, node_idx], label=f'Node {node_idx}', 
                color=colors_solar[i], linewidth=1.2, alpha=0.8)
axes[1].axhline(y=1.06, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].axhline(y=0.94, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].axhline(y=1.00, color='gray', linestyle=':', linewidth=1, alpha=0.5)
#axes[1].fill_between(time_axis, 0.94, 1.06, color='green', alpha=0.1)
axes[1].set_ylabel("Voltage (p.u.)", fontweight='bold')
axes[1].set_ylim(0.85, 1.15)
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=2, framealpha=0.9, fontsize=19)
axes[1].grid(True, alpha=0.3)

# Plot 3: Load-Only Nodes
axes[2].set_title("Voltage Profiles at Load-Only Nodes", fontsize=23, fontweight='bold')
colors_load = plt.cm.tab10(np.linspace(0, 1, len(LOAD_ONLY_NODE_INDICES)))
for i, node_idx in enumerate(LOAD_ONLY_NODE_INDICES):
    axes[2].plot(time_axis, voltage_matrix[:, node_idx], label=f'Node {node_idx}', 
                color=colors_load[i], linewidth=1.2, alpha=0.8)
axes[2].axhline(y=1.06, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Limits')
axes[2].axhline(y=0.94, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[2].axhline(y=1.00, color='gray', linestyle=':', linewidth=1, alpha=0.5)
#axes[2].fill_between(time_axis, 0.94, 1.06, color='green', alpha=0.1)
axes[2].set_ylabel("Voltage (p.u.)", fontweight='bold')
axes[2].set_xlabel("Time (Hours)", fontweight='bold')
axes[2].set_ylim(0.85, 1.15)
axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=2, framealpha=0.9, fontsize=19)
axes[2].grid(True, alpha=0.3)

# Format x-axis
for ax in axes:
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(0, 24)

plt.subplots_adjust(left=0.05, bottom=0.08, right=0.85, top=0.94, hspace=0.22)
output_file_4 = f"{SCENARIO_FOLDER}/2_voltage_profiles.png"
plt.savefig(output_file_4, dpi=300, bbox_inches='tight')
print(f"[OK] Saved '{output_file_4}'")
output_file_4_pdf = f"{SCENARIO_FOLDER}/2_voltage_profiles.pdf"
plt.savefig(output_file_4_pdf, format='pdf', bbox_inches='tight')
print(f"[OK] Saved '{output_file_4_pdf}'")

# ==========================================
# FIGURE 6: Economic Performance & Rewards
# ==========================================
print("[INFO] Generating Figure 6 (Economics & Rewards)...")
fig5, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Plot 1: Cumulative Reward
cumulative_rewards = np.cumsum(history["rewards"])
axes[0].set_title("Cumulative Reward Over Time", fontsize=23, fontweight='bold')
axes[0].plot(time_axis, cumulative_rewards, color='#2E8B57', linewidth=1.5)
axes[0].fill_between(time_axis, 0, cumulative_rewards, color='#2E8B57', alpha=0.3)
axes[0].set_ylabel("Cumulative Reward", fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Plot 2: Instantaneous Reward
axes[1].set_title("Instantaneous Reward per Timestep", fontsize=23, fontweight='bold')
axes[1].plot(time_axis, history["rewards"], color='#4169E1', linewidth=1.2)
axes[1].fill_between(time_axis, 0, history["rewards"], where=np.array(history["rewards"])>0, 
                     color='#32CD32', alpha=0.4, label='Positive Reward')
axes[1].fill_between(time_axis, 0, history["rewards"], where=np.array(history["rewards"])<0, 
                     color='#FF6347', alpha=0.4, label='Negative Reward')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_ylabel("Reward", fontweight='bold')
axes[1].set_xlabel("Time (Hours)", fontweight='bold')
axes[1].legend(loc="upper right", framealpha=0.9, fontsize=17)
axes[1].grid(True, alpha=0.3)

# Format x-axis
for ax in axes:
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(0, 24)

plt.subplots_adjust(left=0.06, bottom=0.08, right=0.97, top=0.95, hspace=0.2)
output_file_5 = f"{SCENARIO_FOLDER}/3_rewards.png"
plt.savefig(output_file_5, dpi=300, bbox_inches='tight')
print(f"[OK] Saved '{output_file_5}'")

# ==========================================
# FIGURE 6: BESS Voltage Profile (Enlarged from Figure 4 Plot 1)
# ==========================================
print("[INFO] Generating Figure 6 (BESS Voltage - Enlarged)...")
fig6, ax = plt.subplots(1, 1, figsize=(16, 6))

ax.set_title("Voltage Comparison: BESS and Node 1", fontsize=23, fontweight='bold')
ax.plot(time_axis, voltage_matrix[:, BESS_NODE_INDEX], color="#002AFF", linewidth=1.5, label=f'Node {BESS_NODE_INDEX} (BESS)')
ax.plot(time_axis, voltage_matrix[:, 1], color='#1E90FF', linewidth=1.5, label='Node 1')
ax.axhline(y=1.06, color='black', linestyle='--', linewidth=1.5, label='Upper Limit (1.06 p.u.)')
ax.axhline(y=1.00, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Nominal (1.00 p.u.)')
ax.axhline(y=0.94, color='black', linestyle='--', linewidth=1.5, label='Lower Limit (0.94 p.u.)')
#ax.fill_between(time_axis, 0.94, 1.06, color='green', alpha=0.1, label='Safe Zone')
ax.set_ylabel("Voltage (p.u.)", fontweight='bold', fontsize=20)
ax.set_xlabel("Time (Hours)", fontweight='bold', fontsize=20)
ax.set_ylim(0.85, 1.15)
ax.set_xticks(np.arange(0, 25, 2))
ax.set_xlim(0, 24)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), framealpha=0.9, fontsize=19)
ax.grid(True, alpha=0.3)

plt.subplots_adjust(left=0.08, bottom=0.12, right=0.85, top=0.93)
output_file_6 = f"{SCENARIO_FOLDER}/4_bess_voltage_enlarged.png"
plt.savefig(output_file_6, dpi=300, bbox_inches='tight')
print(f"[OK] Saved '{output_file_6}'")
output_file_6_pdf = f"{SCENARIO_FOLDER}/4_bess_voltage_enlarged.pdf"
plt.savefig(output_file_6_pdf, format='pdf', bbox_inches='tight')
print(f"[OK] Saved '{output_file_6_pdf}'")

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

csv_file_path = f"{SCENARIO_FOLDER}/detailed_simulation_results.csv"
results_df.to_csv(csv_file_path, index=False)
print(f"[OK] Saved detailed results to '{csv_file_path}'")

print("\n" + "="*50)
print("All plots saved successfully!")
print(f"Output directory: {SCENARIO_FOLDER}/")
print("="*50)

#plt.show()