from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import matplotlib.pyplot as plt
from vpp_env import UrbanVPPEnv 

# --- CONFIGURATION ---
MODEL_PATH = "ppo_vpp_aggregator" # Make sure this matches your saved filename (without .zip)
STATS_PATH = "vecnormalize_stats.pkl" # Path to the saved normalization stats

steps_to_plot = 96  # One day (15 min intervals)

# Define which nodes are "Solar PCCs" 
# Usually indices 0, 1, 2, 4 ,6, 8 gfare houses, and 10 is the BESS in small grid models.
SOLAR_NODE_INDICES = [0, 1, 2, 4, 6, 8]
LOAD_ONLY_NODE_INDICES = [3, 5, 7, 9]  
bess_voltage_node_index = 10

# 1. Load the Environment and Model
def make_env():
    return UrbanVPPEnv()

# Recreate env
env = DummyVecEnv([make_env])
#env = DummyVecEnv([lambda: UrbanVPPEnv()])
try:
    # Load the saved normalization stats
    env = VecNormalize.load(STATS_PATH, env)
    # Important: Turn off training updates for the stats during test
    env.training = False 
    env.norm_reward = False
    print(f"✅ Normalization stats loaded from '{STATS_PATH}'")
except FileNotFoundError:
    print(f"❌ Error: {STATS_PATH} not found. Run train.py first.")
    exit()
real_env = env.envs[0]  # Unwrap to access internal variables if needed
real_env.max_steps = 96  # Limit to one day for plotting
obs = env.reset()

try:
    model = PPO.load(MODEL_PATH)
    print(f"✅ Model '{MODEL_PATH}' loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Could not find '{MODEL_PATH}.zip'.")
    print("   Make sure you ran train.py and it saved the model!")
    exit()

# 2. Run the Simulation for One Day
# We will store data here to plot later
history = {
    "soc_bess": [], "soc_hb1": [], "soc_hb2": [],
    "solar": [], "load": [],
    "bess_power": [], "hb1_power": [], "hb2_power": [],
    "all_voltages": [] # <--- Store ALL voltages here
}

print("Running simulation...")
for _ in range(steps_to_plot):
    action, _ = model.predict(obs, deterministic=True) # Ask the agent for an action
    obs, reward, dones, infos = env.step(action) # Execute the action
    
    reward = reward[0]
    done = dones[0]
    info = infos[0]


    # --- COLLECT DATA ---
    # Extract raw data from the environment wrapper
    # Note: We need to access the internal variables of env
    history["soc_bess"].append(real_env.soc[real_env.storage_map.index(10)]) # Index 10 is BESS
    history["soc_hb1"].append(real_env.soc[real_env.storage_map.index(0)]) # Index 0 is Home Battery 1
    history["soc_hb2"].append(real_env.soc[real_env.storage_map.index(2)]) # Index 2 is Home Battery 2
    
    t = max(real_env.current_step-1, 0) # Current time index for episode data
    history["solar"].append(np.sum(real_env.solar_episode[t]))
    history["load"].append(np.sum(real_env.load_episode[t]))

    history["hb1_power"].append(real_env.node_battery_power_kw[0])
    history["hb2_power"].append(real_env.node_battery_power_kw[2])
    history["bess_power"].append(real_env.node_battery_power_kw[10]) 
    
    # Capture ALL voltages for the new plot (Make a copy to be safe)
    history["all_voltages"].append(real_env.voltages.copy())

    if done: break

    
    
# Convert list of arrays to a 2D numpy array for easier slicing [time, node_index]
voltage_matrix = np.array(history["all_voltages"]) 
time_axis = np.arange(steps_to_plot) * 15 / 60 

# ==========================================
# FIGURE 1: Main Thesis Plot (3 Subplots)
# ==========================================
print("Generating Figure 1 (General Results)...")
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), sharex=True)

# Plot 1: Power
ax1.set_title("Feeder Power Balance")
ax1.plot(time_axis, history["solar"], color='orange', label='Solar Gen', alpha=0.7)
ax1.plot(time_axis, history["load"], color='blue', label='Load Demand', alpha=0.7)
ax1.bar(time_axis, history["bess_power"], color='green', width=0.2, label='BESS Power (kW)', alpha=0.5)
ax1.plot(time_axis, history["hb1_power"], color='purple', linestyle='--', label='Home Battery 1 (kW)')
ax1.plot(time_axis, history["hb2_power"], color='magenta', linestyle=':', label='Home Battery 2 (kW)')
ax1.set_ylabel("Power (kW)")
ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
ax1.grid(True, alpha=0.3)

# Plot 2: SoC
ax2.set_title("Battery State of Charge")
ax2.plot(time_axis, history["soc_bess"], color='green', linewidth=2.5, label='Central BESS')
ax2.plot(time_axis, history["soc_hb1"], color='purple', linestyle='--', label='Home Battery 1')
ax2.plot(time_axis, history["soc_hb2"], color='magenta', linestyle=':', label='Home Battery 2')
ax2.set_ylabel("SoC (0-1)")
ax2.set_ylim(0, 1.05)
ax2.set_xlabel("Time (Hours)")
ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
ax2.grid(True, alpha=0.3)

ax2.set_xticks(np.arange(0, 25, 4)) 

plt.subplots_adjust(top=0.94, bottom=0.07,left=0.05, right=0.85, hspace=0.2)
#plt.tight_layout()
plt.savefig("thesis_result_plot.png", bbox_inches='tight') 
print("✅ Saved 'thesis_result_plot.png'")

# ==========================================
# FIGURE 2: DETAILED VOLTAGES (Solar PCCs)
# ==========================================
print("Generating Figure 2 (Voltages)...")
fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)

# Plot 1: Voltage (Just BESS Node for Summary)
ax1.set_title("Grid Voltage at BESS Node")
ax1.plot(time_axis, voltage_matrix[:, bess_voltage_node_index], color='red', linewidth=2, label='BESS Node Voltage (p.u.)')
ax1.axhline(y=1.10, color='black', linestyle='--', label='Trip Limit (1.10)')
ax1.axhline(y=1.00, color='gray', linestyle=':', label='Nominal (1.00)')
ax1.axhline(y=0.90, color='black', linestyle='--', label='Trip Limit (0.90)')
ax1.set_ylabel("Voltage (p.u.)")
ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
ax1.grid(True, alpha=0.3)

ax2.set_title("Voltage Profiles at Solar PCC Nodes")

# Loop through the list of nodes we defined at the top
colors = plt.cm.tab10(np.linspace(0, 1, len(SOLAR_NODE_INDICES))) # Generate distinct colors

for i, node_idx in enumerate(SOLAR_NODE_INDICES):
    # voltage_matrix[:, node_idx] gets the voltage for that node across all time steps
    ax2.plot(time_axis, voltage_matrix[:, node_idx], label=f'Node {node_idx}', color=colors[i], linewidth=2)

# Add Limits
ax2.axhline(y=1.10, color='black', linestyle='--', linewidth=1.5, label='Upper Limit (1.10)')
ax2.axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, label='Lower Limit (0.90)')
ax2.axhline(y=1.00, color='gray', linestyle=':', alpha=0.5)

ax2.set_ylabel("Voltage (p.u.)")
ax2.set_xticks(np.arange(0, 25, 4)) 
ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.01), title="Nodes")
ax2.grid(True, alpha=0.3)

# Plot 3: Voltage Profiles at Load-Only Nodes
ax3.set_title("Voltage Profiles at Load-Only Nodes")

colors_load = plt.cm.tab10(np.linspace(0, 1, len(LOAD_ONLY_NODE_INDICES))) # Generate distinct colors

for i, node_idx in enumerate(LOAD_ONLY_NODE_INDICES):
    ax3.plot(time_axis, voltage_matrix[:, node_idx], label=f'Node {node_idx}', color=colors_load[i], linewidth=2)

# Add Limits
ax3.axhline(y=1.10, color='black', linestyle='--', linewidth=1.5, label='Upper Limit (1.10)')
ax3.axhline(y=0.90, color='black', linestyle='--', linewidth=1.5, label='Lower Limit (0.90)')
ax3.axhline(y=1.00, color='gray', linestyle=':', alpha=0.5)

ax3.set_ylabel("Voltage (p.u.)")
ax3.set_xlabel("Time (Hours)")
ax3.set_xticks(np.arange(0, 25, 4)) 
ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="Nodes")
ax3.grid(True, alpha=0.3)

plt.subplots_adjust(top=0.94, bottom=0.07,left=0.07, right=0.83, hspace=0.2)
#plt.tight_layout()
plt.savefig("solar_voltages.png", bbox_inches='tight')
print("✅ Saved 'solar_voltages.png'")

plt.show()