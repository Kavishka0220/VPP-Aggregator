from vpp_env import UrbanVPPEnv
import numpy as np

# 1. Initialize the environment
env = UrbanVPPEnv()
env.reset()

# 2. Run one step to generate data
# Create a dummy action (e.g., all zeros)
dummy_action = np.zeros(3, dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(dummy_action)

# --- DEBUGGING SECTION ---
print("\n--- 🔍 VPP Environment Variables Inspection ---")

# List of internal variables you want to inspect
# These names must match "self.variable_name" in your vpp_env.py
vars_to_check = [
    "soc", 
    "node_battery_power_kw", 
    "remaining_demand", 
    "start_idx",
    "voltages", 
    "current_step",
    "bess_power"
]

for var_name in vars_to_check:
    # getattr(obj, name) gets the variable from the object by string name
    if hasattr(env, var_name):
        value = getattr(env, var_name)
        print(f"Name: {var_name}")
        print(f"Type: {type(value)}")
        print(f"Value: {value}")
        print("-" * 30)
    else:
        print(f"❌ Error: '{var_name}' not found in env.")

# Example: Accessing specific nested values
print("\n--- Specific Checks ---")
print(f"BESS SoC (Type: {type(env.soc[2])}): {env.soc[2]}")
print(f"Max Voltage (Type: {type(np.max(env.voltages))}): {np.max(env.voltages)}")