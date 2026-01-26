import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
days = 1
intervals_per_day = 96 # 15 mins
total_steps = days * intervals_per_day
n_houses = 10
solar_indices = [0, 1, 2, 4, 6, 8] # The 6 Solar Houses

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")

os.makedirs(DATA_DIR, exist_ok=True)


# Create Time Index
start_time = pd.Timestamp("2026-01-01 00:00:00")
time_index = pd.date_range(start=start_time, periods=total_steps, freq="15min")

# --- 1. GENERATE COMMON SOLAR DATA (Identical Pattern) ---
solar_data = pd.DataFrame(index=time_index)

# A. Create the Master "Clear Sky" Curve (Sine Wave)
x = np.linspace(0, np.pi, 48) # 12 hours of sun
sun_profile = np.sin(x) 
full_day_sun = np.zeros(96)
full_day_sun[24:72] = sun_profile # Sun from 06:00 to 18:00
base_irradiance = np.tile(full_day_sun, days)

# B. Add "Common Cloud Noise" (Affects EVERYONE the same)
# This creates the "Common Solar" signal you asked for
cloud_noise = np.random.normal(0, 0.05, size=total_steps)
master_solar_curve = base_irradiance + cloud_noise
master_solar_curve = np.clip(master_solar_curve, 0, None)

# C. Apply to Houses (Scaling only, no pattern variation)
for i in range(n_houses):
    if i in solar_indices:
        # "Generation does not vary highly" -> Tight range (3.8 to 4.2 kW)
        panel_capacity = np.random.uniform(3.8, 4.2) 
        solar_data[f'House{i}'] = master_solar_curve * panel_capacity
    else:
        solar_data[f'House{i}'] = 0.0

solar_data.to_csv(os.path.join(DATA_DIR, 'solar_forecast_formatted.csv'), index=False)
print("✅ Solar Data: Common pattern generated (High correlation).")

# --- 2. GENERATE 10 SEPARATE LOADS (High Variation) ---
load_data = pd.DataFrame(index=time_index)

# A. Define a Base Behavior (Wake up -> Work -> Evening Peak -> Sleep)
daily_load_pattern = np.concatenate([
    np.full(24, 0.2),              # Night (0.2 kW)
    np.linspace(0.2, 0.8, 12),     # Morning Rise
    np.full(36, 0.3),              # Day Dip (Work)
    np.linspace(0.3, 1.2, 16),     # Evening Peak (High)
    np.linspace(1.2, 0.2, 8)       # Night Fall
])
base_load_profile = np.tile(daily_load_pattern, days)

for i in range(n_houses):
    # 1. Randomize Peak Consumption (Small house vs Big house)
    peak_scale = np.random.uniform(1.5, 4.0)
    
    # 2. Add "Behavioral Noise" (Random spikes per house)
    behavior_noise = np.random.normal(0, 0.15, size=total_steps)
    
    # 3. Time Shift (Some people come home at 5pm, some at 7pm)
    shift = np.random.randint(-6, 6) # +/- 1.5 hour shift
    house_base = np.roll(base_load_profile, shift)
    
    # Combine
    house_load = (house_base * peak_scale) + behavior_noise
    house_load = np.clip(house_load, 0.1, None) # Min load 100W
    
    load_data[f'House{i}'] = house_load

load_data.to_csv(os.path.join(DATA_DIR, 'load_forecast.csv'), index=False)
print("✅ Load Data: 10 Separate profiles generated.")

# --- PLOT TO CONFIRM ---
plt.figure(figsize=(12, 6))
# Plot Solar: All lines should be "parallel" / strictly correlated
for i in solar_indices:
    plt.plot(solar_data[f'House{i}'][:96], label=f'Solar {i}')
    
# Plot Load: Lines should be messy and distinct
for i in range(n_houses):
    plt.plot(load_data[f'House{i}'][:96], '--', linewidth=1, label=f'Load H{i}')

plt.title("Confirmation: Common Solar Pattern vs Separate Loads")
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.subplots_adjust(top=0.94, bottom=0.07, left=0.07, right=0.83, hspace=0.2)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title="Nodes")
plt.savefig("solar_load.png", bbox_inches='tight')
print("✅ Saved 'solar_load.png'")
plt.show()
# --- PLOT TOTAL LOAD AND SOLAR ---
plt.figure(figsize=(12, 6))

# Calculate totals
total_load = load_data.sum(axis=1)[:96]
total_solar = solar_data.sum(axis=1)[:96]

plt.plot(total_load, 'r-', linewidth=2, label='Total Load')
plt.plot(total_solar, 'g-', linewidth=2, label='Total Solar Generation')
plt.title("Total Load vs Total Solar Generation")
plt.xlabel("Time Interval (15-min)")
plt.ylabel("Power (kW)")
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.subplots_adjust(top=0.94, bottom=0.07, left=0.07, right=0.83, hspace=0.2)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title="Nodes")
plt.savefig("total_load_solar.png", bbox_inches='tight')
print("✅ Saved 'total_load_solar.png'")
plt.show()
