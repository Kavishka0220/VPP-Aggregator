import pandas as pd
import numpy as np

# 1. Load the raw weather data
# Add the slash after the dots
df = pd.read_csv('../data/solar_forecast.csv') # Make sure this path is correct

# 2. Convert 'datetime' to actual time format and set as index
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# 3. Resample from 1 Hour -> 15 Minutes (Interpolate)
# This fills in the gaps between 1:00 and 2:00 smoothly
df_15min = df['solarradiation'].resample('15min').interpolate(method='linear')

# 4. Generate Data for 10 Houses
# Formula: Power (kW) = Radiation (W/m2) * Area (m2) * Efficiency / 1000
solar_data = pd.DataFrame(index=df_15min.index)

np.random.seed(42) # For consistent results

for i in range(1, 11):
    # Random Area for each house (e.g., between 15m2 and 30m2)
    area = np.random.uniform(15, 30)
    # Efficiency of panels (e.g., 18%)
    efficiency = 0.18 
    
    # Calculate Power in kW
    col_name = f'House{i}'
    # We add a little random noise (0.9 to 1.1) so houses aren't identical
    noise = np.random.uniform(0.9, 1.1, size=len(df_15min))
    
    solar_data[col_name] = (df_15min * area * efficiency / 1000) * noise
    
    # Clip negative values to 0 (just in case)
    solar_data[col_name] = solar_data[col_name].clip(lower=0)

# 5. Save to the file your Agent expects
solar_data.to_csv('../data/solar_forecast_formatted.csv', index=False)
print("✅ Converted! Saved as 'data/solar_forecast_formatted.csv'")
print(solar_data.head())