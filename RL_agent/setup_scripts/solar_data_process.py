"""
Convert Solar Irradiance Forecast to Solar Generation by Node
Maps irradiance to generation based on panel capacities at each node
"""

import pandas as pd
import numpy as np
import os

# Configuration
DATA_PATH = r"d:\UoM\FYP\VPP-Aggregator\data"  # Absolute path
IRRADIANCE_FILE = os.path.join(DATA_PATH, "Solar_Irradiance_Forecast_Dec_11.csv")
OUTPUT_FILE = os.path.join(DATA_PATH, "forecast_scenarios", "solar_Next_Day_Forecast_21_nodes.csv")

# Node solar panel capacities (kW)
SOLAR_PANEL_CAPACITY = {
    3: 5.0,    # Node 3 has 5kW
    5: 5.0,    # Node 5 has 5kW
    7: 5.0,    # Node 7 has 5kW
    10: 5.0,   # Node 10 has 5kW
    11: 5.0,   # Node 11 has 5kW
    13: 5.0,   # Node 13 has 5kW
    15: 6.0,   # Node 15 has 6kW
    17: 5.0,   # Node 17 has 5kW
    18: 5.0,   # Node 18 has 5kW
    19: 5.0,   # Node 19 has 5kW
    20: 15.0,  # Node 20 has 15kW
}

# Standard Test Conditions (STC) irradiance
STC_IRRADIANCE = 1000.0  # W/m² (standard)

# Solar panel efficiency (typical for residential ~18%)
PANEL_EFFICIENCY = 0.95  # 95% efficiency (accounting for inverter, wiring losses)

def irradiance_to_generation(irradiance_wm2, panel_capacity_kw, efficiency=PANEL_EFFICIENCY, stc_irradiance=STC_IRRADIANCE):
    """
    Convert irradiance to solar generation
    
    Formula: Generation (kW) = (Irradiance / STC_Irradiance) * Panel_Capacity * Efficiency
    
    Args:
        irradiance_wm2: Solar irradiance in W/m²
        panel_capacity_kw: Panel capacity in kW
        efficiency: System efficiency (default 0.95)
        stc_irradiance: STC irradiance (default 1000 W/m²)
    
    Returns:
        Generation in kW
    """
    generation = (irradiance_wm2 / stc_irradiance) * panel_capacity_kw * efficiency
    return max(0.0, generation)  # Ensure no negative values

def main():
    print("[INFO] Loading irradiance forecast...")
    
    # Read irradiance forecast
    if not os.path.exists(IRRADIANCE_FILE):
        print(f"[ERROR] File not found: {IRRADIANCE_FILE}")
        return
    
    irradiance_df = pd.read_csv(IRRADIANCE_FILE)
    print(f"[OK] Loaded {len(irradiance_df)} timesteps of irradiance data")
    
    # Create output dataframe with 21 columns (nodes 0-20)
    solar_gen_df = pd.DataFrame(index=irradiance_df.index)
    
    # Initialize all columns with zeros
    for node_idx in range(21):
        col_name = f"Node_{node_idx}" if node_idx != 20 else "Node_20"
        solar_gen_df[col_name] = 0.0
    
    # Fill in solar generation for nodes with panels
    print("\n[INFO] Converting irradiance to generation...")
    for node_idx, capacity_kw in SOLAR_PANEL_CAPACITY.items():
        col_name = f"Node_{node_idx}" if node_idx != 20 else "Node_20"
        
        # Convert each irradiance value to generation
        solar_gen_df[col_name] = irradiance_df['Predicted_Irradiance'].apply(
            lambda irr: irradiance_to_generation(irr, capacity_kw)
        )
        
        print(f"  ✓ Node {node_idx}: {capacity_kw:.1f}kW panel")
        print(f"    Max generation: {solar_gen_df[col_name].max():.2f}kW")
        print(f"    Total daily energy: {solar_gen_df[col_name].sum() * 0.25 / 1000:.3f}kWh")
    
    # Reorder columns to be Node_0, Node_1, ..., Node_20
    column_order = [f"Node_{i}" if i != 20 else "Node_20" for i in range(21)]
    solar_gen_df = solar_gen_df[column_order]
    
    # Create output directory if not exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV (without index, without timestamp column)
    print(f"\n[INFO] Saving to: {OUTPUT_FILE}")
    solar_gen_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"[OK] Saved! File contains {len(solar_gen_df)} rows x {len(solar_gen_df.columns)} columns")
    print(f"\n[SUMMARY]")
    print(f"  Total solar capacity: {sum(SOLAR_PANEL_CAPACITY.values()):.1f}kW")
    print(f"  Peak generation: {solar_gen_df.sum(axis=1).max():.2f}kW")
    print(f"  Total daily energy: {solar_gen_df.sum().sum() * 0.25 / 1000:.2f}kWh")
    
    # Show sample
    print(f"\n[SAMPLE] First 5 timesteps:")
    print(solar_gen_df.head())

if __name__ == "__main__":
    main()
