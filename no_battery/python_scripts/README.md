# No Battery Analysis Scripts

This folder contains scripts for analyzing VPP feeder behavior without battery storage.

## Main Script: plot_no_battery.py

**Primary script** for no-battery analysis. Can run in two modes:

### Mode 1: Full Analysis (with plots)
```bash
python plot_no_battery.py                    # Uses default scenario
python plot_no_battery.py heatwave_day       # Specific scenario
```

**Output:**
- Console statistics
- CSV data file: `../data_output/no_battery_results_<scenario>.csv`
- Power plots: `../plots/no_battery_scenario_<scenario>.png`
- Voltage plots: `../plots/no_battery_voltages_<scenario>.png`
- Summary file: `../documentation/SUMMARY_<scenario>.txt`

### Mode 2: Quick Test (no plots - faster)
```bash
python plot_no_battery.py --no-plots              # Quick test
python plot_no_battery.py heatwave_day --no-plots # Quick test specific scenario
```

**Output:**
- Console statistics only
- CSV data file
- No plots (faster for testing)

## Legacy Script: test_no_batteries.py

⚠️ **Deprecated** - Kept for backward compatibility only.

Use `plot_no_battery.py --no-plots` instead for the same functionality.

## Changing Scenarios

**Method 1: Edit the file**
Open `plot_no_battery.py` and change this line:
```python
DEFAULT_SCENARIO = "Next_Day_Forecast_21"  # <-- Edit this
```

**Method 2: Command line**
```bash
python plot_no_battery.py heatwave_day
```

### Available Scenarios

- `Next_Day_Forecast_21` - Load dip 00:00-02:00
- `heatwave_day` - Higher daytime load + reduced solar
- `solar_unavailable_day` - Solar set to 0 all day
- `cloudy_reduced_solar` - Solar reduced with variability
- `intermittent_solar_dropouts` - Random solar dropouts
- `weekend_low_load` - All loads scaled down
- `daytime_peak_load_day` - Extra midday load bump
- `evening_peak_load_day` - Extra evening load bump
- `load_higher_day` - All loads scaled up
- `solar_shifted_late` - Solar delayed (morning clouds)

## Output Organization

- **../data_output/** - CSV files with time-series data
- **../plots/** - PNG visualization files
- **../documentation/** - Text summary reports

## Requirements

- Python 3.x
- Dependencies from main project requirements.txt
- Access to RL_agent/vpp_env.py module
- Forecast data in ../../data/ folder

## Time-of-Use Pricing

Economic calculations use LKR (Sri Lankan Rupee) with time-based rates:
- **Day (6am-6pm):** Buy=35 LKR/kWh, Sell=19 LKR/kWh
- **Peak (6pm-11pm):** Buy=67 LKR/kWh, Sell=45 LKR/kWh  
- **Night (11pm-6am):** Buy=21 LKR/kWh, Sell=0 LKR/kWh
