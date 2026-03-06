# No-Battery Scenario Analysis

This folder contains **all files related to no-battery feeder analysis** - scripts, data, plots, and documentation.

## Contents (27 files)

### Python Scripts (2)
- `plot_no_battery.py` - Main visualization script
- `test_no_batteries.py` - Testing script for no-battery scenarios

### Data Files (7 CSV)
- Scenario results with 15-minute timestep data
- Test results from openDSS simulations

### Plots (12 PNG)
- Main scenario plots (6 scenarios)
- Voltage analysis plots (6 scenarios)

### Documentation (6)
- README.md (this file)
- Summary text files for each scenario

## Purpose

These simulations show how the VPP feeder operates with:
- ✅ Solar PV generation
- ✅ Variable household loads  
- ❌ **NO battery storage** (batteries disabled)

This provides a baseline to compare against battery-enabled scenarios.

## Visualization Changes

Since batteries are disabled:
- ✅ Power generation and consumption plots
- ✅ Grid power exchange (import/export)
- ✅ Net demand analysis
- ✅ Voltage profiles for all nodes
- ❌ **Battery SoC plots removed** (not applicable without batteries)
- ❌ **Battery power plots removed** (not applicable without batteries)

## Scenarios Analyzed

### 1. **Default Scenario**
- Standard load and solar profiles
- Files: `no_battery_scenario.png`, `no_battery_voltages.png`, `no_battery_results.csv`
- Summary: `SUMMARY.txt`

### 2. **Daytime Peak Load Day**
- High daytime electricity demand
- Files: `no_battery_scenario_daytime_peak_load_day.*`
- Summary: `SUMMARY_daytime_peak_load_day.txt`

### 3. **Evening Peak Load Day**
- Peak demand during evening hours
- Files: `no_battery_scenario_evening_peak_load_day.*`
- Summary: `SUMMARY_evening_peak_load_day.txt`

### 4. **Heatwave Day**
- High cooling loads, high solar generation
- Files: `no_battery_scenario_heatwave_day.*`
- Summary: `SUMMARY_heatwave_day.txt`

### 5. **Solar Unavailable Day**
- No solar generation (cloudy/rainy day)
- Worst-case scenario for grid dependency
- Files: `no_battery_scenario_solar_unavailable_day.*`

### 6. **Weekend Low Load**
- Reduced household consumption
- High solar export potential
- Files: `no_battery_scenario_weekend_low_load.*`
- Summary: `SUMMARY_weekend_low_load.txt`

## Key Plots

Each scenario includes:

1. **Main Scenario Plot** (`no_battery_scenario_*.png`) - 3 panels:
   - Power generation and consumption
   - Grid power exchange (import/export)
   - Net demand (load - solar)

2. **Voltage Analysis** (`no_battery_voltages_*.png`)
   - Voltage profiles for all 11 nodes
   - Min/max voltage range
   - Voltage limit compliance check

3. **Data Export** (`no_battery_results_*.csv`)
   - Time-series data (15-minute intervals)
   - All metrics for further analysis

4. **Text Summary** (`SUMMARY_*.txt`)
   - Statistical summary
   - Energy and economic metrics
   - Voltage statistics

## Usage

To regenerate or add more scenarios:

```bash
# From VPP-Aggregator root:
cd "no battery"
python plot_no_battery.py                    # Default scenario
python plot_no_battery.py heatwave_day       # Specific scenario
python plot_no_battery.py <scenario_name>    # Any scenario from data/forecast_scenarios/
```

All outputs will be saved to this folder: `VPP-Aggregator/no battery/`

### Running Test Script

```bash
cd "no battery"
python test_no_batteries.py
```

## Next Steps

Compare these baseline results with battery-enabled simulations to quantify:
- Peak demand reduction
- Solar self-consumption improvement
- Grid cost savings
- Voltage regulation improvement
- Battery ROI analysis

---

*Generated with `plot_no_battery.py`*
