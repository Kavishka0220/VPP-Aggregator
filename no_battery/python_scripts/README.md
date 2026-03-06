# No Battery Analysis Scripts

This folder contains Python scripts for analyzing VPP feeder behavior without battery storage.

## Scripts

### 1. test_no_batteries.py
Basic test script that runs a single scenario and saves results.

**Usage:**
```bash
cd no_battery/python_scripts
python test_no_batteries.py
```

**Output:**
- CSV file saved to: `../data/no_battery_test_results.csv`
- Console summary of feeder performance

### 2. plot_no_battery.py
Comprehensive analysis script that runs multiple scenarios and generates detailed visualizations.

**Usage:**
```bash
cd no_battery/python_scripts
python plot_no_battery.py
```

**Outputs:**
- Plot files saved to: `../plots/`
- CSV data files saved to: `../data/`
- Summary reports saved to: `../documentation/`

## Available Scenarios

1. Default (next day forecast)
2. Daytime Peak Load Day
3. Evening Peak Load Day
4. Heatwave Day
5. Solar Unavailable Day
6. Weekend Low Load

## Output Organization

- **../data/** - CSV files with time-series data
- **../plots/** - PNG visualization files
- **../documentation/** - Text summary reports
- **../python_scripts/** - Python scripts (this folder)

## Requirements

- Python 3.x
- Dependencies from main project requirements.txt
- Access to RL_agent/vpp_env.py module
- Forecast data in ../../data/ folder
