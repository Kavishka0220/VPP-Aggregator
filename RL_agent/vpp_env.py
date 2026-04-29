import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to find openDSS module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from openDSS.run_opendss import VPPDSSRunner

class UrbanVPPEnv(gym.Env):
    """
    Final Thesis VPP Environment
    - Constraints: Voltage must be between 0.94 and 1.06 p.u.
    - Inputs: Common Solar, 21 Loads, 22 Node Voltages, 3 SoCs, Time.
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, data_path="./data", scenario_name=None, start_index=None, verbose=False):
        super(UrbanVPPEnv, self).__init__()
        
        # Testing configuration
        self.verbose = verbose
        self.default_start_index = start_index
        self.scenario_name = scenario_name

        # Initialize OpenDSS Runner
        dss_file = os.path.join(parent_dir, "openDSS", "feeder_houses.dss")
        self.dss_runner = VPPDSSRunner(dss_file)
        self.dss_runner.compile()  # Compile circuit at initialization
        
        # --- 1. SYSTEM CONFIGURATION ---
        self.n_nodes = 22  # 21 Houses (L0-L20) + 1 BESS Node (21)
        self.solar_indices = [3, 5, 7, 10, 11, 13, 15, 17, 18, 19, 20]  # 11 nodes have solar panels
        
        # Validate solar indices
        if not all(0 <= idx < 21 for idx in self.solar_indices):
            raise ValueError(f"Solar indices must be in range [0, 20]. Got: {self.solar_indices}")
        
        # Which nodes have Batteries?
        self.home_batt_indices = [3, 5]  # Home Batteries at nodes 3 & 5
        self.bess_index = 21  # BESS at node 21 (end of feeder)
        
        # Map actions to physical nodes: Action[0]->Node3, Action[1]->Node5, Action[2]->Node21(BESS)
        self.storage_map = self.home_batt_indices + [self.bess_index]
        self.n_storage_units = len(self.storage_map)  # Calculate from storage_map, not hardcoded
        
        # Specs
        self.home_batt_cap = 13.5 # kWh
        self.bess_cap = 200.0 # kWh
        self.home_batt_power = 5.0 # kW
        self.bess_power = 40.0 # kW

        # --- Battery Ramp Rate Limits (kW per 15 min step) ---
        self.home_batt_ramp = 2.0     # kW / step
        self.bess_batt_ramp = 10.0    # kW / step

        # --- 2. ACTION SPACE ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # --- 3. OBSERVATION SPACE ---
        # 1(Solar) + 21(Loads) + 22(All Node Voltages) + 3(SoCs) + 4(Time) = 51
        self.obs_size = 51
        self.observation_space = spaces.Box(
            #low=-np.inf, high=np.inf,
            low=-5, high=5,
            shape=(self.obs_size,), dtype=np.float32
        )

        # State Variables
        self.state = None
        self.current_step = 0
        self.max_steps = 96
        self.soc = np.ones(self.n_storage_units) * 0.5 
        self.prev_batt_power = np.zeros(self.n_storage_units)
        self.prev_grid_net_power = 0.0  # Track previous grid net import/export for smoothing
        # We need 22 voltage values internally (for 3-phase monitoring)
        self.voltages = np.ones(self.n_nodes, dtype=np.float32)      # Min voltage per bus (RL sees this)
        self.voltages_min = np.ones(self.n_nodes, dtype=np.float32)  # For undervoltage checking
        self.voltages_max = np.ones(self.n_nodes, dtype=np.float32)  # For overvoltage checking

        # --- LOAD DATA ---
        try:
            if scenario_name:
                scenario_folder = os.path.join(data_path, "forecast_scenarios")
                solar_file = os.path.join(scenario_folder, f"solar_{scenario_name}.csv")
                load_file = os.path.join(scenario_folder, f"load_{scenario_name}.csv")
                
                # Verify files exist
                if not os.path.exists(solar_file):
                    raise FileNotFoundError(f"Scenario file not found: {solar_file}")
                if not os.path.exists(load_file):
                    raise FileNotFoundError(f"Scenario file not found: {load_file}")
                
                print(f"[INFO] Loading Scenario: {scenario_name}")
                if self.verbose:
                    print(f"[DEBUG] Solar file: {solar_file}")
                    print(f"[DEBUG] Load file: {load_file}")
                self.solar_df = pd.read_csv(solar_file)
                self.load_df = pd.read_csv(load_file)
            else:
                # These files contain 21 columns (House 0 to House 20)
                solar_file = os.path.join(data_path, "solar_forecast_formatted.csv")
                load_file = os.path.join(data_path, "load_forecast.csv")
                print(f"[INFO] Loading default data files")
                if self.verbose:
                    print(f"[DEBUG] Solar file: {solar_file}")
                    print(f"[DEBUG] Load file: {load_file}")
                self.solar_df = pd.read_csv(solar_file)
                self.load_df = pd.read_csv(load_file)
            
            # Clean and validate data - handle various CSV formats
            # Process solar dataframe
            # Drop timestamp/date columns
            timestamp_cols = [col for col in self.solar_df.columns if col.lower() in ['timestamp', 'time', 'date', 'datetime']]
            if timestamp_cols:
                self.solar_df = self.solar_df.drop(columns=timestamp_cols)
            
            # Convert all columns to numeric
            for col in self.solar_df.columns:
                self.solar_df[col] = pd.to_numeric(self.solar_df[col], errors='coerce')
            
            # Drop any completely empty columns
            self.solar_df = self.solar_df.dropna(axis=1, how='all')
            
            # Select only first 21 columns if more exist
            if self.solar_df.shape[1] > 21:
                print(f"[WARNING] solar has {self.solar_df.shape[1]} columns, using first 21")
                self.solar_df = self.solar_df.iloc[:, :21]
            elif self.solar_df.shape[1] < 21:
                raise ValueError(f"solar has only {self.solar_df.shape[1]} columns, need 21")
            
            # Process load dataframe
            # Drop timestamp/date columns
            timestamp_cols = [col for col in self.load_df.columns if col.lower() in ['timestamp', 'time', 'date', 'datetime']]
            if timestamp_cols:
                self.load_df = self.load_df.drop(columns=timestamp_cols)
            
            # Convert all columns to numeric
            for col in self.load_df.columns:
                self.load_df[col] = pd.to_numeric(self.load_df[col], errors='coerce')
            
            # Drop any completely empty columns
            self.load_df = self.load_df.dropna(axis=1, how='all')
            
            # Select only first 21 columns if more exist
            if self.load_df.shape[1] > 21:
                print(f"[WARNING] load has {self.load_df.shape[1]} columns, using first 21")
                self.load_df = self.load_df.iloc[:, :21]
            elif self.load_df.shape[1] < 21:
                raise ValueError(f"load has only {self.load_df.shape[1]} columns, need 21")
            
            # Final validation
            if self.solar_df.shape[1] != 21 or self.load_df.shape[1] != 21:
                raise ValueError(f"Data must have 21 columns. Got solar: {self.solar_df.shape[1]}, load: {self.load_df.shape[1]}")
            
            # Handle potential length mismatch (e.g., Load is 1 day, Solar is 1 year)
            len_solar = len(self.solar_df)
            len_load = len(self.load_df)
            
            if len_solar != len_load:
                print(f"[WARNING] Data length mismatch. Solar: {len_solar}, Load: {len_load}")
                
                # If Load is just 1 day (96 steps) and Solar is many days
                if len_load == 96 and len_solar > 96:
                    print(f"[INFO] Repeating Load profile to match Solar data length.")
                    dataset_days = int(np.ceil(len_solar / 96))
                    self.load_df = pd.concat([self.load_df] * dataset_days, ignore_index=True)
                    self.load_df = self.load_df.iloc[:len_solar] # Trim to exact match
                
                # If Solar is just 1 day and Load is many days
                elif len_solar == 96 and len_load > 96:
                     print(f"[INFO] Repeating Solar profile to match Load data length.")
                     dataset_days = int(np.ceil(len_load / 96))
                     self.solar_df = pd.concat([self.solar_df] * dataset_days, ignore_index=True)
                     self.solar_df = self.solar_df.iloc[:len_load]
                
                # Update lengths
                len_solar = len(self.solar_df)
                len_load = len(self.load_df)
                
                # If still mismatched (e.g. random lengths), trim to minimum
                if len_solar != len_load:
                     min_len = min(len_solar, len_load)
                     print(f"[WARNING] Trimming to minimum common length: {min_len}")
                     self.solar_df = self.solar_df.iloc[:min_len]
                     self.load_df = self.load_df.iloc[:min_len]
            
            if len(self.solar_df) < self.max_steps:
                raise ValueError(f"Data must have at least {self.max_steps} rows. Got: {len(self.solar_df)}")
            
            # Log final verification
            scenario_info = f"Scenario: {scenario_name}" if scenario_name else "Default data"
            print(f"[OK] Data Loaded Successfully! {scenario_info}")
            if self.verbose:
                print(f"[OK] Solar shape: {self.solar_df.shape}, Load shape: {self.load_df.shape}")
                print(f"[OK] Final Length: {len(self.solar_df)}")
                print(f"[DEBUG] Solar sample (first row): {self.solar_df.iloc[0].values[:5]}...")
                print(f"[DEBUG] Load sample (first row): {self.load_df.iloc[0].values[:5]}...")
        except FileNotFoundError as e:
            # Fallback dummy data
            print(f"[WARNING] File loading error: {e}")
            print("[WARNING] Using dummy random data due to file loading error.")
            print("[WARNING] This is a fallback - verify your data_path and scenario_name!")
            self.solar_df = pd.DataFrame(np.random.rand(1000, 21) * 5.0)
            self.load_df = pd.DataFrame(np.random.rand(1000, 21) * 3.0)    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        # Initialize SoC randomly within safe operating range (0.3 to 0.7) for robustness
        # This simulates realistic starting conditions and improves training generalization
        #self.soc = np.random.uniform(0.3, 0.7, size=3)
        self.soc = np.full(3, 0.2)
        self.prev_batt_power = np.zeros(self.n_storage_units)
        self.prev_grid_net_power = 0.0  # Reset grid tracking
        # Reset voltages to 1.0 p.u. (nominal)
        self.voltages = np.ones(self.n_nodes, dtype=np.float32)
        self.voltages_min = np.ones(self.n_nodes, dtype=np.float32)
        self.voltages_max = np.ones(self.n_nodes, dtype=np.float32)
        
        # Pick random day or use provided start index
        if options is not None and "start_step" in options:
            self.start_idx = options["start_step"]
            if "episode_len" in options:
                self.max_steps = options["episode_len"]
            print(f"[CONFIG] Starting at step {self.start_idx}, length {self.max_steps}")
        elif self.default_start_index is not None:
             self.start_idx = self.default_start_index
             print(f"[CONFIG] Starting at FIXED default step {self.start_idx}")
        else:
            max_start = len(self.solar_df) - self.max_steps
            if max_start > 0:
                self.start_idx = np.random.randint(0, max_start)
            else:
                self.start_idx = 0  # Use all available data if it matches episode length

        # Slice Data
        self.solar_episode = self.solar_df.iloc[self.start_idx : self.start_idx + self.max_steps].values
        self.load_episode = self.load_df.iloc[self.start_idx : self.start_idx + self.max_steps].values
        
        # Apply Mask (Only keep solar for nodes with panels: 3,5,7,10,11,13,15,17,18,19,20)
        # Even if CSV has data for all houses, we zero out houses without panels
        self.solar_mask = np.zeros(21)
        self.solar_mask[self.solar_indices] = 1.0
        self.solar_episode = self.solar_episode * self.solar_mask

        return self._get_obs(), {}

    def _max_charge_power_from_headroom(self, soc, cap, p_max):
        """Maximum feasible charging power for one 15-minute step."""
        if soc >= 0.8:
            return 0.0
        headroom_kwh = (0.8 - soc) * cap
        return min(p_max, headroom_kwh / (0.25 * 0.95))

    def _max_discharge_power_from_soc(self, soc, cap, p_max):
        """Maximum feasible discharge power for one 15-minute step."""
        if soc <= 0.2:
            return 0.0
        deliverable_kwh = (soc - 0.2) * cap * 0.95
        return min(p_max, deliverable_kwh / 0.25)

    def _compute_peak_import_target(self, hour):
        """Spread BESS energy across the remaining peak window to flatten imports."""
        if not (18 <= hour < 23):
            return None

        peak_end_step = min(self.max_steps, self.current_step + int((23 - hour) * 4))
        if peak_end_step <= self.current_step:
            return 0.0

        future_load = np.sum(self.load_episode[self.current_step:peak_end_step], axis=1)
        future_solar = np.sum(self.solar_episode[self.current_step:peak_end_step], axis=1)
        future_net_demand = np.maximum(0.0, future_load - future_solar)

        if future_net_demand.size == 0:
            return 0.0

        deliverable_energy_kwh = max(0.0, (self.soc[2] - 0.2) * self.bess_cap * 0.95)
        if deliverable_energy_kwh <= 0.0:
            return float(np.max(future_net_demand))

        low = 0.0
        high = float(np.max(future_net_demand))
        for _ in range(24):
            mid = 0.5 * (low + high)
            shaved_profile = np.minimum(np.maximum(future_net_demand - mid, 0.0), self.bess_power)
            required_energy_kwh = np.sum(shaved_profile) * 0.25
            if required_energy_kwh > deliverable_energy_kwh:
                low = mid
            else:
                high = mid

        return high
    
    def step(self, action):
        # --- 1. GET DATA FIRST (Moved to top) ---
        # We must know the Total Load/Solar BEFORE deciding battery actions
        full_solar_profile = np.zeros(self.n_nodes) 
        full_load_profile = np.zeros(self.n_nodes) 

        # Fill with current step data
        full_solar_profile[:21] = self.solar_episode[self.current_step]
        full_load_profile[:21] = self.load_episode[self.current_step]

        # Calculate Limits
        total_load = np.sum(full_load_profile)
        total_solar = np.sum(full_solar_profile)
        
        # Calculate current hour for time-based logic
        hour = (self.current_step % 96) / 4  # 15-min steps → hours
        
        # Discharge Limit: Batteries fill the gap between Load and Solar.
        # If Solar > Load, limit is 0 (No discharge allowed).
        net_demand = max(0.0, total_load - total_solar)
        net_solar_surplus = total_solar - total_load # Positive if surplus
        home_solar_surplus = full_solar_profile - full_load_profile  # Per-house surplus
        self.remaining_demand = net_demand # Decreases as we iterate batteries
        remaining_solar_surplus = max(0.0, net_solar_surplus)
        peak_import_target = self._compute_peak_import_target(hour)
        
        # --- AUTOMATIC BESS CHARGING LOGIC ---
        action_modified = action.copy()
        bess_action_idx = 2  # BESS is the third storage unit
        bess_soc = self.soc[bess_action_idx]
        
        # Strategy 1: Charge from solar surplus when available (PRIORITY)
        if net_solar_surplus > 0 and bess_soc < 0.8:
            # Solar surplus available and BESS not full - FORCE charging from solar
            # This overrides RL agent action to prevent solar waste
            # During daytime, charge at higher intensity to absorb excess generation
            if 6 <= hour < 18:  # Daytime
                charge_intensity = min(1.0, net_solar_surplus / (0.8 * self.bess_power))  # More aggressive charging (80% of power cap)
            else:
                charge_intensity = min(1.0, net_solar_surplus / self.bess_power)
            action_modified[bess_action_idx] = -charge_intensity  # Negative = charging
            if self.verbose:
                print(f"[SOLAR_PRIORITY] Hour {(self.current_step % 96)/4:.1f}: Forced BESS charge from solar surplus {net_solar_surplus:.2f}kW (SoC {bess_soc:.2f}) intensity={charge_intensity:.2f}")
        elif net_solar_surplus > 0 and bess_soc >= 0.8:
            # Solar surplus but BESS is FULL - this excess goes to grid or home batteries
            if self.verbose:
                print(f"[SOLAR_WASTE] Hour {(self.current_step % 96)/4:.1f}: Solar surplus {net_solar_surplus:.2f}kW but BESS FULL at SoC {bess_soc:.2f}")
        elif net_solar_surplus <= 0 and 6 <= hour < 18:
            # Daytime but NO solar surplus - BESS trying to discharge (should be blocked!)
            if self.verbose:
                print(f"[NO_SOLAR] Hour {(self.current_step % 96)/4:.1f}: No surplus (Solar={total_solar:.2f}, Load={total_load:.2f}), solar_surplus={net_solar_surplus:.2f}")

        # Strategy 1c: During peak hours, vary BESS discharge to flatten grid imports.
        if peak_import_target is not None:
            shaped_bess_discharge = np.clip(net_demand - peak_import_target, 0.0, self.bess_power)
            max_bess_discharge = self._max_discharge_power_from_soc(
                bess_soc, self.bess_cap, self.bess_power
            )
            shaped_bess_discharge = min(shaped_bess_discharge, max_bess_discharge)
            action_modified[bess_action_idx] = shaped_bess_discharge / self.bess_power if self.bess_power > 0 else 0.0
            if self.verbose:
                print(f"[PEAK_SHAPING] Hour {hour:.1f}: Target import {peak_import_target:.2f}kW, BESS discharge {shaped_bess_discharge:.2f}kW")
        
        # Strategy 1b: Also charge HOME batteries from solar surplus during daytime
        if 6 <= hour < 18 and net_solar_surplus > 0:
            # During daytime with solar surplus - force home batteries to charge
            home_batt_avg_soc = np.mean([self.soc[0], self.soc[1]])
            if home_batt_avg_soc < 0.8:
                # Home batteries not full - share solar surplus with them
                # Reduce BESS charge intensity to allow home batteries to charge from solar
                total_home_batt_soc = self.soc[0] + self.soc[1]
                home_charge_share = min(0.7, net_solar_surplus / 8.0)  # Increased from 0.5 to 0.7 (70% of surplus)
                
                for home_batt_idx, node_idx in enumerate(self.home_batt_indices):
                    if self.soc[home_batt_idx] < 0.80:  # Increased threshold
                        # Force home battery to charge from solar with higher intensity
                        home_charge_intensity = min(0.9, home_charge_share / self.home_batt_power)  # Increased from 0.6 to 0.9
                        action_modified[home_batt_idx] = -home_charge_intensity
                        if self.verbose:
                            print(f"[HOME_SOLAR] Hour {(self.current_step % 96)/4:.1f}: Forcing Home Battery {home_batt_idx} to charge from solar at {home_charge_intensity:.2f} intensity")
        
        # Strategy 2: Predictive charging from grid during cheap off-peak rates (12pm-6am, 21 LKR)
        # Calculate if today's solar surplus will be sufficient to fully charge BESS
        # NOTE: Charging blocked between 11pm-12am
        elif hour < 6:
            # Look ahead to predict today's daytime solar surplus
            steps_remaining = self.max_steps - self.current_step
            steps_until_peak = min(steps_remaining, int((18 - hour) * 4))  # Until peak (6pm)
            
            if steps_until_peak > 4:  # Need at least 1 hour of data to predict
                # Get future solar and load data
                future_solar = self.solar_episode[self.current_step:self.current_step + steps_until_peak]
                future_load = self.load_episode[self.current_step:self.current_step + steps_until_peak]
                
                # Calculate total expected solar surplus during daytime
                future_solar_total = np.sum(future_solar)
                future_load_total = np.sum(future_load)
                expected_daytime_surplus = future_solar_total - future_load_total
                
                if expected_daytime_surplus > 0:
                    # Convert surplus energy to how much it can charge BESS
                    # Each 15-min step with surplus can charge: surplus_power * 0.25 hours * efficiency
                    # Simplified: assume average surplus is spread across daytime hours
                    expected_surplus_energy = expected_daytime_surplus * 0.25 * 0.95  # kWh with efficiency
                    
                    # Calculate BESS charging need (from current SoC 0.2 to target 0.8)
                    target_soc = 0.8
                    min_soc = 0.2  # Realistic minimum starting SoC
                    current_energy = bess_soc * self.bess_cap  # Current energy in kWh
                    target_energy = target_soc * self.bess_cap  # Target energy in kWh
                    energy_needed = max(0, target_energy - current_energy)  # How much energy needed
                    
                    # KEY LOGIC: Only charge during off-peak if daytime solar is INSUFFICIENT
                    # If daytime solar surplus >= energy needed, DON'T charge from grid during off-peak
                    if expected_surplus_energy >= energy_needed:
                        # Daytime solar is ENOUGH to charge battery - don't consume grid power during off-peak
                        action_modified[bess_action_idx] = 0.0  # NO charging from grid
                    else:
                        # Daytime solar is INSUFFICIENT - charge from grid during off-peak to bridge deficit
                        energy_deficit = energy_needed - expected_surplus_energy
                        
                        # Calculate remaining off-peak hours for charging (11pm-6am)
                        if hour >= 23:
                            # From 11pm to midnight + midnight to 6am
                            steps_until_daytime = int((24 - hour) * 4 + 6 * 4)
                        else:
                            # Already past midnight, just count to 6am
                            steps_until_daytime = int((6 - hour) * 4)
                        
                        if steps_until_daytime > 0:
                            # Spread deficit charging across remaining off-peak hours
                            power_per_step = energy_deficit / (steps_until_daytime * 0.25)
                            charge_intensity = min(1.0, power_per_step / self.bess_power)
                        else:
                            charge_intensity = 0.5
                        action_modified[bess_action_idx] = -charge_intensity
                else:
                    # No daytime solar surplus expected - charge from grid during off-peak
                    # For solar unavailable days, MUST charge BESS fully since no daytime solar backup
                    if bess_soc < 0.8:
                        # Calculate how much charge is needed (target 0.8 for peak discharge)
                        energy_deficit = (0.8 - bess_soc) * self.bess_cap  # kWh needed
                        
                        # Calculate remaining off-peak hours for charging (11pm-6am = 7 hours max)
                        if hour >= 23:
                            steps_until_daytime = int((24 - hour) * 4 + 6 * 4)
                        else:
                            steps_until_daytime = int((6 - hour) * 4)
                        
                        if steps_until_daytime > 0:
                            # Spread charging across remaining off-peak hours
                            power_per_step = energy_deficit / (steps_until_daytime * 0.25)
                            charge_intensity = min(1.0, power_per_step / self.bess_power)
                            action_modified[bess_action_idx] = -charge_intensity
                        else:
                            action_modified[bess_action_idx] = -0.8  # Strong charge if close to daytime
                    else:
                        action_modified[bess_action_idx] = 0.0  # Don't charge if already sufficient
            else:
                # Insufficient lookahead data - fallback to SoC-based charging
                if bess_soc < 0.4:
                    action_modified[bess_action_idx] = -0.8  # Strong charge when low
                elif bess_soc < 0.6:
                    action_modified[bess_action_idx] = -0.5  # Moderate charge
        
        # --- 2. PHYSICS: APPLY ACTIONS ---
        # Create an array of size 11 for the grid physics
        self.node_battery_power_kw = np.zeros(self.n_nodes)
        
        # Store previous battery power for cycling cost calculation
        prev_batt_power_copy = self.prev_batt_power.copy()
        
        for i, node_idx in enumerate(self.storage_map):
            
            is_bess = (node_idx == self.bess_index)

            p_max = self.bess_power if is_bess else self.home_batt_power
            cap = self.bess_cap if is_bess else self.home_batt_cap
            ramp = self.bess_batt_ramp if is_bess else self.home_batt_ramp
            desired_power = action_modified[i] * p_max # Convert normalized action [-1,1] → real power (kW)
                   
            # --- CONSTRAINT 1: SoC Limits (0.2 - 0.8 safe zone)
            # Check this early to avoid other constraints with depleted batteries
            if self.soc[i] <= 0.2 and desired_power > 0:
                desired_power = 0.0  # Prevent discharge when battery low
            if self.soc[i] >= 0.8 and desired_power < 0:
                desired_power = 0.0  # Prevent charging when battery full

            # --- CONSTRAINT 2: BESS CHARGING STRATEGY ---
            # PRIORITY 1: Charge from solar surplus during daytime (BEFORE off-peak grid)
            # PRIORITY 2: Only use off-peak grid if solar insufficient
            if is_bess and desired_power < 0:  # BESS trying to charge
                # Always block charging during 11pm-12am transition
                if 23 <= hour:
                    desired_power = 0.0
                # PRIORITY: If there's solar surplus available, MUST use it, NOT grid
                # Force solar charging during daytime (6am-6pm) when solar surplus exists
                elif 6 <= hour < 18 and remaining_solar_surplus > 0:
                    # Solar surplus during daytime - charge from solar ONLY, not grid
                    if self.soc[i] < 0.8:
                        available_charge = min(
                            remaining_solar_surplus,
                            self._max_charge_power_from_headroom(self.soc[i], cap, p_max),
                        )
                        desired_power = -available_charge
                        if self.verbose:
                            print(f"[FORCE_SOLAR_CHARGE] Hour {hour:.1f}: BESS absorbing {available_charge:.2f}kW of remaining solar surplus")
                    else:
                        desired_power = 0.0  # BESS full, no more charging
                # Block grid charging outside off-peak if solar is insufficient
                elif net_solar_surplus <= 0 and 6 <= hour < 18:
                    desired_power = 0.0  # Daytime without solar - only off-peak allowed
                    
            # CRITICAL: Prevent BESS DISCHARGE when there's solar surplus during daytime
            # Stay charged for peak hours instead of wasting solar
            if is_bess and desired_power > 0 and 6 <= hour < 18:  # BESS trying to discharge during daytime
                if net_solar_surplus > 0:
                    # Solar surplus available - BLOCK discharge, save battery for peak hours
                    desired_power = 0.0
                    if self.verbose:
                        print(f"[BLOCK_DISCHARGE] Hour {(self.current_step % 96)/4:.1f}: Blocked BESS discharge during daytime with solar surplus {net_solar_surplus:.2f}kW")
            
            # NEW: Limit BESS discharge power to prevent overvoltage at end-of-feeder
            if is_bess and desired_power > 0:
                # DAYTIME: Reduce discharge intensity if solar production is high (leads to overvoltage)
                if total_solar > 0.8 * self.remaining_demand:  # High solar relative to demand
                    # Limit BESS discharge to avoid pushing voltage up
                    max_bess_discharge = 0.2 * p_max  # Cap at 20% of max power
                    desired_power = min(desired_power, max_bess_discharge)
                    if self.verbose:
                        print(f"[LIMIT_DISCHARGE] Hour {(self.current_step % 96)/4:.1f}: Limited BESS discharge to {desired_power:.1f}kW (high solar)")
            
            # NEW: Limit solar + BESS combined injection to prevent overvoltage
            if total_solar + np.maximum(0, desired_power) > total_load + 1.5:  # Even stricter threshold (was 2.0)
                # Too much generation - reduce battery discharge more aggressively
                excess = total_solar + np.maximum(0, desired_power) - total_load - 1.5
                if is_bess and desired_power > 0:
                    # During daytime, block discharge entirely if excess is significant
                    if 6 <= hour < 18 and excess > 0.5:
                        desired_power = 0.0  # Block BESS discharge during daytime with significant excess
                        if self.verbose:
                            print(f"[BLOCK_EXCESS] Hour {hour:.1f}: Blocked BESS discharge, excess {excess:.2f}kW")
                    else:
                        desired_power = max(0, desired_power - excess * 0.8)  # More aggressive reduction (was 0.5)
            
            # --- CONSTRAINT 3: Home Battery Daytime Solar Charging ---
            # AGGRESSIVE PRIORITY: Home batteries at Nodes 3&5 MUST absorb solar locally
            # This prevents solar from flowing all the way to BESS node (end-of-feeder overvoltage)
            if not is_bess and 6 <= hour < 18 and desired_power < 0:  # Home batt trying to charge
                if net_solar_surplus > 0:
                    available_charge = min(
                        remaining_solar_surplus,
                        self._max_charge_power_from_headroom(self.soc[i], cap, p_max),
                    )
                    desired_power = -available_charge
                    if self.verbose:
                        print(f"[HOME_SOLAR_LOCAL] Hour {hour:.1f}: Node {node_idx} absorbing {available_charge:.2f}kW before export")
                elif net_solar_surplus <= 0:
                    # No solar surplus available - block daytime grid charging
                    desired_power = 0.0  # Block daytime grid charging for home batteries
            
            # --- CONSTRAINT 4: Home Battery Discharge Strategy ---
            # Home batteries should ONLY discharge during peak hours (6pm-11pm, 67 LKR)
            # STRICTLY block daytime discharge to prevent overvoltage from solar + battery combination
            if not is_bess and desired_power > 0:  # Home battery trying to discharge
                if not (18 <= hour < 23):  # NOT during peak hours (6pm-11pm)
                    desired_power = 0.0  # Block discharge outside peak - save for expensive hours
            
            # --- CONSTRAINT 5: Block all battery charging between 11pm-12am ---
            if 18 <= hour < 24 and desired_power < 0:
                desired_power = 0.0  # No charging allowed between 11pm-12am
            
            # Limit discharge to actual remaining demand
            if desired_power > 0:
                desired_power = min(desired_power, self.remaining_demand)
                self.remaining_demand -= desired_power

            # --- RAMP RATE LIMITING ---
            # Prevent sudden power changes (use 'ramp' already calculated at line 157)
            delta_p = desired_power - self.prev_batt_power[i]
            delta_p = np.clip(delta_p, -ramp, ramp)
            final_power = self.prev_batt_power[i] + delta_p
            
            # Ensure final power doesn't exceed physical limits
            final_power = np.clip(final_power, -p_max, p_max)
            
            # Final voltage safety check: trim discharge only if local voltage is already high.
            if is_bess and final_power > 0 and self.voltages[self.bess_index] > 1.03:
                voltage_factor = max(0.2, min(1.0, (1.06 - self.voltages[self.bess_index]) / (1.06 - 1.03)))
                final_power = final_power * voltage_factor
                if self.verbose and voltage_factor < 1.0:
                    print(f"[VOLTAGE_SAFETY_CHECK] BESS voltage {self.voltages[self.bess_index]:.4f}p.u., reduced discharge to {final_power:.1f}kW (factor {voltage_factor:.1%})")
            
            # Update SoC (0.25 hour = 15 min timestep)
            eff = 0.95
            if final_power >= 0:  # Discharging
                # Battery loses energy: SoC decreases
                # Efficiency loss means battery gives up more than grid receives
                energy_lost = (final_power * 0.25) / eff
                self.soc[i] -= energy_lost / cap
            else:  # Charging (final_power < 0)
                # Battery gains energy: SoC increases (but final_power is negative, so -= increases SoC)
                # Efficiency loss means battery stores less than grid provides
                energy_gained = final_power * 0.25 * eff  # Negative value
                self.soc[i] -= energy_gained / cap  # -= negative = increase
            
            # Enforce strict SOC limits: 0.2 to 0.8 for battery health
            self.soc[i] = np.clip(self.soc[i], 0.2, 0.8)

            if 6 <= hour < 18 and final_power < 0 and remaining_solar_surplus > 0:
                remaining_solar_surplus = max(
                    0.0,
                    remaining_solar_surplus - min(-final_power, remaining_solar_surplus),
                )
            
            self.node_battery_power_kw[node_idx] = final_power
            self.prev_batt_power[i] = final_power


        # --- 3. PHYSICS: CALCULATE VOLTAGES (OpenDSS) ---
        
        # Calculate net power injection at ALL 11 nodes (Generation - Load + Battery)
        self.net_injection = full_solar_profile + self.node_battery_power_kw - full_load_profile

        # Prepare inputs for OpenDSS
        loads_kw = full_load_profile[:21].tolist()
        pv_kw = {idx: full_solar_profile[idx] for idx in self.solar_indices}
        
        batt_home3 = self.node_battery_power_kw[3]
        batt_home5 = self.node_battery_power_kw[5]
        bess = self.node_battery_power_kw[21]
        
        # Run OpenDSS Step
        step_res = self.dss_runner.step(
            loads_kw=loads_kw,
            pv_kw=pv_kw,
            batt_home3_kw=batt_home3,
            batt_home5_kw=batt_home5,
            bess_kw=bess,
            auto_compile=False
        )
        
        # Voltages are now in fixed order: N0, N1, ..., N20, NBESS (indices 0-21)
        # Store min voltage for observations (RL agent sees this)
        self.voltages = np.array(step_res.vmin_pu_by_bus, dtype=np.float32)
        
        # For penalty calculation, check both min and max across 3 phases
        self.voltages_min = self.voltages.copy()
        self.voltages_max = np.zeros(self.n_nodes, dtype=np.float32)
        
        # Calculate max voltage per bus from 3-phase data
        for i, (va, vb, vc) in enumerate(step_res.vabc_pu_by_bus):
            phases = [v for v in [va, vb, vc] if not np.isnan(v)]
            self.voltages_max[i] = max(phases) if phases else 1.0
        
        # --- 4. REWARD CALCULATION ---
        # A. Economic Profit
        # --- Time-of-Use Pricing (Sri Lankan Rupees - LKR) ---
        
        if 6 <= hour < 18:         # Daytime / solar hours (6am-6pm)
            buy_price, sell_price = 35, 19  # LKR per kWh per 15-min
        elif 18 <= hour < 23:      # Peak  (6pm-11pm)
            buy_price, sell_price = 67, 45  # LKR per kWh per 15-min
        else:                      # Off-peak  (11pm-6am)
            buy_price, sell_price = 21, 0   # LKR per kWh per 15-min

        # Grid economics based on net injection (solar + battery - load)
        # Positive = export to grid (earn money), Negative = import from grid (pay money)
        grid_export = np.maximum(0, self.net_injection)   # Power sold to grid
        grid_import = np.maximum(0, -self.net_injection)  # Power bought from grid

        # Calculate overall grid costs and revenues
        grid_export_revenue = np.sum(grid_export * sell_price)
        grid_import_cost = np.sum(grid_import * buy_price)
        
        # --- BESS-Specific Economics ---
        bess_power = self.node_battery_power_kw[self.bess_index]  # Node 21 (end-of-feeder)
        
        # BESS Discharge Revenue (positive power = discharging)
        if bess_power > 0:
            # BESS is discharging - supplying power to load or grid
            # Calculate value based on current pricing
            bess_discharge_revenue = bess_power * sell_price  # LKR per 15-min
        else:
            bess_discharge_revenue = 0.0
        
        # BESS Charge Cost (negative power = charging)
        if bess_power < 0:
            # BESS is charging - consuming power from grid or solar
            # Check if charging from solar surplus or grid
            if net_solar_surplus > 0:
                # Charging from solar surplus - minimal cost (only opportunity cost)
                bess_charge_cost = 0.0  # Free solar energy
            else:
                # Charging from grid - pay the buy price
                bess_charge_cost = abs(bess_power) * buy_price  # LKR per 15-min
        else:
            bess_charge_cost = 0.0
        
        # Legacy variables for backward compatibility
        revenue = grid_export_revenue
        cost = grid_import_cost

        # ====== TIME-BASED CHARGING/DISCHARGING STRATEGY ======
        # Organized by the three pricing periods
        
        # ----- SECTION 1: DAYTIME / SOLAR HOURS (6am-6pm) -----
        # Strategy: Charge from excess solar ONLY, save for evening peak
        daytime_solar_bonus = 0.0
        solar_waste_penalty = 0.0
        if 6 <= hour < 18:  # Daytime hours
            total_charge_power = np.sum(np.minimum(0, self.node_battery_power_kw))  # Negative when charging
            total_discharge_power = np.sum(np.maximum(0, self.node_battery_power_kw))  # Positive when discharging
            
            if net_solar_surplus > 0:
                # Solar surplus available: reward for charging from excess solar
                
                daytime_solar_bonus += 30.0 * (-total_charge_power)
                
                # Extra incentive for BESS to absorb community solar (priority target)
                bess_charge_power = np.minimum(0, self.node_battery_power_kw[self.bess_index])
                surplus_factor = min(1.0, net_solar_surplus / 20.0)
                daytime_solar_bonus += 35.0 * (-bess_charge_power) * (1.0 + surplus_factor)
                
                # SOLAR WASTE PENALTY: Penalize exported solar when batteries have available capacity
                # Only penalize if batteries are not nearly full
                avg_soc = np.mean(self.soc)
                if avg_soc < 0.8:  # Batteries have room to charge
                    # Calculate solar wasted (surplus not captured in batteries)
                    solar_wasted = net_solar_surplus - (-total_charge_power)
                    if solar_wasted > 0.2:  # Only penalize meaningful waste
                        # Reduced from 50 to 25 LKR per kW wasted
                        solar_waste_penalty = -30.0 * solar_wasted
                        if self.verbose:
                            print(f"[SOLAR_WASTE] Hour {hour:.1f}: {solar_wasted:.2f}kW solar exported despite {(0.8-avg_soc):.2%} battery capacity available → Penalty -{30.0*solar_wasted:.0f}")
            else:
                # NO solar surplus - penalize BESS discharge during daytime to save for peak hours
                bess_discharge_power = np.maximum(0, self.node_battery_power_kw[self.bess_index])
                # Reduced from 50 to 25
                daytime_solar_bonus -= 25.0 * bess_discharge_power
            
            # NEW: BONUS for HOME BATTERIES absorbing solar locally (prevents end-of-feeder overvoltage)
            # Home batteries at Nodes 3 & 5 reduce solar flow to BESS node (Node 21)
            home_batt_solar_charge = np.sum([np.minimum(0, self.node_battery_power_kw[idx]) 
                                             for idx in self.home_batt_indices])  # Negative when charging
            if home_batt_solar_charge < 0 and net_solar_surplus > 0:
                # Reward home batteries for absorbing solar locally (reduced from 40 to 20)
                daytime_solar_bonus += 20.0 * (-home_batt_solar_charge)
                if self.verbose:
                    print(f"[LOCAL_SOLAR_ABSORPTION] Hour {hour:.1f}: Home batteries absorbing {-home_batt_solar_charge:.2f}kW locally → Reduces BESS voltage rise")
            
        
        # ----- SECTION 2: PEAK (6pm-11pm) -----
        # Strategy: Discharge at HIGH prices to maximize revenue at peak demand times
        # Peak discharge value = 45 LKR sell price (much higher than daytime 19 LKR)
        peak_bonus = 0.0
        peak_charge_penalty = 0.0  # NEW: Penalize charging during peak hours
        if 18 <= hour < 23:  # Evening peak hours (6pm-11pm)
            if np.mean(self.soc) > 0.3:  # Only discharge if battery has energy
                total_discharge_power = np.sum(np.maximum(0, self.node_battery_power_kw))
                
                peak_bonus = 35.0 * total_discharge_power
        
        # ----- SECTION 3: OFF-PEAK (11pm-6am) -----
        # Strategy: HOME batteries and BESS can charge at cheap rates (21 LKR)
        # BESS charges during off-peak to supplement INSUFFICIENT daytime solar only
        # NOTE: Charging blocked between 11pm-12am
        offpeak_bonus = 0.0
        if hour < 6:  # Off-peak hours (12am-6am only)
            # HOME BATTERY CHARGING
            home_batt_soc = [self.soc[0], self.soc[1]]  # Home batteries only
            if np.mean(home_batt_soc) < 0.8:  # Room to charge
                home_charge_power = self.node_battery_power_kw[3] + self.node_battery_power_kw[5]
                home_charge_power = min(0, home_charge_power)  # Negative when charging
                
                # Predictive charging: Check if tomorrow's solar will be sufficient
                solar_will_be_sufficient = False
                steps_ahead = min(96, self.max_steps - self.current_step)
                
                if steps_ahead > 24:  # Need enough data to predict
                    future_solar = self.solar_episode[self.current_step:self.current_step + steps_ahead]
                    future_load = self.load_episode[self.current_step:self.current_step + steps_ahead]
                    
                    # Calculate expected solar during next daylight (6am-6pm)
                    daylight_start = max(0, int((6 - hour) * 4))  # Steps until 6am
                    daylight_end = min(steps_ahead, daylight_start + 48)  # 12 hours of daylight
                    
                    if daylight_end > daylight_start:
                        expected_solar = np.sum(future_solar[daylight_start:daylight_end])
                        expected_load = np.sum(future_load[daylight_start:daylight_end])
                        expected_surplus = expected_solar - expected_load
                        
                        home_capacity = 2 * self.home_batt_cap
                        energy_needed = home_capacity * (0.8 - np.mean(home_batt_soc))
                        
                        # If solar can provide 90%+ of needed energy, don't use grid
                        if expected_surplus > energy_needed * 0.9:
                            solar_will_be_sufficient = True
                
                # Decision based on solar forecast (only for home batteries)
                if solar_will_be_sufficient:
                    # PENALTY: Don't charge from grid, save capacity for solar
                    offpeak_bonus = 4.0 * home_charge_power  
                else:
                    # REWARD: Charge at cheap off-peak rates (solar won't be enough)
                    offpeak_bonus = -8.0 * home_charge_power  
            
            # BESS OFF-PEAK CHARGING - Predictive Strategy
            # Reward BESS for intelligently pre-charging based on daytime solar forecast
            bess_soc = self.soc[2]  # BESS SoC
            bess_charge_power = self.node_battery_power_kw[self.bess_index]  # Node 21
            
            if bess_charge_power < 0:  # BESS is charging
                # Look ahead to assess if this off-peak charging is justified
                steps_remaining = self.max_steps - self.current_step
                steps_until_peak = min(steps_remaining, int((18 - hour) * 4))
                
                if steps_until_peak > 4:
                    future_solar = self.solar_episode[self.current_step:self.current_step + steps_until_peak]
                    future_load = self.load_episode[self.current_step:self.current_step + steps_until_peak]
                    expected_daytime_surplus = np.sum(future_solar) - np.sum(future_load)
                    
                    # Calculate if daytime solar will be sufficient
                    expected_surplus_energy = expected_daytime_surplus * 0.25 * 0.95
                    energy_needed = (0.75 - bess_soc) * self.bess_cap
                    
                    if expected_surplus_energy < energy_needed:
                        # Daytime solar insufficient - reward for smart off-peak charging (reduced from -18 to -10)
                        offpeak_bonus += -10.0 * bess_charge_power
                    else:
                        # Daytime solar will be sufficient - moderate reward (reduced from -8 to -5)
                        offpeak_bonus += -5.0 * bess_charge_power
                else:
                    # Not enough lookahead data - reward based on SoC state
                    if bess_soc < 0.4:
                        offpeak_bonus += -8.0 * bess_charge_power  # Reduced from -15
                    else:
                        offpeak_bonus += -5.0 * bess_charge_power  # Reduced from -10

        # NEW: Grid Import/Export Smoothing Penalty
        # Penalize rapid changes in grid flows to stabilize the grid
        # Calculate current grid net power (positive = export to grid, negative = import from grid)
        current_grid_net = np.sum(self.net_injection)  # Positive = injection, Negative = withdrawal
        
        # Calculate change from previous step (per 15-min interval)
        grid_power_change = abs(current_grid_net - self.prev_grid_net_power)
        
        # IMPROVED: Penalize large ramps GENTLY to prevent oscillations
        # Allow reasonable ramps (5 kW per 15 min) without penalty
        ramp_threshold = 5.0  # kW per 15-min step is reasonable (increased from 3.0)
        if grid_power_change > ramp_threshold:
            # Excess ramp beyond threshold gets LINEAR penalty (not quadratic!)
            excess_ramp = grid_power_change - ramp_threshold
            # Linear penalty for stability: small penalty for gradual learning
            grid_smoothing_penalty = -0.1 * excess_ramp  # Reduced from -0.5 * excess_ramp^2
            if self.verbose:
                print(f"[GRID_SMOOTHING] Hour {hour:.1f}: Grid power changed {grid_power_change:.2f}kW (was {self.prev_grid_net_power:.1f}, now {current_grid_net:.1f}) → Penalty {grid_smoothing_penalty:.1f}")
        else:
            grid_smoothing_penalty = 0.0
        
        # Store current grid net power for next step comparison
        self.prev_grid_net_power = current_grid_net

        # B. Voltage Violation Penalty (0.94 to 1.06 p.u. limits)
        # Monitor all nodes for grid safety compliance (3-phase aware)
        critical_nodes = list(range(21)) + [self.bess_index]

        # Check undervoltage: min voltage per bus should be >= 0.94
        min_voltages = self.voltages_min[critical_nodes]
        under_voltage = np.maximum(0, 0.94 - min_voltages)
        
        # Check overvoltage: max voltage per bus should be <= 1.06
        max_voltages = self.voltages_max[critical_nodes]
        over_voltage = np.maximum(0, max_voltages - 1.06)
        
        total_violation = np.sum(over_voltage + under_voltage)
        
        # IMPROVED: Gradient-based penalty that guides learning
        # Use softer penalties that give gradient signal to improve
        # Violation between 0 and 0.03 p.u: soft penalty (allows learning)
        # Violation > 0.03 p.u: harder penalty (prevents severe violations)
        soft_violations = np.minimum(0.03, over_voltage + under_voltage)
        hard_violations = np.maximum(0, (over_voltage + under_voltage) - 0.03)
        
        voltage_penalty = -50.0 * np.sum(soft_violations) - 500.0 * np.sum(hard_violations)
        
        # IMPROVED: VOLTAGE STABILITY BONUS - Reward nodes in safe bands
        # Generous bonus for nodes in ideal range to guide agent
        ideal_min = 0.98
        ideal_max = 1.02
        ideal_nodes = np.sum((min_voltages >= ideal_min) & (max_voltages <= ideal_max))
        
        acceptable_min = 0.94
        acceptable_max = 1.06
        acceptable_nodes = np.sum((min_voltages >= acceptable_min) & (max_voltages <= acceptable_max))
        
        # Bonus for ideal control + smaller bonus for acceptable control
        voltage_stability_bonus = 20.0 * ideal_nodes + 5.0 * (acceptable_nodes - ideal_nodes)

        # C. Battery Health & Smoothness
        # Penalize rapid power changes to reduce battery stress
        # Calculate actual power changes using stored previous values
        final_power_array = np.array([self.node_battery_power_kw[node_idx] 
                                      for node_idx in self.storage_map])
        power_changes = final_power_array - prev_batt_power_copy
        
        # Balanced cycling penalty - don't penalize smooth operation too much
        # Focus penalty on sudden jumps only
        cycling_cost = -0.3 * np.sum(np.abs(power_changes)) - 0.1 * np.sum(power_changes ** 2) 
        
        # D. SOC Health Penalty - Encourage keeping SOC in 0.2-0.8 range
        # This promotes battery longevity by avoiding deep discharge/overcharge
        soc_health_penalty = 0.0
        for i in range(len(self.soc)):
            if self.soc[i] < 0.2:
                # Penalty increases quadratically as SOC approaches 0
                soc_health_penalty -= 50.0 * (0.2 - self.soc[i]) ** 2
            elif self.soc[i] > 0.8:
                # Penalty increases quadratically as SOC approaches 1
                soc_health_penalty -= 50.0 * (self.soc[i] - 0.8) ** 2
        
        # E. Total Reward
        # Improved balance between economic and technical objectives
        reward = (revenue * 0.6                # Scale down economics (grid costs are high)
                  - cost * 0.6                 # Scale down costs proportionally 
                  + voltage_penalty 
                  + voltage_stability_bonus        # Reward tight voltage control
                  + daytime_solar_bonus           # Daytime solar charging (6am-6pm)
                  + solar_waste_penalty           # Penalize wasted solar exports
                  + peak_bonus                    # Peak discharge (6pm-11pm)
                  + offpeak_bonus                 # Off-peak charging (11pm-6am)
                  + grid_smoothing_penalty        # Penalize rapid grid power changes (reduced)
                  + cycling_cost
                  + soc_health_penalty)
        
        # Reward normalization and clipping for stability
        # Divide by higher value and clip to [-10, 10] range for stable learning
        reward = reward / 5.0  # Normalized divisor for better gradient scaling
        #reward = np.clip(reward, -20.0, 20.0)  # Allow wider range for better signal
        
        # Debug output for reward analysis (only when verbose)
        if self.verbose and self.current_step % 24 == 0:  # Every 6 hours
            print(f"[REWARD_DEBUG] Hour {hour:.1f}:")
            print(f"  Economic: revenue={revenue:.0f}, cost={cost:.0f}, net={revenue-cost:.0f}, scaled={(revenue-cost)*0.6:.0f}")
            print(f"  Voltages: min={np.min(min_voltages):.4f}, max={np.max(max_voltages):.4f}, penalty={voltage_penalty:.0f}, bonus={voltage_stability_bonus:.0f}")
            print(f"  Battery: cycling_cost={cycling_cost:.0f}, soc_health={soc_health_penalty:.0f}")
            print(f"  Bonuses: daytime={daytime_solar_bonus:.0f}, peak={peak_bonus:.0f}, offpeak={offpeak_bonus:.0f}")
            total_pre_norm = (revenue*0.6 - cost*0.6 + voltage_penalty + voltage_stability_bonus + daytime_solar_bonus + solar_waste_penalty + peak_bonus + offpeak_bonus + grid_smoothing_penalty + cycling_cost + soc_health_penalty)
            print(f"  Total reward (before norm): {total_pre_norm:.0f}")
            print(f"  Total reward (after norm): {reward:.2f}")

        # --- 4. NEXT STEP TRANSITION ---
        self.current_step += 1
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        obs = self._get_obs() if not terminated else self.state
        #obs = self._get_obs()
        
        # Pass info for debugging and monitoring
        info = {
            "hour": hour,
            "net_demand": net_demand,
            "remaining_demand": self.remaining_demand,
            "max_voltage": np.max(max_voltages),  # Max across all phases and buses
            "min_voltage": np.min(min_voltages),  # Min across all phases and buses
            "violation": total_violation,
            "solar_surplus": net_solar_surplus,
            "total_load": total_load,
            "total_solar": total_solar,
            "revenue": revenue,
            "cost": cost,
            "profit": revenue - cost,
            # Separate economic metrics
            "grid_export_revenue": grid_export_revenue,
            "grid_import_cost": grid_import_cost,
            "bess_discharge_revenue": bess_discharge_revenue,
            "bess_charge_cost": bess_charge_cost,
            # Battery states
            "soc_home3": self.soc[0],
            "soc_home5": self.soc[1],
            "soc_bess": self.soc[2],
            "bess_power": self.node_battery_power_kw[self.bess_index],
            "voltage_penalty": voltage_penalty,
            "grid_smoothing_penalty": grid_smoothing_penalty,
            "grid_power_change": grid_power_change,
            "current_grid_net": current_grid_net,
            "peak_import_target": peak_import_target
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        """Constructs the exact 51-value input vector for the RL agent.
        
        Observation Structure:
        - [0]:     Common solar forecast (kW)
        - [1-21]:  Load forecasts for Houses 0-20 (kW)
        - [22-43]: Voltages at ALL nodes [0,1,2,...,20,BESS] (p.u.)
        - [44-46]: Battery SoCs [Home3, Home5, BESS] (0-1)
        - [47-50]: Time features [sin(time), cos(time), sin(day), cos(day)]
        """
        
        # 1. Common Solar Forecast (1 Value)
        if self.current_step < self.max_steps:
            # Use raw weather data from Node 0 as the 'signal'
            common_solar = np.array([self.solar_episode[self.current_step][0]])
            load_step = self.load_episode[self.current_step]
        else:
            # Safety: Return zeros if episode has ended
            common_solar = np.array([0.0])
            load_step = np.zeros(21)
        
        # 2. Load Forecasts (21 Values)
        # Already extracted as load_step above
        
        # 3. All Node Voltages (22 Values) - CRITICAL INPUT
        # Complete grid visibility: Houses 0-20 + BESS node
        all_voltages = self.voltages

        # 4. Battery States of Charge (3 Values)
        # Already stored in self.soc
        
        # 5. Date & Time (4 Values)
        # Circular encoding for smooth periodic representation
        time_angle = (self.current_step / self.max_steps) * 2 * np.pi
        day_angle = ((self.start_idx // 96) / 365.0) * 2 * np.pi
        
        date_time_feats = np.array([
            np.sin(time_angle), np.cos(time_angle),
            np.sin(day_angle),  np.cos(day_angle)
        ])

        # 6. Pack State Vector (Total: 51 values)
        self.state = np.concatenate([
            common_solar,    # 1
            load_step,       # 21
            all_voltages,    # 22 (Complete grid visibility!)
            self.soc,        # 3
            date_time_feats  # 4
        ]).astype(np.float32)
        
        return self.state
    
    
