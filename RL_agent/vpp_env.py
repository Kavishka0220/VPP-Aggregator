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
        self.n_storage_units = 3
        
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
        self.soc = np.ones(3) * 0.5 
        self.prev_batt_power = np.zeros(3)
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
        
        # --- AUTOMATIC BESS CHARGING LOGIC ---
        action_modified = action.copy()
        bess_action_idx = 2  # BESS is the third storage unit
        bess_soc = self.soc[bess_action_idx]
        
        # Strategy 1: Charge from solar surplus when available
        if net_solar_surplus > 0:
            # Force BESS to charge from excess solar
            charge_intensity = min(1.0, net_solar_surplus / self.bess_power)
            action_modified[bess_action_idx] = -charge_intensity  # Negative = charging
        
        # Strategy 2: Predictive charging from grid during cheap night rates (12am-6am, 21 LKR)
        # Calculate if today's solar surplus will be sufficient to fully charge BESS
        # NOTE: Charging blocked between 11pm-12am
        elif hour < 6:
            # Look ahead to predict today's solar surplus
            steps_remaining = self.max_steps - self.current_step
            steps_until_evening = min(steps_remaining, int((18 - hour) * 4))  # Until 6pm
            
            if steps_until_evening > 4:  # Need at least 1 hour of data to predict
                # Get future solar and load data
                future_solar = self.solar_episode[self.current_step:self.current_step + steps_until_evening]
                future_load = self.load_episode[self.current_step:self.current_step + steps_until_evening]
                
                # Calculate total expected solar surplus throughout the day
                future_solar_total = np.sum(future_solar)
                future_load_total = np.sum(future_load)
                expected_daily_surplus = future_solar_total - future_load_total
                
                if expected_daily_surplus > 0:
                    # Convert surplus energy to how much it can charge BESS
                    # Each 15-min step with surplus can charge: surplus_power * 0.25 hours * efficiency
                    # Simplified: assume average surplus is spread across daytime hours
                    expected_surplus_energy = expected_daily_surplus * 0.25 * 0.95  # kWh with efficiency
                    
                    # Calculate BESS charging need (from current SoC 0.2 to target 0.8)
                    target_soc = 0.8
                    min_soc = 0.2  # Realistic minimum starting SoC
                    current_energy = bess_soc * self.bess_cap  # Current energy in kWh
                    target_energy = target_soc * self.bess_cap  # Target energy in kWh
                    energy_needed = max(0, target_energy - current_energy)  # How much energy needed
                    
                    # KEY LOGIC: Only charge at night if solar is INSUFFICIENT
                    # If solar surplus >= energy needed, DON'T charge at night (skip grid charging)
                    if expected_surplus_energy >= energy_needed:
                        # Solar is ENOUGH to charge battery - don't consume grid power at night
                        action_modified[bess_action_idx] = 0.0  # NO charging from grid
                    else:
                        # Solar is INSUFFICIENT - charge from grid at night to bridge deficit
                        energy_deficit = energy_needed - expected_surplus_energy
                        
                        # Calculate remaining night hours for charging (12am-6am)
                        if hour >= 23:
                            # From 11pm to midnight + midnight to 6am
                            steps_until_6am = int((24 - hour) * 4 + 6 * 4)
                        else:
                            # Already past midnight, just count to 6am
                            steps_until_6am = int((6 - hour) * 4)
                        
                        if steps_until_6am > 0:
                            # Spread deficit charging across remaining night hours
                            power_per_step = energy_deficit / (steps_until_6am * 0.25)
                            charge_intensity = min(1.0, power_per_step / self.bess_power)
                        else:
                            charge_intensity = 0.5
                        action_modified[bess_action_idx] = -charge_intensity
                else:
                    # No solar surplus expected - charge from grid at night
                    if bess_soc < 0.5:
                        action_modified[bess_action_idx] = -0.6  # Moderate charge
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
            # BESS prefers solar but can use cheap night grid power (12am-6am)
            # to ensure sufficient charge for evening peak
            # NOTE: Charging blocked between 11pm-12am (hour 23-24)
            if is_bess and desired_power < 0:  # BESS trying to charge
                # Allow charging if: (1) solar surplus available, OR (2) cheap night hours (12am-6am)
                if net_solar_surplus <= 0 and (6 <= hour or hour >= 23):
                    desired_power = 0.0  # Block grid charging outside night hours and during 11pm-12am
            
            # --- CONSTRAINT 3: Home Battery Daytime Solar Charging ---
            # Home batteries prefer solar during daytime but can use grid at night
            # During daytime hours, ONLY allow charging if there's excess solar
            if not is_bess and 6 <= hour < 18 and desired_power < 0:
                if net_solar_surplus <= 0:  # No solar surplus available
                    desired_power = 0.0  # Block daytime grid charging for home batteries
            
            # --- CONSTRAINT 4: Block all battery charging between 11pm-12am ---
            if 23 <= hour < 24 and desired_power < 0:
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
        bess_power = self.node_battery_power_kw[self.bess_index]  # Node 10
        
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
        if 6 <= hour < 18:  # Daytime hours
            total_charge_power = np.sum(np.minimum(0, self.node_battery_power_kw))  # Negative when charging
            total_discharge_power = np.sum(np.maximum(0, self.node_battery_power_kw))  # Positive when discharging
            
            if net_solar_surplus > 0:
                # Solar surplus available: STRONG reward for charging from excess solar
                daytime_solar_bonus += -10.0 * total_charge_power  # Strong solar charging
                
                # Extra BESS bonus for absorbing community solar
                bess_charge_power = np.minimum(0, self.node_battery_power_kw[self.bess_index])
                surplus_factor = min(1.0, net_solar_surplus / 20.0)
                daytime_solar_bonus += -10.0 * bess_charge_power * (1.0 + surplus_factor)
            
            
        
        # ----- SECTION 2: PEAK (6pm-11pm) -----
        # Strategy: Discharge at HIGH prices (67 LKR) to maximize revenue AND reduce evening peak demand
        peak_bonus = 0.0
        if 18 <= hour < 23:  # Evening peak hours
            if np.mean(self.soc) > 0.3:  # Only discharge if battery has energy
                total_discharge_power = np.sum(np.maximum(0, self.node_battery_power_kw))
                # STRONG incentive to discharge at peak prices (67 LKR)
                peak_bonus = 12.0 * total_discharge_power
        
        # ----- SECTION 3: OFF-PEAK (11pm-6am) -----
        # Strategy: HOME batteries and BESS can charge at cheap rates (21 LKR)
        # BESS charges at night to supplement insufficient solar generation
        # NOTE: Charging blocked between 11pm-12am
        offpeak_bonus = 0.0
        if hour < 6:  # Night hours (12am-6am only)
            # HOME BATTERY CHARGING
            home_batt_soc = [self.soc[0], self.soc[1]]  # Home batteries only
            if np.mean(home_batt_soc) < 0.7:  # Room to charge
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
                        energy_needed = home_capacity * (0.75 - np.mean(home_batt_soc))
                        
                        # If solar can provide 70%+ of needed energy, don't use grid
                        if expected_surplus > energy_needed * 0.7:
                            solar_will_be_sufficient = True
                
                # Decision based on solar forecast (only for home batteries)
                if solar_will_be_sufficient:
                    # PENALTY: Don't charge from grid, save capacity for solar
                    offpeak_bonus = 8.0 * home_charge_power
                else:
                    # REWARD: Charge at cheap off-peak rates (solar won't be enough)
                    offpeak_bonus = -15.0 * home_charge_power
            
            # BESS NIGHT CHARGING - Predictive Strategy
            # Reward BESS for intelligently pre-charging based on solar forecast
            bess_soc = self.soc[2]  # BESS SoC
            bess_charge_power = self.node_battery_power_kw[self.bess_index]  # Node 10
            
            if bess_charge_power < 0:  # BESS is charging
                # Look ahead to assess if this night charging is justified
                steps_remaining = self.max_steps - self.current_step
                steps_until_evening = min(steps_remaining, int((18 - hour) * 4))
                
                if steps_until_evening > 4:
                    future_solar = self.solar_episode[self.current_step:self.current_step + steps_until_evening]
                    future_load = self.load_episode[self.current_step:self.current_step + steps_until_evening]
                    expected_surplus = np.sum(future_solar) - np.sum(future_load)
                    
                    # Calculate if solar will be sufficient
                    expected_surplus_energy = expected_surplus * 0.25 * 0.95
                    energy_needed = (0.75 - bess_soc) * self.bess_cap
                    
                    if expected_surplus_energy < energy_needed:
                        # Solar insufficient - STRONG reward for smart off-peak charging
                        offpeak_bonus += -18.0 * bess_charge_power
                    else:
                        # Solar will be sufficient - moderate reward (still economical at 21 LKR)
                        offpeak_bonus += -8.0 * bess_charge_power
                else:
                    # Not enough lookahead data - reward based on SoC state
                    if bess_soc < 0.4:
                        offpeak_bonus += -15.0 * bess_charge_power
                    else:
                        offpeak_bonus += -10.0 * bess_charge_power

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
        
        # Heavy Penalty: -100 per unit of violation
        # Example: 0.01 p.u. deviation → -1 penalty
        # Example: 0.01 p.u. deviation → -1 penalty
        # voltage_penalty = -100.0 * total_violation
        voltage_penalty = -100.0 * (total_violation ** 2)

        # C. Battery Health & Smoothness
        # Penalize rapid power changes to reduce battery stress
        # Calculate actual power changes using stored previous values
        final_power_array = np.array([self.node_battery_power_kw[node_idx] 
                                      for node_idx in self.storage_map])
        power_changes = final_power_array - prev_batt_power_copy
        
        # INCREASED cycling penalty to reduce alternating behavior
        # Linear penalty for any changes + quadratic penalty for large changes
        cycling_cost = -2.0 * np.sum(np.abs(power_changes)) - 3.0 * np.sum(power_changes ** 2) 
        
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
        reward = (revenue 
                  - cost 
                  + voltage_penalty 
                  + daytime_solar_bonus      # Daytime solar charging (6am-6pm) + morning peak shaving
                  + peak_bonus               # Peak discharge (6pm-11pm)
                  + offpeak_bonus            # Off-peak charging (11pm-6am)
                  + cycling_cost
                  + soc_health_penalty)
        
        # Reward normalization - scale down to help with learning stability
        # Reduced normalization to preserve strong economic signals
        reward = reward / 5.0

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
            "voltage_penalty": voltage_penalty
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
    
    