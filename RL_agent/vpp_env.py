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
    - Constraints: Voltage must be between 0.9 and 1.1 p.u.
    - Inputs: Common Solar, 10 Loads, 6 PCC Voltages, 3 SoCs, Time.
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, data_path="./data", scenario_name=None, start_index=None):
        super(UrbanVPPEnv, self).__init__()
        
        # Testing configuration
        self.default_start_index = start_index

        # Initialize OpenDSS Runner
        dss_file = os.path.join(parent_dir, "openDSS", "feeder_houses.dss")
        self.dss_runner = VPPDSSRunner(dss_file)

        # --- 1. SYSTEM CONFIGURATION ---
        self.n_nodes = 11 # 0 to 9 (Houses) + 10 (BESS Node)
        self.solar_indices = [0, 1, 2, 4, 6, 8] # Which nodes have Solar? (0, 1, 2, 4, 6, 8)
        # Which nodes have Batteries?
        self.home_batt_indices = [0, 2] # Home Batteries at 0 & 2
        self.bess_index = 10 # BESS at 10
        
        # Map actions to physical nodes: Action[0]->Node0, Action[1]->Node2, Action[2]->Node6
        self.storage_map = self.home_batt_indices + [self.bess_index]
        self.n_storage_units = 3

        # Specs
        self.home_batt_cap = 13.5 # kWh
        self.bess_cap = 120.0 # kWh
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
        # 1(Solar) + 10(Loads) + 6(PCC Voltages) + 3(SoCs) + 4(Time) = 24
        self.obs_size = 24
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
        # We need 11 voltage values internally
        self.voltages = np.ones(self.n_nodes, dtype=np.float32) # Store all voltages internally

        # --- LOAD DATA ---
        try:
            if scenario_name:
                scenario_folder = os.path.join(data_path, "forecast_scenarios")
                print(f"[INFO] Loading Scenario: {scenario_name}")
                self.solar_df = pd.read_csv(f"{scenario_folder}/solar_{scenario_name}.csv")
                self.load_df = pd.read_csv(f"{scenario_folder}/load_{scenario_name}.csv")
            else:
                # These files contain 10 columns (House 0 to House 9)
                self.solar_df = pd.read_csv(f"{data_path}/solar_forecast_formatted.csv")
                self.load_df = pd.read_csv(f"{data_path}/load_forecast.csv")
            
            # Validate data shape
            if self.solar_df.shape[1] != 10 or self.load_df.shape[1] != 10:
                raise ValueError(f"Data must have 10 columns. Got solar: {self.solar_df.shape[1]}, load: {self.load_df.shape[1]}")
            
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
            
            print(f"[OK] Data Loaded Successfully! Final Length: {len(self.solar_df)}")
        except FileNotFoundError:
            # Fallback dummy data
            print("[WARNING] Using dummy random data")
            self.solar_df = pd.DataFrame(np.random.rand(1000, 10) * 5.0)
            self.load_df = pd.DataFrame(np.random.rand(1000, 10) * 3.0)    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        # Initialize SoC randomly within safe operating range (0.3 to 0.7) for robustness
        # This simulates realistic starting conditions and improves training generalization
        #self.soc = np.random.uniform(0.3, 0.7, size=3)
        self.soc = np.full(3, 0.2)
        self.prev_batt_power = np.zeros(self.n_storage_units)
        self.voltages = np.ones(self.n_nodes, dtype=np.float32) # Reset voltages to 1.0
        
        # Pick random day or use provided start index
        if options is not None and "start_step" in options:
            self.start_idx = options["start_step"]
            if "episode_len" in options:
                self.max_steps = options["episode_len"]
            print(f"[CONFIG] Starting at step {self.start_idx}, length {self.max_steps}")
        elif self.default_start_index is not None:
             self.start_idx = self.default_start_index
             # print(f"[CONFIG] Starting at FIXED default step {self.start_idx}")
        else:
            max_start = len(self.solar_df) - self.max_steps
            if max_start > 0:
                self.start_idx = np.random.randint(0, max_start)
            else:
                self.start_idx = 0  # Use all available data if it matches episode length

        # Slice Data
        self.solar_episode = self.solar_df.iloc[self.start_idx : self.start_idx + self.max_steps].values
        self.load_episode = self.load_df.iloc[self.start_idx : self.start_idx + self.max_steps].values
        
        # Apply Mask (Only keep solar for nodes with panels: 0,1,2,4,6,8)
        # Even if CSV has data for all houses, we zero out houses without panels
        self.solar_mask = np.zeros(10)
        self.solar_mask[self.solar_indices] = 1.0
        self.solar_episode = self.solar_episode * self.solar_mask

        return self._get_obs(), {}
    
    def step(self, action):
        # --- 1. GET DATA FIRST (Moved to top) ---
        # We must know the Total Load/Solar BEFORE deciding battery actions
        full_solar_profile = np.zeros(self.n_nodes) 
        full_load_profile = np.zeros(self.n_nodes) 

        # Fill with current step data
        full_solar_profile[:10] = self.solar_episode[self.current_step]
        full_load_profile[:10] = self.load_episode[self.current_step]

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
        
        # Strategy 2: Predictive charging from grid during cheap night rates (0-6am, 13 cents)
        # Calculate if today's solar surplus will be sufficient to fully charge BESS
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
                    
                    # Calculate BESS charging need (from current SoC to 80%)
                    target_soc = 0.8
                    current_energy = bess_soc * self.bess_cap  # Current energy in kWh
                    target_energy = target_soc * self.bess_cap  # Target energy in kWh
                    energy_needed = max(0, target_energy - current_energy)  # How much energy needed
                    
                    # Calculate deficit that solar can't provide
                    energy_deficit = max(0, energy_needed - expected_surplus_energy)
                    
                    if energy_deficit > 5.0:  # More than 5 kWh deficit
                        # Solar won't be enough - charge from grid at night
                        # Calculate remaining night hours for charging (0-6am = up to 24 steps)
                        steps_until_6am = int((6 - hour) * 4)
                        if steps_until_6am > 0:
                            # Spread deficit charging across remaining night hours
                            power_per_step = energy_deficit / (steps_until_6am * 0.25)
                            charge_intensity = min(1.0, power_per_step / self.bess_power)
                        else:
                            charge_intensity = 0.5
                        action_modified[bess_action_idx] = -charge_intensity
                    elif bess_soc < 0.3:
                        # Solar will be enough, but SoC is critically low - small safety charge
                        action_modified[bess_action_idx] = -0.3
                else:
                    # No solar surplus expected - charge aggressively at night
                    if bess_soc < 0.3:
                        action_modified[bess_action_idx] = -1.0  # Full charge
                    else:
                        action_modified[bess_action_idx] = -0.7  # High charge
        
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
            # BESS prefers solar but can use cheap night grid power (0-6am)
            # to ensure sufficient charge for evening peak
            if is_bess and desired_power < 0:  # BESS trying to charge
                # Allow charging if: (1) solar surplus available, OR (2) cheap night hours
                if net_solar_surplus <= 0 and hour >= 6:
                    desired_power = 0.0  # Block grid charging outside night hours
            
            # --- CONSTRAINT 3: Home Battery Daytime Solar Charging ---
            # Home batteries prefer solar during daytime but can use grid at night
            # During daytime hours, ONLY allow charging if there's excess solar
            if not is_bess and 6 <= hour < 18 and desired_power < 0:
                if net_solar_surplus <= 0:  # No solar surplus available
                    desired_power = 0.0  # Block daytime grid charging for home batteries
            
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
        loads_kw = full_load_profile[:10].tolist()
        pv_kw = {idx: full_solar_profile[idx] for idx in self.solar_indices}
        
        batt0 = self.node_battery_power_kw[0]
        batt2 = self.node_battery_power_kw[2]
        bess = self.node_battery_power_kw[10]
        
        # Run OpenDSS Step
        step_res = self.dss_runner.step(
            loads_kw=loads_kw,
            pv_kw=pv_kw,
            batt_home0_kw=batt0,
            batt_home2_kw=batt2,
            bess_kw=bess
        )
        
        # Map OpenDSS voltages to self.voltages (indices 0-10)
        name_to_idx = {f"N{i}": i for i in range(10)}
        name_to_idx["NBESS"] = 10
        
        new_voltages = np.ones(self.n_nodes, dtype=np.float32)
        for bus_name, v_pu in zip(step_res.buses, step_res.vmag_pu):
            bus_name_upper = bus_name.upper()
            if bus_name_upper in name_to_idx:
                idx = name_to_idx[bus_name_upper]
                new_voltages[idx] = v_pu
        
        self.voltages = new_voltages
        voltages_for_penalty = self.voltages.copy()
        
        # --- 4. REWARD CALCULATION ---
        # A. Economic Profit
        # --- Time-of-Use Pricing ---
        
        if 6 <= hour < 18:         # Daytime / solar hours (6am-6pm)
            buy_price, sell_price = 35, 19
        elif 18 <= hour < 23:      # Evening peak (6pm-11pm)
            buy_price, sell_price = 67, 45
        else:                      # Night (11pm-6am)
            buy_price, sell_price = 21, 0

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
            bess_discharge_revenue = bess_power * sell_price  # cents per 15-min
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
                bess_charge_cost = abs(bess_power) * buy_price  # cents per 15-min
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
                daytime_solar_bonus += -8.0 * total_charge_power  # Strong solar charging
                
                # Extra BESS bonus for absorbing community solar
                bess_charge_power = np.minimum(0, self.node_battery_power_kw[self.bess_index])
                surplus_factor = min(1.0, net_solar_surplus / 20.0)
                daytime_solar_bonus += -10.0 * bess_charge_power * (1.0 + surplus_factor)
            
            
        
        # ----- SECTION 2: EVENING PEAK (6pm-11pm) -----
        # Strategy: Discharge at high prices to maximize revenue
        evening_peak_bonus = 0.0
        if 18 <= hour < 23:  # Evening peak hours
            if np.mean(self.soc) > 0.3:  # Only discharge if battery has energy
                total_discharge_power = np.sum(np.maximum(0, self.node_battery_power_kw))
                # STRONG incentive to discharge at peak prices (54 cents)
                evening_peak_bonus = 10.0 * total_discharge_power
        
        # ----- SECTION 3: NIGHT (11pm-6am) -----
        # Strategy: HOME batteries and BESS can charge at cheap rates (13 cents)
        # BESS charges at night to supplement insufficient solar generation
        night_charge_bonus = 0.0
        if hour < 6 or hour >= 24:  # Night hours
            # HOME BATTERY CHARGING
            home_batt_soc = [self.soc[0], self.soc[1]]  # Home batteries only
            if np.mean(home_batt_soc) < 0.7:  # Room to charge
                home_charge_power = self.node_battery_power_kw[0] + self.node_battery_power_kw[2]
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
                    night_charge_bonus = 8.0 * home_charge_power
                else:
                    # REWARD: Charge at cheap night rates (solar won't be enough)
                    night_charge_bonus = -15.0 * home_charge_power
            
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
                        # Solar insufficient - STRONG reward for smart night charging
                        night_charge_bonus += -18.0 * bess_charge_power
                    else:
                        # Solar will be sufficient - moderate reward (still economical at 13c)
                        night_charge_bonus += -8.0 * bess_charge_power
                else:
                    # Not enough lookahead data - reward based on SoC state
                    if bess_soc < 0.4:
                        night_charge_bonus += -15.0 * bess_charge_power
                    else:
                        night_charge_bonus += -10.0 * bess_charge_power

        # B. Voltage Violation Penalty (0.9 to 1.1 p.u. limits)
        # Monitor all nodes for grid safety compliance
        critical_nodes = list(range(10)) + [self.bess_index]

        pcc_voltages = voltages_for_penalty[critical_nodes]

        # Calculate how far we are outside the safe zone
        # Logic: max(0, V - 1.1) + max(0, 0.9 - V)
        over_voltage = np.maximum(0, pcc_voltages - 1.1)
        under_voltage = np.maximum(0, 0.9 - pcc_voltages)
        
        total_violation = np.sum(over_voltage + under_voltage)
        
        # Heavy Penalty: -100 per unit of violation
        # Example: 0.01 p.u. deviation → -1 penalty
        # voltage_penalty = -100.0 * total_violation
        voltage_penalty = -50.0 * (total_violation ** 2)

        # C. Battery Health & Smoothness
        # Penalize rapid power changes to reduce battery stress
        # Calculate actual power changes using stored previous values
        final_power_array = np.array([self.node_battery_power_kw[node_idx] 
                                      for node_idx in self.storage_map])
        power_changes = final_power_array - prev_batt_power_copy
        cycling_cost = -0.5 * np.sum(np.abs(power_changes)) 
        
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
                  + daytime_solar_bonus      # Daytime solar charging (6am-6pm)
                  + evening_peak_bonus       # Evening peak discharge (6pm-11pm)
                  + night_charge_bonus       # Night cheap charging (11pm-6am)
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
            "max_voltage": np.max(pcc_voltages),
            "min_voltage": np.min(pcc_voltages),
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
            "soc_home0": self.soc[0],
            "soc_home2": self.soc[1],
            "soc_bess": self.soc[2],
            "bess_power": self.node_battery_power_kw[self.bess_index],
            "voltage_penalty": voltage_penalty
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        """Constructs the exact 24-value input vector for the RL agent.
        
        Observation Structure:
        - [0]:     Common solar forecast (kW)
        - [1-10]:  Load forecasts for Houses 0-9 (kW)
        - [11-16]: Voltages at solar nodes [0,1,2,4,6,8] (p.u.)
        - [17-19]: Battery SoCs [Home0, Home2, BESS] (0-1)
        - [20-23]: Time features [sin(time), cos(time), sin(day), cos(day)]
        """
        
        # 1. Common Solar Forecast (1 Value)
        if self.current_step < self.max_steps:
            # Use raw weather data from Node 0 as the 'signal'
            common_solar = np.array([self.solar_episode[self.current_step][0]])
            load_step = self.load_episode[self.current_step]
        else:
            # Safety: Return zeros if episode has ended
            common_solar = np.array([0.0])
            load_step = np.zeros(10)
        
        # 2. Load Forecasts (10 Values)
        # Already extracted as load_step above
        
        # 3. Solar PCC Voltages (6 Values) - CRITICAL INPUT
        # Only voltages at nodes with solar panels
        pcc_voltages = self.voltages[self.solar_indices]

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

        # 6. Pack State Vector (Total: 24 values)
        self.state = np.concatenate([
            common_solar,    # 1
            load_step,       # 10
            pcc_voltages,    # 6 (The Agent sees these!)
            self.soc,        # 3
            date_time_feats  # 4
        ]).astype(np.float32)
        
        return self.state
    
    