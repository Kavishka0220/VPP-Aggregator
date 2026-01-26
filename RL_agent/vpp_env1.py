import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class UrbanVPPEnv(gym.Env):
    """
    Final Thesis VPP Environment
    - Constraints: Voltage must be between 0.9 and 1.1 p.u.
    - Inputs: Common Solar, 10 Loads, 6 PCC Voltages, 3 SoCs, Time.
    """
    def __init__(self):
        super().__init__()

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
        self.bess_cap = 100.0 # kWh
        self.home_batt_power = 5.0 # kW
        self.bess_power = 50.0 # kW

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
            # These files contain 10 columns (House 0 to House 9)
            self.solar_df = pd.read_csv("../data/solar_forecast_formatted.csv")
            self.load_df = pd.read_csv("../data/load_forecast.csv")
            print("✅ Data Loaded Successfully!")
        except FileNotFoundError:
            # Fallback dummy data
            self.solar_df = pd.DataFrame(np.random.rand(1000, 10) * 5.0)
            self.load_df = pd.DataFrame(np.random.rand(1000, 10) * 3.0)    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.prev_batt_power = np.zeros(self.n_storage_units)

        self.current_step = 0
        self.soc = np.ones(3) * 0.5
        self.voltages = np.ones(self.n_nodes, dtype=np.float32) # Reset voltages to 1.0
        
        # Pick random day
        #max_start = len(self.solar_df) - self.max_steps
        #self.start_idx = np.random.randint(0, max_start)
        
        #-------
        # --- MODIFIED LOGIC START ---
        # check if a specific start index was passed in options
        if options is not None and "start_step" in options:
            self.start_idx = options["start_step"]
            # Optional: Allow changing duration (e.g., only optimize next 4 steps = 1 hour)
            if "episode_len" in options:
                self.max_steps = options["episode_len"]
            else:
                self.max_steps = 96 # Default to 24h
                
            print(f"🔧 Optimization starting at Step Index: {self.start_idx}")
            
        else:
            # Default Random Training Behavior
            self.max_steps = 96
            total_days = len(self.solar_df) // self.max_steps
            max_start = len(self.solar_df) - self.max_steps
            random_day = np.random.randint(0, total_days)
            self.start_idx = random_day * self.max_steps
        # --- MODIFIED LOGIC END ---

        # Safety Check: Ensure we don't run off the end of the data
        if self.start_idx + self.max_steps > len(self.solar_df):
            raise ValueError("Start index is too close to end of data!")
            self.max_steps = len(self.solar_df) - self.start_idx
        #-------

        # Slice Data
        self.solar_episode = self.solar_df.iloc[self.start_idx : self.start_idx + self.max_steps].values
        self.load_episode = self.load_df.iloc[self.start_idx : self.start_idx + self.max_steps].values
        
        # Apply Mask (Only keep solar for indices 0-4,6)
        # Even if CSV has data for House 6-9, we zero it out (No Panels)
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
        
        # Discharge Limit: Batteries fill the gap between Load and Solar.
        # If Solar > Load, limit is 0 (No discharge allowed).
        net_demand = max(0.0, total_load - total_solar)
        net_solar_surplus = total_solar - total_load # Positive if surplus
        home_solar_surplus = full_solar_profile - full_load_profile
        self.remaining_demand = net_demand # Decreases as we iterate batteries
        
        # --- 2. PHYSICS: APPLY ACTIONS ---
        # Create an array of size 11 for the grid physics
        self.node_battery_power_kw = np.zeros(self.n_nodes)
        hour = (self.current_step % 96) / 4  # 15-min steps → hours
        
        for i, node_idx in enumerate(self.storage_map):
            
            is_bess = (node_idx == self.bess_index)

            p_max = self.bess_power if is_bess else self.home_batt_power
            cap = self.bess_cap if is_bess else self.home_batt_cap
            ramp_limit = self.bess_batt_ramp if is_bess else self.home_batt_ramp
            desired_power = action[i] * p_max # Convert normalized action [-1,1] → real power (kW)

            # ----- CHARGING RULES (FINAL) -----
            if desired_power < 0:
                if is_bess:
                    if net_solar_surplus <= 0:
                        desired_power = 0.0
                    else:
                        desired_power = max(desired_power, -net_solar_surplus)
                else:
                    if home_solar_surplus[node_idx] <= 0:
                        desired_power = 0.0

             # -----CONSTRAINT: SOC SAFETY -----
            if self.soc[i] <= 0.2 and desired_power > 0:desired_power = 0.0
            if self.soc[i] >= 0.8 and desired_power < 0:desired_power = 0.0

            # ----- RAMP RATE -----
            delta_p = np.clip(desired_power - self.prev_batt_power[i], -ramp_limit, ramp_limit)
            final_power = self.prev_batt_power[i] + delta_p
            self.prev_batt_power[i] = final_power

            # ----- Update SoC -----
            eff = 0.95
            if final_power >= 0: # Discharging (efficiency loss)
                self.soc[i] -= (final_power * 0.25) / cap
            else:
                self.soc[i] -= (final_power * 0.25) / (cap * eff)
            self.soc[i] = np.clip(self.soc[i], 0, 1)
            self.node_battery_power_kw[node_idx] = final_power

            '''# --- CONSTRAINT 1: No Charging at Night (6pm - 6am) ---
            # (FIXED): Allow BESS night charging ---
            if hour < 6 or hour > 18:
                if desired_power < 0 and node_idx != self.bess_index:
                    desired_power = 0.0
            
            # --- CONSTRAINT 2: Demand-following discharge ---
            if desired_power > 0:
                desired_power = min(desired_power, self.remaining_demand)
                self.remaining_demand -= desired_power
            
            if desired_power < 0 and node_idx == self.bess_index:
                if net_solar_surplus <= 0:
                    desired_power = 0.0

            # --- CONSTRAINT 4: Ramp Rate Limits ---
            # Identify if BESS or Home battery
            
            if desired_power < 0:  # charging
                if Home_solar_surplus[node_idx] <= 0:
                    desired_power = 0.0
            
            self.prev_batt_power[i] = final_power'''


        # --- 3. PHYSICS: CALCULATE VOLTAGES ---
        
        # Record power at the correct grid node (0, 2, or 10)
        self.net_injection = full_solar_profile + self.node_battery_power_kw - full_load_profile
        
        # Simple Impedance Model (V = V_grid + I*R)
        # 1.0 is slack bus voltage
        # alphas increases with distance (Node 10 is furthest)
        alphas = np.linspace(0.005, 0.025, self.n_nodes)
        
        # Calculate voltage for ALL 11 nodes
        raw_voltages = 1.0 + alphas * self.net_injection
        # Save unclipped voltages for penalty calculation
        voltages_for_penalty = raw_voltages.copy()
        #self.voltages = np.clip(raw_voltages, 0.9, 1.1)
        self.voltages = raw_voltages
        
        # --- 4. REWARD CALCULATION ---
        
        # A. Voltage Violation Penalty (0.9 to 1.1)
        #pcc_voltages_solar = self.voltages[self.solar_indices]
        critical_nodes = list(range(10)) + [self.bess_index]

        pcc_voltages = voltages_for_penalty[critical_nodes]

        # Calculate how far we are outside the safe zone
        # Logic: max(0, V - 1.1) + max(0, 0.9 - V)
        over_voltage = np.maximum(0, pcc_voltages - 1.1)
        under_voltage = np.maximum(0, 0.9 - pcc_voltages)
        
        total_violation = np.sum(over_voltage + under_voltage)
        
        # Heavy Penalty:Even a 0.01 deviation gets a -10 penalty
        voltage_penalty = -1000.0 * total_violation
        
        # -------------------------
        '''# B. Economic Profit
        # --- Time-of-Use Pricing ---
        
        if 8 <= hour <= 18:        # Daytime / solar hours
            buy_price, sell_price = 0.08, 0.10
        else:                      # Evening & night
            buy_price, sell_price = 0.25, 0.30

        # Export earns money, import costs money
        export_power = np.maximum(0, self.node_battery_power_kw)
        import_power = np.maximum(0, -self.node_battery_power_kw)

        revenue = np.sum(export_power * sell_price)
        cost = np.sum(import_power * buy_price)

        solar_charge_bonus = 0.0
        if net_solar_surplus > 0:
            # Reward charging when solar is abundant
            solar_charge_bonus = 3.0 * np.sum(np.minimum(0, self.node_battery_power_kw))

        night_discharge_bonus = 0.0
        if (hour < 6 or hour > 18) and np.mean(self.soc) > 0.3:
            night_discharge_bonus = 0.5 * np.sum(np.maximum(0, self.node_battery_power_kw))
        '''
        solar_charge_reward = 0.5 * np.sum(np.maximum(0, -self.node_battery_power_kw))
        revenue = solar_charge_reward
        # -----------------------------------------

        ramp_penalty = -0.05 * np.sum(np.abs(self.prev_batt_power))

        # C. Battery Health (Optional: small penalty for cycling)
        cycling_cost = -0.1 * np.sum(np.abs(action)) 
        
        # D. Total Reward
        reward = (# cost + solar_charge_bonus  + night_discharge_bonus 
                  +revenue
                  + voltage_penalty 
                  + cycling_cost
                  + ramp_penalty)

        reward = np.clip(reward, -200, 5)

        # --- 4. NEXT STEP TRANSITION ---
        self.current_step += 1
        terminated = (self.current_step >= self.max_steps)
        truncated = False
        
        #obs = self._get_obs() if not terminated else self.state
        #obs = self._get_obs() if not terminated else None
        obs = self._get_obs()
        
        # Pass info for debugging
        info = {
            "net_solar_surplus": net_solar_surplus,
            "home_solar_surplus": home_solar_surplus.copy(),
        }
        # Save battery dispatch for plotting and evaluation
        self.node_battery_power_kw = self.node_battery_power_kw.copy()

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        #Constructs the exact 24-value input vector
        idx = min(self.current_step, self.max_steps - 1)
        # 1. Common Solar Forecast (1 Value)
        common_solar = np.array([self.solar_episode[idx][0]])
        load_step = self.load_episode[idx]  # 10 Values

        # 3. Solar PCC Voltages (6 Values) - CRITICAL INPUT
        pcc_voltages = self.voltages[self.solar_indices]

        # 4. Date & Time (4 Values)
        time_angle = (idx / self.max_steps) * 2 * np.pi
        day_angle = ((self.start_idx // 96) / 365.0) * 2 * np.pi
        
        date_time_feats = np.array([
            np.sin(time_angle), np.cos(time_angle),
            np.sin(day_angle),  np.cos(day_angle)
        ])

        # 5. Pack State
        self.state = np.concatenate([
            common_solar,    # 1
            load_step,       # 10
            pcc_voltages,    # 6 (The Agent sees these!)
            self.soc,        # 3
            date_time_feats  # 4
        ]).astype(np.float32)
        
        return self.state
    
    