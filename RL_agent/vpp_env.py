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
    
    def __init__(self, data_path="./data", scenario_name=None):
        super(UrbanVPPEnv, self).__init__()

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
        
        # Discharge Limit: Batteries fill the gap between Load and Solar.
        # If Solar > Load, limit is 0 (No discharge allowed).
        net_demand = max(0.0, total_load - total_solar)
        net_solar_surplus = total_solar - total_load # Positive if surplus
        home_solar_surplus = full_solar_profile - full_load_profile  # Per-house surplus
        self.remaining_demand = net_demand # Decreases as we iterate batteries
        
        # --- 2. PHYSICS: APPLY ACTIONS ---
        # Create an array of size 11 for the grid physics
        self.node_battery_power_kw = np.zeros(self.n_nodes)
        hour = (self.current_step % 96) / 4  # 15-min steps → hours
        
        # Store previous battery power for cycling cost calculation
        prev_batt_power_copy = self.prev_batt_power.copy()
        
        for i, node_idx in enumerate(self.storage_map):
            
            is_bess = (node_idx == self.bess_index)

            p_max = self.bess_power if is_bess else self.home_batt_power
            cap = self.bess_cap if is_bess else self.home_batt_cap
            ramp = self.bess_batt_ramp if is_bess else self.home_batt_ramp
            desired_power = action[i] * p_max # Convert normalized action [-1,1] → real power (kW)
                   
            # --- CONSTRAINT 1: SoC Limits (0.2 - 0.8 safe zone)
            # Check this early to avoid other constraints with depleted batteries
            if self.soc[i] <= 0.2 and desired_power > 0:
                desired_power = 0.0  # Prevent discharge when battery low
            if self.soc[i] >= 0.8 and desired_power < 0:
                desired_power = 0.0  # Prevent charging when battery full

            # --- CONSTRAINT 3: Prefer solar charging for home batteries (soft preference) ---
            # Home batteries get a small penalty if charging without local surplus
            # but are still allowed to charge from grid during cheap hours
            # This is handled in the reward function, not as a hard constraint
            
            # --- CONSTRAINT 4: REMOVED - Allow flexible charging/discharging ---
            # Batteries can charge or discharge based on economic signals
            # Agent will learn optimal strategies through reward function
            
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
            buy_price, sell_price = 25, 19
        elif 18 <= hour < 23:      # Evening peak (6pm-11pm)
            buy_price, sell_price = 54, 45
        else:                      # Night (11pm-6am)
            buy_price, sell_price = 13, 0

        # Grid economics based on net injection (solar + battery - load)
        # Positive = export to grid (earn money), Negative = import from grid (pay money)
        grid_export = np.maximum(0, self.net_injection)   # Power sold to grid
        grid_import = np.maximum(0, -self.net_injection)  # Power bought from grid

        revenue = np.sum(grid_export * sell_price)
        cost = np.sum(grid_import * buy_price)

        # Bonus: Encourage charging when solar surplus exists
        # Charging power is negative, so negate to get positive reward
        solar_charge_bonus = 0.0
        if net_solar_surplus > 0:
            total_charge_power = np.sum(np.minimum(0, self.node_battery_power_kw))  # Negative value
            solar_charge_bonus = -3.0 * total_charge_power  # Convert to positive reward

        # Bonus: BESS Excess Solar Charging - Strongly encourage BESS to absorb community solar surplus
        # BESS acts as the main buffer for community-wide excess solar generation
        bess_solar_charge_bonus = 0.0
        if net_solar_surplus > 0:
            # Get BESS charging power (index 2 in storage_map = BESS at node 10)
            bess_charge_power = np.minimum(0, self.node_battery_power_kw[self.bess_index])  # Negative if charging
            # Strong incentive: BESS should prioritize absorbing excess solar
            # Scale bonus with amount of solar surplus to encourage maximum absorption
            surplus_factor = min(1.0, net_solar_surplus / 20.0)  # Normalize by typical surplus
            bess_solar_charge_bonus = -10.0 * bess_charge_power * (1.0 + surplus_factor)  # Strong positive reward

        # Bonus: Smart daytime charging (6am-6pm) - prioritize solar, allow grid when beneficial
        # Encourage charging during moderate price hours when batteries need it
        daytime_charge_bonus = 0.0
        if 6 <= hour < 18 and np.mean(self.soc) < 0.7:  # Daytime with room to charge
            total_charge_power = np.sum(np.minimum(0, self.node_battery_power_kw))  # Negative value
            # Strong bonus with solar surplus, moderate bonus otherwise (still cheaper than evening)
            if net_solar_surplus > 0:
                daytime_charge_bonus = -5.0 * total_charge_power  # Strong incentive with solar
            else:
                daytime_charge_bonus = -1.0 * total_charge_power  # Mild incentive from grid

        # Bonus: Cheap night charging (11pm-6am) - arbitrage opportunity
        # Charge at low prices (buy=13) to discharge later at high prices
        night_charge_bonus = 0.0
        if (hour < 6 or hour >= 23) and np.mean(self.soc) < 0.7:  # Cheapest hours with room to charge
            total_charge_power = np.sum(np.minimum(0, self.node_battery_power_kw))  # Negative value
            # Good incentive for grid arbitrage - buy cheap, sell expensive later
            night_charge_bonus = -3.0 * total_charge_power
        
        # Bonus: Evening discharge (6pm-11pm) - capitalize on high prices
        # Encourage using stored energy during expensive hours
        evening_discharge_bonus = 0.0
        if 18 <= hour < 23 and np.mean(self.soc) > 0.3:  # Peak hours with energy available
            evening_discharge_bonus = 2.0 * np.sum(np.maximum(0, self.node_battery_power_kw))

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
        voltage_penalty = -100.0 * total_violation

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
                  + solar_charge_bonus
                  + bess_solar_charge_bonus
                  + daytime_charge_bonus
                  + night_charge_bonus
                  + evening_discharge_bonus
                  + cycling_cost
                  + soc_health_penalty)
        
        # Reward normalization - scale down to help with learning stability
        # Typical rewards are in range [-100, 100], normalize to reasonable range
        reward = reward / 10.0

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
    
    