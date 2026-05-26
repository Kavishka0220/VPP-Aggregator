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

    IMPROVEMENTS over original:
    1. Softened hard action overrides → RL agent has more control / higher ceiling
    2. Reward normalisation fixed  → divide by 200 (matches actual ~600 peak scale)
    3. Reward clipping re-enabled  → [-10, 10] for stable PPO gradients
    4. Peak-discharge clean bonus  → extra signal when BESS serves peak without violations
    5. Minor off-peak penalty fix  → stop penalising home-battery charging when solar forecasted

    VOLTAGE VIOLATION FIXES (addresses 0.90 p.u. undervoltage during off-peak charging):
    V1. Voltage-aware charge limiting  → scale back BESS charging when node voltage already low
    V2. Time-aware ramp rates          → tighter ramp during off-peak charging (5 kW/step)
                                         vs peak discharge (10 kW/step)
    V3. Severe violation penalty tier  → −500 per p.u. beyond 0.06 violation (quadratic signal)

    CODE QUALITY IMPROVEMENTS:
    - Extracted reward constants to class-level attributes for easy tuning
    - Extracted data loading to _load_data()
    - Broke step() into sub-methods for readability
    - Fixed peak charge penalty indexing bug (was reading wrong nodes)
    - Added OpenDSS convergence check
    - Normalised SoC in observation space to [-1, +1]
    - Added terminal SoC reward for end-of-day battery health
    """

    metadata = {'render_modes': []}

    # ==================== REWARD TUNING CONSTANTS ====================
    # All reward weights in one place for easy experimentation.
    # Naming convention: R_ prefix for reward constants.

    # Economic
    R_ECONOMIC_WEIGHT        = 0.6

    # Daytime solar
    R_SOLAR_CHARGE_BONUS     = 30.0   # per kW of solar charging
    R_BESS_SOLAR_CHARGE      = 35.0   # per kW of BESS solar charging
    R_SOLAR_WASTE_PENALTY    = -30.0  # per kW of wasted solar
    R_BESS_DAYTIME_DISCHARGE = -25.0  # per kW of inappropriate daytime discharge
    R_HOME_SOLAR_CHARGE      = 20.0   # per kW of home battery solar charging

    # Peak hours
    R_PEAK_DISCHARGE_BONUS   = 35.0   # per kW of peak discharge
    R_CLEAN_PEAK_BONUS       = 50.0   # per kW of clean (no-violation) peak discharge
    R_PEAK_CHARGE_PENALTY    = -50.0  # per kW of charging during peak

    # Off-peak
    R_HOME_OFFPEAK_SOLAR_OK  = 4.0    # home charging penalty when solar will suffice
    R_HOME_OFFPEAK_NEEDED    = -8.0   # home charging bonus when solar won't suffice

    # Voltage
    R_VOLT_SOFT_PER_PU       = -25.0   # 0 to 0.03 p.u. violation
    R_VOLT_HARD_PER_PU       = -100.0  # 0.03 to 0.06 p.u. violation
    R_VOLT_SEVERE_PER_PU     = -500.0  # beyond 0.06 p.u. violation
    R_IDEAL_NODE_BONUS       = 20.0    # per node in [0.98, 1.02]
    R_ACCEPTABLE_NODE_BONUS  = 5.0     # per node in [0.94, 1.06] but not ideal

    # Battery health
    R_CYCLING_LINEAR         = -0.3
    R_CYCLING_QUADRATIC      = -0.1
    R_SOC_HEALTH_PENALTY     = -50.0

    # Grid smoothing
    R_RAMP_THRESHOLD         = 5.0    # kW change before penalty kicks in
    R_RAMP_PENALTY_RATE      = -0.1   # per kW above threshold

    # Terminal
    R_TERMINAL_SOC_PENALTY   = -2.0   # per unit SoC deviation from 0.5

    # Normalisation
    R_NORMALIZER             = 200.0
    R_CLIP_LOW               = -10.0
    R_CLIP_HIGH              = 10.0

    # Convergence failure
    R_CONVERGENCE_PENALTY    = -5.0   # heavy normalised penalty for non-convergence

    # =================================================================

    def __init__(self, data_path="./data", scenario_name=None, start_index=None, verbose=False):
        super(UrbanVPPEnv, self).__init__()

        # Testing configuration
        self.verbose = verbose
        self.default_start_index = start_index
        self.scenario_name = scenario_name

        # Initialize OpenDSS Runner
        dss_file = os.path.join(parent_dir, "openDSS", "feeder_houses.dss")
        self.dss_runner = VPPDSSRunner(dss_file)
        self.dss_runner.compile()

        # --- 1. SYSTEM CONFIGURATION ---
        self.n_nodes = 22
        self.solar_indices = [3, 5, 7, 10, 11, 13, 15, 17, 18, 19, 20]

        if not all(0 <= idx < 21 for idx in self.solar_indices):
            raise ValueError(f"Solar indices must be in range [0, 20]. Got: {self.solar_indices}")

        self.home_batt_indices = [3, 5]
        self.bess_index = 21
        self.storage_map = self.home_batt_indices + [self.bess_index]
        self.n_storage_units = len(self.storage_map)

        # Critical nodes for voltage checking (all houses + BESS)
        self.critical_nodes = list(range(21)) + [self.bess_index]

        # Specs
        self.home_batt_cap  = 13.5   # kWh
        self.bess_cap       = 200.0  # kWh
        self.home_batt_power = 5.0   # kW
        self.bess_power      = 40.0  # kW

        # Ramp rate limits (kW per 15-min step)
        # FIX V2: Split BESS ramp into two modes:
        #   - Off-peak charging (midnight–6am): 5 kW/step → smaller per-step
        #     current draw → less voltage sag at end-of-feeder Node 21
        #   - Peak discharging (6pm–11pm): 10 kW/step → fast response kept
        self.home_batt_ramp         = 2.0    # kW/step (unchanged)
        self.bess_ramp_peak         = 10.0   # kW/step during peak discharge
        self.bess_ramp_offpeak      = 5.0    # kW/step during off-peak charging

        # --- 2. ACTION SPACE ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # --- 3. OBSERVATION SPACE ---
        # 1(Solar) + 21(Loads) + 22(Voltages) + 3(SoCs) + 4(Time) = 51
        # All values are normalised in _get_obs() so they fit inside [-1, 1].
        self.obs_size = 51

        # Normalisation constants (per observation group)
        self.OBS_SOLAR_MAX  = 10.0   # kW  – max per-node solar generation
        self.OBS_LOAD_MAX   = 10.0   # kW  – max per-node load
        self.OBS_VOLT_NOM   = 1.0    # p.u. nominal  (centred, then divided by range)
        self.OBS_VOLT_RANGE = 0.15   # p.u. half-range (0.85-1.15 maps to -1..+1)
        self.OBS_SOC_CENTER = 0.5    # SoC midpoint
        self.OBS_SOC_RANGE  = 0.3    # SoC half-range (0.2–0.8 maps to -1..+1)

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.obs_size,), dtype=np.float32
        )

        # State Variables
        self.state = None
        self.current_step = 0
        self.max_steps = 96
        self.soc = np.ones(self.n_storage_units, dtype=np.float32) * 0.5
        self.prev_batt_power = np.zeros(self.n_storage_units, dtype=np.float32)
        self.prev_grid_net_power = 0.0
        self.voltages     = np.ones(self.n_nodes, dtype=np.float32)
        self.voltages_min = np.ones(self.n_nodes, dtype=np.float32)
        self.voltages_max = np.ones(self.n_nodes, dtype=np.float32)

        # --- LOAD DATA ---
        self._load_data(data_path, scenario_name)

    # ------------------------------------------------------------------
    def _load_data(self, data_path, scenario_name):
        """Load and validate solar/load CSV data."""
        try:
            if scenario_name:
                scenario_folder = os.path.join(data_path, "forecast_scenarios")
                solar_file = os.path.join(scenario_folder, f"solar_{scenario_name}.csv")
                load_file  = os.path.join(scenario_folder, f"load_{scenario_name}.csv")

                if not os.path.exists(solar_file):
                    raise FileNotFoundError(f"Scenario file not found: {solar_file}")
                if not os.path.exists(load_file):
                    raise FileNotFoundError(f"Scenario file not found: {load_file}")

                print(f"[INFO] Loading Scenario: {scenario_name}")
                self.solar_df = pd.read_csv(solar_file)
                self.load_df  = pd.read_csv(load_file)
            else:
                solar_file = os.path.join(data_path, "solar_forecast_formatted.csv")
                load_file  = os.path.join(data_path, "load_forecast.csv")
                print(f"[INFO] Loading default data files")
                self.solar_df = pd.read_csv(solar_file)
                self.load_df  = pd.read_csv(load_file)

            # ---- Clean solar DF ----
            ts_cols = [c for c in self.solar_df.columns if c.lower() in ['timestamp','time','date','datetime']]
            if ts_cols:
                self.solar_df = self.solar_df.drop(columns=ts_cols)
            for col in self.solar_df.columns:
                self.solar_df[col] = pd.to_numeric(self.solar_df[col], errors='coerce')
            self.solar_df = self.solar_df.dropna(axis=1, how='all')
            if self.solar_df.shape[1] > 21:
                self.solar_df = self.solar_df.iloc[:, :21]
            elif self.solar_df.shape[1] < 21:
                raise ValueError(f"solar has only {self.solar_df.shape[1]} columns, need 21")

            # ---- Clean load DF ----
            ts_cols = [c for c in self.load_df.columns if c.lower() in ['timestamp','time','date','datetime']]
            if ts_cols:
                self.load_df = self.load_df.drop(columns=ts_cols)
            for col in self.load_df.columns:
                self.load_df[col] = pd.to_numeric(self.load_df[col], errors='coerce')
            self.load_df = self.load_df.dropna(axis=1, how='all')
            if self.load_df.shape[1] > 21:
                self.load_df = self.load_df.iloc[:, :21]
            elif self.load_df.shape[1] < 21:
                raise ValueError(f"load has only {self.load_df.shape[1]} columns, need 21")

            # ---- Handle length mismatch ----
            len_solar = len(self.solar_df)
            len_load  = len(self.load_df)
            if len_solar != len_load:
                print(f"[WARNING] Data length mismatch. Solar: {len_solar}, Load: {len_load}")
                if len_load == 96 and len_solar > 96:
                    days = int(np.ceil(len_solar / 96))
                    self.load_df = pd.concat([self.load_df] * days, ignore_index=True).iloc[:len_solar]
                elif len_solar == 96 and len_load > 96:
                    days = int(np.ceil(len_load / 96))
                    self.solar_df = pd.concat([self.solar_df] * days, ignore_index=True).iloc[:len_load]
                len_solar = len(self.solar_df)
                len_load  = len(self.load_df)
                if len_solar != len_load:
                    min_len = min(len_solar, len_load)
                    self.solar_df = self.solar_df.iloc[:min_len]
                    self.load_df  = self.load_df.iloc[:min_len]

            if len(self.solar_df) < self.max_steps:
                raise ValueError(f"Data must have at least {self.max_steps} rows.")

            print(f"[OK] Data Loaded Successfully! Scenario: {scenario_name or 'default'}")

        except FileNotFoundError as e:
            print(f"[WARNING] File loading error: {e}")
            print("[WARNING] Using dummy random data.")
            self.solar_df = pd.DataFrame(np.random.rand(1000, 21).astype(np.float32) * 5.0)
            self.load_df  = pd.DataFrame(np.random.rand(1000, 21).astype(np.float32) * 3.0)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.soc = np.full(3, 0.2, dtype=np.float32)
        self.prev_batt_power    = np.zeros(self.n_storage_units, dtype=np.float32)
        self.prev_grid_net_power = 0.0
        self.voltages     = np.ones(self.n_nodes, dtype=np.float32)
        self.voltages_min = np.ones(self.n_nodes, dtype=np.float32)
        self.voltages_max = np.ones(self.n_nodes, dtype=np.float32)

        if options is not None and "start_step" in options:
            self.start_idx = options["start_step"]
            if "episode_len" in options:
                self.max_steps = options["episode_len"]
        elif self.default_start_index is not None:
            self.start_idx = self.default_start_index
        else:
            max_start = len(self.solar_df) - self.max_steps
            self.start_idx = np.random.randint(0, max_start) if max_start > 0 else 0

        self.solar_episode = self.solar_df.iloc[self.start_idx:self.start_idx + self.max_steps].values.astype(np.float32)
        self.load_episode  = self.load_df.iloc[self.start_idx:self.start_idx + self.max_steps].values.astype(np.float32)

        self.solar_mask = np.zeros(21, dtype=np.float32)
        self.solar_mask[self.solar_indices] = 1.0
        self.solar_episode = self.solar_episode * self.solar_mask

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _max_charge_power_from_headroom(self, soc, cap, p_max):
        if soc >= 0.8:
            return 0.0
        headroom_kwh = (0.8 - soc) * cap
        return min(p_max, headroom_kwh / (0.25 * 0.95))

    def _max_discharge_power_from_soc(self, soc, cap, p_max):
        if soc <= 0.2:
            return 0.0
        deliverable_kwh = (soc - 0.2) * cap * 0.95
        return min(p_max, deliverable_kwh / 0.25)

    def _compute_peak_import_target(self, hour):
        if not (18 <= hour < 23):
            return None
        peak_end_step = min(self.max_steps, self.current_step + int((23 - hour) * 4))
        if peak_end_step <= self.current_step:
            return 0.0

        future_load  = np.sum(self.load_episode[self.current_step:peak_end_step], axis=1)
        future_solar = np.sum(self.solar_episode[self.current_step:peak_end_step], axis=1)
        future_net_demand = np.maximum(0.0, future_load - future_solar)

        if future_net_demand.size == 0:
            return 0.0

        deliverable_energy_kwh = max(0.0, (self.soc[2] - 0.2) * self.bess_cap * 0.95)
        if deliverable_energy_kwh <= 0.0:
            return float(np.max(future_net_demand))

        low  = 0.0
        high = float(np.max(future_net_demand))
        for _ in range(24):
            mid = 0.5 * (low + high)
            shaved = np.minimum(np.maximum(future_net_demand - mid, 0.0), self.bess_power)
            if np.sum(shaved) * 0.25 > deliverable_energy_kwh:
                low = mid
            else:
                high = mid
        return high

    # ------------------------------------------------------------------
    # STEP SUB-METHODS
    # ------------------------------------------------------------------

    def _apply_soft_guidance(self, action, hour, net_solar_surplus, net_demand,
                             bess_soc, peak_import_target):
        """
        Apply soft action guidance to steer the RL agent toward sensible
        behaviour without hard-blocking.  Returns modified action array.

        Instead of hard-blocking the RL agent (desired_power = 0.0),
        we use "soft guidance": steer the action toward what the
        rules want but still let the agent deviate by a small margin.
        Hard blocks are kept ONLY for true physical constraints
        (SoC limits, ramp rates, voltage safety).
        """
        action_modified = action.copy()
        bess_action_idx = 2

        # --- BESS: solar-surplus charging guidance (soft) ---
        if net_solar_surplus > 0 and bess_soc < 0.8:
            if 6 <= hour < 18:
                charge_intensity = min(1.0, net_solar_surplus / (0.8 * self.bess_power))
            else:
                charge_intensity = min(1.0, net_solar_surplus / self.bess_power)
            # SOFT: blend agent action toward charging, not a hard override
            target = -charge_intensity
            action_modified[bess_action_idx] = 0.6 * target + 0.4 * action[bess_action_idx]

        # --- BESS: peak-hour discharge shaping (soft) ---
        if peak_import_target is not None:
            shaped_discharge = np.clip(net_demand - peak_import_target, 0.0, self.bess_power)
            max_bess_d = self._max_discharge_power_from_soc(bess_soc, self.bess_cap, self.bess_power)
            shaped_discharge = min(shaped_discharge, max_bess_d)
            target = shaped_discharge / self.bess_power if self.bess_power > 0 else 0.0
            # SOFT: blend toward shaped target
            action_modified[bess_action_idx] = 0.7 * target + 0.3 * action[bess_action_idx]

        # --- Home batteries: daytime solar charging guidance (soft) ---
        if 6 <= hour < 18 and net_solar_surplus > 0:
            home_batt_avg_soc = np.mean([self.soc[0], self.soc[1]])
            if home_batt_avg_soc < 0.8:
                home_charge_share = min(0.7, net_solar_surplus / 8.0)
                for hb_idx in range(len(self.home_batt_indices)):
                    if self.soc[hb_idx] < 0.80:
                        home_charge_intensity = min(0.9, home_charge_share / self.home_batt_power)
                        target = -home_charge_intensity
                        # SOFT: blend
                        action_modified[hb_idx] = 0.6 * target + 0.4 * action[hb_idx]

        # --- Off-peak BESS predictive charging (soft) ---
        if hour < 6:
            steps_remaining   = self.max_steps - self.current_step
            steps_until_peak  = min(steps_remaining, int((18 - hour) * 4))

            if steps_until_peak > 4:
                future_solar = self.solar_episode[self.current_step:self.current_step + steps_until_peak]
                future_load  = self.load_episode[self.current_step:self.current_step + steps_until_peak]
                expected_daytime_surplus = np.sum(future_solar) - np.sum(future_load)

                if expected_daytime_surplus > 0:
                    expected_surplus_energy = expected_daytime_surplus * 0.25 * 0.95
                    energy_needed = max(0, (0.8 - bess_soc) * self.bess_cap)

                    if expected_surplus_energy >= energy_needed:
                        # Solar will be enough – gently discourage grid charging
                        action_modified[bess_action_idx] = np.clip(action[bess_action_idx], -0.1, 1.0)
                    else:
                        energy_deficit = energy_needed - expected_surplus_energy
                        if hour >= 23:
                            steps_until_daytime = int((24 - hour) * 4 + 6 * 4)
                        else:
                            steps_until_daytime = int((6 - hour) * 4)

                        if steps_until_daytime > 0:
                            power_per_step    = energy_deficit / (steps_until_daytime * 0.25)
                            charge_intensity  = min(1.0, power_per_step / self.bess_power)
                        else:
                            charge_intensity = 0.5

                        target = -charge_intensity
                        action_modified[bess_action_idx] = 0.6 * target + 0.4 * action[bess_action_idx]
                else:
                    # No daytime solar expected – encourage grid charging
                    if bess_soc < 0.8:
                        if hour >= 23:
                            steps_until_daytime = int((24 - hour) * 4 + 6 * 4)
                        else:
                            steps_until_daytime = int((6 - hour) * 4)

                        if steps_until_daytime > 0:
                            power_per_step   = ((0.8 - bess_soc) * self.bess_cap) / (steps_until_daytime * 0.25)
                            charge_intensity = min(1.0, power_per_step / self.bess_power)
                            target = -charge_intensity
                            action_modified[bess_action_idx] = 0.6 * target + 0.4 * action[bess_action_idx]
            else:
                if bess_soc < 0.4:
                    action_modified[bess_action_idx] = np.clip(action[bess_action_idx], -1.0, -0.5)
                elif bess_soc < 0.6:
                    action_modified[bess_action_idx] = np.clip(action[bess_action_idx], -0.7, 0.0)

        return action_modified

    # ------------------------------------------------------------------
    def _apply_physics_constraints(self, action_modified, hour, net_solar_surplus,
                                    remaining_solar_surplus, total_solar, total_load):
        """
        Apply hard physical constraints (SoC, ramp, voltage safety) and
        update battery SoC.  Returns (remaining_demand, remaining_solar_surplus,
        prev_batt_power_copy).
        """
        self.node_battery_power_kw = np.zeros(self.n_nodes, dtype=np.float32)
        prev_batt_power_copy = self.prev_batt_power.copy()
        remaining_demand = max(0.0, total_load - total_solar)

        for i, node_idx in enumerate(self.storage_map):

            is_bess  = (node_idx == self.bess_index)
            p_max    = self.bess_power      if is_bess else self.home_batt_power
            cap      = self.bess_cap        if is_bess else self.home_batt_cap

            # FIX V2: time-aware ramp rate for BESS
            # Off-peak charging uses tighter ramp to prevent voltage sag
            if is_bess:
                ramp = self.bess_ramp_offpeak if hour < 6 else self.bess_ramp_peak
            else:
                ramp = self.home_batt_ramp

            desired_power = action_modified[i] * p_max

            # -- Hard physical constraints (these stay hard) --

            # CONSTRAINT 1: SoC limits
            if self.soc[i] <= 0.2 and desired_power > 0:
                desired_power = 0.0
            if self.soc[i] >= 0.8 and desired_power < 0:
                desired_power = 0.0

            # CONSTRAINT 2: Block ALL BESS activity 11pm–midnight (transition hour)
            # BESS must be idle during this period — no charging or discharging
            if is_bess and 23 <= hour:
                desired_power = 0.0
            # Home batteries: still block charging only
            elif not is_bess and 23 <= hour and desired_power < 0:
                desired_power = 0.0

            # CONSTRAINT 3: Block ALL charging during peak (6pm–11pm)
            if 18 <= hour < 23 and desired_power < 0:
                desired_power = 0.0

            # CONSTRAINT 4: Home batteries – only discharge during peak (6pm–11pm)
            # (kept as hard rule because discharging home batteries at other times
            #  always causes overvoltage in the feeder)
            if not is_bess and desired_power > 0:
                if not (18 <= hour < 23):
                    desired_power = 0.0

            # CONSTRAINT 5: BESS – block discharge during daytime with solar surplus
            # (kept hard: solar + BESS discharge simultaneously → certain overvoltage)
            if is_bess and desired_power > 0 and 6 <= hour < 18:
                if net_solar_surplus > 0:
                    desired_power = 0.0

            # CONSTRAINT 6: Voltage-safety discharge limit for BESS
            if is_bess and desired_power > 0:
                if total_solar > 0.8 * remaining_demand:
                    desired_power = min(desired_power, 0.2 * p_max)

            # CONSTRAINT 7: Cap combined solar+battery injection to avoid overvoltage
            if total_solar + max(0, desired_power) > total_load + 1.5:
                excess = total_solar + max(0, desired_power) - total_load - 1.5
                if is_bess and desired_power > 0:
                    if 6 <= hour < 18 and excess > 0.5:
                        desired_power = 0.0
                    else:
                        desired_power = max(0, desired_power - excess * 0.8)

            # CONSTRAINT 8: BESS solar-charging uses available surplus
            if is_bess and desired_power < 0 and 6 <= hour < 18 and remaining_solar_surplus > 0:
                if self.soc[i] < 0.8:
                    available_charge = min(
                        remaining_solar_surplus,
                        self._max_charge_power_from_headroom(self.soc[i], cap, p_max),
                    )
                    desired_power = -available_charge
                else:
                    desired_power = 0.0

            # CONSTRAINT 9: Home battery daytime solar charging
            if not is_bess and 6 <= hour < 18 and desired_power < 0:
                if net_solar_surplus > 0:
                    available_charge = min(
                        remaining_solar_surplus,
                        self._max_charge_power_from_headroom(self.soc[i], cap, p_max),
                    )
                    desired_power = -available_charge
                else:
                    desired_power = 0.0

            # Limit discharge to remaining demand
            if desired_power > 0:
                desired_power = min(desired_power, remaining_demand)
                remaining_demand -= desired_power

            # -- Ramp rate --
            delta_p     = np.clip(desired_power - self.prev_batt_power[i], -ramp, ramp)
            final_power = np.clip(self.prev_batt_power[i] + delta_p, -p_max, p_max)

            # Voltage safety trim (hard – prevents physical damage)
            if is_bess and final_power > 0 and self.voltages[self.bess_index] > 1.03:
                voltage_factor = max(0.2, min(1.0, (1.06 - self.voltages[self.bess_index]) / (1.06 - 1.03)))
                final_power = final_power * voltage_factor

            # ==============================================================
            # FIX V1 – VOLTAGE-AWARE CHARGE LIMITING
            # When BESS draws heavy current during off-peak (e.g. −40 kW),
            # it causes a large voltage drop at end-of-feeder Node 21
            # (seen as 0.90 p.u. in the uploaded image).
            # Solution: scale back charging power proportionally when the
            # BESS node voltage is already approaching the 0.94 p.u. limit.
            #
            # Scaling zones:
            #   voltage >= 0.97  → full charging allowed  (factor = 1.0)
            #   voltage == 0.955 → 50% charging           (factor = 0.5)
            #   voltage <= 0.94  → charging blocked        (factor = 0.0)
            # ==============================================================
            if is_bess and final_power < 0:  # BESS is charging
                bess_voltage = self.voltages[self.bess_index]
                if bess_voltage < 0.97:
                    # Linear scale: 0.0 at 0.94, 1.0 at 0.97
                    voltage_factor = max(0.0, (bess_voltage - 0.94) / (0.97 - 0.94))
                    final_power = final_power * voltage_factor
                    if self.verbose and voltage_factor < 1.0:
                        print(f"[VOLTAGE_CHARGE_LIMIT] Hour {hour:.1f}: BESS voltage "
                              f"{bess_voltage:.4f} p.u. → charge scaled to "
                              f"{final_power:.1f} kW (factor {voltage_factor:.2f})")

            # Update SoC
            eff = 0.95
            if final_power >= 0:
                self.soc[i] -= (final_power * 0.25) / eff / cap
            else:
                self.soc[i] -= (final_power * 0.25 * eff) / cap

            self.soc[i] = np.clip(self.soc[i], 0.2, 0.8)

            # Track remaining solar surplus consumed by charging
            if 6 <= hour < 18 and final_power < 0 and remaining_solar_surplus > 0:
                remaining_solar_surplus = max(0.0, remaining_solar_surplus - min(-final_power, remaining_solar_surplus))

            self.node_battery_power_kw[node_idx] = final_power
            self.prev_batt_power[i] = final_power

        return remaining_demand, remaining_solar_surplus, prev_batt_power_copy

    # ------------------------------------------------------------------
    def _run_opendss(self, full_solar_profile, full_load_profile):
        """Run OpenDSS power flow and update voltage arrays.

        Returns True if converged, False otherwise.
        """
        self.net_injection = full_solar_profile + self.node_battery_power_kw - full_load_profile

        loads_kw = full_load_profile[:21].tolist()
        pv_kw    = {idx: float(full_solar_profile[idx]) for idx in self.solar_indices}

        step_res = self.dss_runner.step(
            loads_kw=loads_kw,
            pv_kw=pv_kw,
            batt_home3_kw=float(self.node_battery_power_kw[3]),
            batt_home5_kw=float(self.node_battery_power_kw[5]),
            bess_kw=float(self.node_battery_power_kw[21]),
            auto_compile=False
        )

        # FIX 1.4: Check convergence — garbage voltages corrupt reward signals
        if not step_res.converged:
            if self.verbose:
                print(f"[WARNING] OpenDSS did not converge at step {self.current_step}")
            # Use safe fallback voltages (nominal)
            self.voltages     = np.ones(self.n_nodes, dtype=np.float32)
            self.voltages_min = np.ones(self.n_nodes, dtype=np.float32)
            self.voltages_max = np.ones(self.n_nodes, dtype=np.float32)
            return False

        self.voltages     = np.array(step_res.vmin_pu_by_bus, dtype=np.float32)
        self.voltages_min = self.voltages.copy()
        self.voltages_max = np.zeros(self.n_nodes, dtype=np.float32)

        for i, (va, vb, vc) in enumerate(step_res.vabc_pu_by_bus):
            phases = [v for v in [va, vb, vc] if not np.isnan(v)]
            self.voltages_max[i] = max(phases) if phases else 1.0

        return True

    # ------------------------------------------------------------------
    def _compute_reward(self, hour, net_demand, net_solar_surplus, total_solar,
                        total_load, prev_batt_power_copy, peak_import_target,
                        converged, remaining_demand):
        """Compute the shaped reward signal.

        Returns (reward, reward_info_dict).
        """
        # --- Non-convergence short-circuit ---
        if not converged:
            return self.R_CONVERGENCE_PENALTY, {
                "revenue": 0.0, "cost": 0.0, "profit": 0.0,
                "grid_export_revenue": 0.0, "grid_import_cost": 0.0,
                "bess_discharge_revenue": 0.0, "bess_charge_cost": 0.0,
                "voltage_penalty": self.R_CONVERGENCE_PENALTY * self.R_NORMALIZER,
                "clean_peak_bonus": 0.0,
                "grid_smoothing_penalty": 0.0,
                "grid_power_change": 0.0,
                "current_grid_net": 0.0,
                "peak_import_target": peak_import_target,
                "total_violation": 0.0,
                "remaining_demand": remaining_demand,
            }

        # Time-of-use pricing (LKR)
        if 6 <= hour < 18:
            buy_price, sell_price = 35, 19
        elif 18 <= hour < 23:
            buy_price, sell_price = 67, 45
        else:
            buy_price, sell_price = 21, 0

        grid_export = np.maximum(0,  self.net_injection)
        grid_import = np.maximum(0, -self.net_injection)

        grid_export_revenue = np.sum(grid_export * sell_price)
        grid_import_cost    = np.sum(grid_import * buy_price)

        bess_power_val = self.node_battery_power_kw[self.bess_index]
        bess_discharge_revenue = bess_power_val * sell_price if bess_power_val > 0 else 0.0
        bess_charge_cost = (abs(bess_power_val) * buy_price
                            if bess_power_val < 0 and net_solar_surplus <= 0 else 0.0)

        revenue = grid_export_revenue
        cost    = grid_import_cost

        # ---- A. Daytime solar bonuses ----
        daytime_solar_bonus = 0.0
        solar_waste_penalty = 0.0

        if 6 <= hour < 18:
            total_charge_power    = np.sum(np.minimum(0, self.node_battery_power_kw))
            total_discharge_power = np.sum(np.maximum(0, self.node_battery_power_kw))

            if net_solar_surplus > 0:
                daytime_solar_bonus += self.R_SOLAR_CHARGE_BONUS * (-total_charge_power)
                bess_charge_power = np.minimum(0, self.node_battery_power_kw[self.bess_index])
                surplus_factor = min(1.0, net_solar_surplus / 20.0)
                daytime_solar_bonus += self.R_BESS_SOLAR_CHARGE * (-bess_charge_power) * (1.0 + surplus_factor)

                avg_soc = np.mean(self.soc)
                if avg_soc < 0.8:
                    solar_wasted = net_solar_surplus - (-total_charge_power)
                    if solar_wasted > 0.2:
                        solar_waste_penalty = self.R_SOLAR_WASTE_PENALTY * solar_wasted
            else:
                bess_discharge_power = np.maximum(0, self.node_battery_power_kw[self.bess_index])
                daytime_solar_bonus += self.R_BESS_DAYTIME_DISCHARGE * bess_discharge_power

            home_batt_solar_charge = np.sum([np.minimum(0, self.node_battery_power_kw[idx])
                                             for idx in self.home_batt_indices])
            if home_batt_solar_charge < 0 and net_solar_surplus > 0:
                daytime_solar_bonus += self.R_HOME_SOLAR_CHARGE * (-home_batt_solar_charge)

        # ---- B. Peak bonuses ----
        peak_bonus         = 0.0
        peak_charge_penalty = 0.0
        clean_peak_bonus   = 0.0

        if 18 <= hour < 23:
            if np.mean(self.soc) > 0.3:
                total_discharge_power = np.sum(np.maximum(0, self.node_battery_power_kw))
                peak_bonus = self.R_PEAK_DISCHARGE_BONUS * total_discharge_power

            # FIX 1.2: Use node_idx (not i) to read battery power at correct nodes.
            # Previously used self.node_battery_power_kw[i] which read nodes 0,1,2
            # instead of the actual battery nodes 3,5,21.
            for i, node_idx in enumerate(self.storage_map):
                charge_power = np.minimum(0, self.node_battery_power_kw[node_idx])
                if charge_power < 0:
                    peak_charge_penalty += self.R_PEAK_CHARGE_PENALTY * (-charge_power)

            # Clean peak discharge bonus
            if (np.all(self.voltages_min[self.critical_nodes] >= 0.94) and
                    np.all(self.voltages_max[self.critical_nodes] <= 1.06)):
                # No violations – reward every kW of clean discharge
                clean_discharge = np.sum(np.maximum(0, self.node_battery_power_kw))
                clean_peak_bonus = self.R_CLEAN_PEAK_BONUS * clean_discharge

        # ---- C. Off-peak bonuses ----
        offpeak_bonus = 0.0
        if hour < 6:
            home_batt_soc = [self.soc[0], self.soc[1]]
            if np.mean(home_batt_soc) < 0.8:
                home_charge_power = (self.node_battery_power_kw[3]
                                     + self.node_battery_power_kw[5])
                home_charge_power = min(0, home_charge_power)

                solar_will_be_sufficient = False
                steps_ahead = min(96, self.max_steps - self.current_step)
                if steps_ahead > 24:
                    future_solar = self.solar_episode[self.current_step:self.current_step + steps_ahead]
                    future_load  = self.load_episode[self.current_step:self.current_step + steps_ahead]
                    daylight_start = max(0, int((6 - hour) * 4))
                    daylight_end   = min(steps_ahead, daylight_start + 48)
                    if daylight_end > daylight_start:
                        expected_surplus = (np.sum(future_solar[daylight_start:daylight_end])
                                            - np.sum(future_load[daylight_start:daylight_end]))
                        energy_needed = 2 * self.home_batt_cap * (0.8 - np.mean(home_batt_soc))
                        if expected_surplus > energy_needed * 0.9:
                            solar_will_be_sufficient = True

                if solar_will_be_sufficient:
                    offpeak_bonus = self.R_HOME_OFFPEAK_SOLAR_OK * home_charge_power
                else:
                    offpeak_bonus = self.R_HOME_OFFPEAK_NEEDED * home_charge_power

            # BESS off-peak charging
            bess_soc        = self.soc[2]
            bess_charge_pwr = self.node_battery_power_kw[self.bess_index]
            if bess_charge_pwr < 0:
                steps_remaining  = self.max_steps - self.current_step
                steps_until_peak = min(steps_remaining, int((18 - hour) * 4))
                if steps_until_peak > 4:
                    fs = self.solar_episode[self.current_step:self.current_step + steps_until_peak]
                    fl = self.load_episode[self.current_step:self.current_step + steps_until_peak]
                    expected_surplus_energy = (np.sum(fs) - np.sum(fl)) * 0.25 * 0.95
                    energy_needed = (0.75 - bess_soc) * self.bess_cap
                    if expected_surplus_energy < energy_needed:
                        offpeak_bonus += -10.0 * bess_charge_pwr
                    else:
                        offpeak_bonus += -5.0 * bess_charge_pwr
                else:
                    offpeak_bonus += (-8.0 if bess_soc < 0.4 else -5.0) * bess_charge_pwr

        # ---- D. Grid smoothing penalty ----
        current_grid_net  = np.sum(self.net_injection)
        grid_power_change = abs(current_grid_net - self.prev_grid_net_power)
        if grid_power_change > self.R_RAMP_THRESHOLD:
            excess_ramp          = grid_power_change - self.R_RAMP_THRESHOLD
            grid_smoothing_penalty = self.R_RAMP_PENALTY_RATE * excess_ramp
        else:
            grid_smoothing_penalty = 0.0
        self.prev_grid_net_power = current_grid_net

        # ---- E. Voltage penalties & bonuses ----
        min_voltages   = self.voltages_min[self.critical_nodes]
        max_voltages   = self.voltages_max[self.critical_nodes]

        under_voltage   = np.maximum(0, 0.94 - min_voltages)
        over_voltage    = np.maximum(0, max_voltages - 1.06)
        total_violation = np.sum(over_voltage + under_voltage)

        # Three-tier voltage penalty
        raw_violations  = over_voltage + under_voltage

        soft_violations   = np.minimum(0.03, raw_violations)
        hard_violations   = np.maximum(0.0, np.minimum(raw_violations, 0.06) - 0.03)
        severe_violations = np.maximum(0.0, raw_violations - 0.06)

        voltage_penalty = (
            self.R_VOLT_SOFT_PER_PU   * np.sum(soft_violations)
            + self.R_VOLT_HARD_PER_PU   * np.sum(hard_violations)
            + self.R_VOLT_SEVERE_PER_PU * np.sum(severe_violations)
        )

        if self.verbose and np.sum(severe_violations) > 0:
            print(f"[SEVERE_VIOLATION] Hour {hour:.1f}: "
                  f"{np.sum(severe_violations > 0)} nodes with severe violations "
                  f"(min={np.min(min_voltages):.4f} p.u.) → penalty {voltage_penalty:.0f}")

        ideal_nodes      = np.sum((min_voltages >= 0.98) & (max_voltages <= 1.02))
        acceptable_nodes = np.sum((min_voltages >= 0.94) & (max_voltages <= 1.06))
        voltage_stability_bonus = (self.R_IDEAL_NODE_BONUS * ideal_nodes
                                   + self.R_ACCEPTABLE_NODE_BONUS * (acceptable_nodes - ideal_nodes))

        # ---- F. Battery cycling & SoC health ----
        final_power_array = np.array([self.node_battery_power_kw[n] for n in self.storage_map])
        power_changes     = final_power_array - prev_batt_power_copy
        cycling_cost      = (self.R_CYCLING_LINEAR * np.sum(np.abs(power_changes))
                             + self.R_CYCLING_QUADRATIC * np.sum(power_changes ** 2))

        soc_health_penalty = 0.0
        for i in range(len(self.soc)):
            if self.soc[i] < 0.2:
                soc_health_penalty += self.R_SOC_HEALTH_PENALTY * (0.2 - self.soc[i]) ** 2
            elif self.soc[i] > 0.8:
                soc_health_penalty += self.R_SOC_HEALTH_PENALTY * (self.soc[i] - 0.8) ** 2

        # ---- G. Total reward ----
        reward = (
            revenue * self.R_ECONOMIC_WEIGHT
            - cost   * self.R_ECONOMIC_WEIGHT
            + voltage_penalty
            + voltage_stability_bonus
            + daytime_solar_bonus
            + solar_waste_penalty
            + peak_bonus
            + clean_peak_bonus
            + peak_charge_penalty
            + offpeak_bonus
            + grid_smoothing_penalty
            + cycling_cost
            + soc_health_penalty
        )

        # Normalise and clip
        reward = reward / self.R_NORMALIZER
        reward = float(np.clip(reward, self.R_CLIP_LOW, self.R_CLIP_HIGH))

        if self.verbose and self.current_step % 24 == 0:
            print(f"[REWARD_DEBUG] Hour {hour:.1f}:")
            print(f"  Economic: revenue={revenue:.0f}, cost={cost:.0f}, "
                  f"net_scaled={(revenue-cost)*self.R_ECONOMIC_WEIGHT:.0f}")
            print(f"  Voltages: min={np.min(min_voltages):.4f}, max={np.max(max_voltages):.4f}, "
                  f"penalty={voltage_penalty:.0f}, bonus={voltage_stability_bonus:.0f}")
            print(f"  Battery: cycling={cycling_cost:.0f}, soc_health={soc_health_penalty:.0f}")
            print(f"  Bonuses: daytime={daytime_solar_bonus:.0f}, peak={peak_bonus:.0f}, "
                  f"clean_peak={clean_peak_bonus:.0f}, offpeak={offpeak_bonus:.0f}")
            print(f"  Total reward (after norm+clip): {reward:.4f}")

        reward_info = {
            "revenue": revenue,
            "cost": cost,
            "profit": revenue - cost,
            "grid_export_revenue": grid_export_revenue,
            "grid_import_cost": grid_import_cost,
            "bess_discharge_revenue": bess_discharge_revenue,
            "bess_charge_cost": bess_charge_cost,
            "voltage_penalty": voltage_penalty,
            "clean_peak_bonus": clean_peak_bonus,
            "grid_smoothing_penalty": grid_smoothing_penalty,
            "grid_power_change": grid_power_change,
            "current_grid_net": current_grid_net,
            "peak_import_target": peak_import_target,
            "total_violation": total_violation,
            "remaining_demand": remaining_demand,
        }

        return reward, reward_info

    # ------------------------------------------------------------------
    def step(self, action):
        """Run one environment step: guidance → physics → simulation → reward."""

        # --- 1. GET DATA ---
        full_solar_profile = np.zeros(self.n_nodes, dtype=np.float32)
        full_load_profile  = np.zeros(self.n_nodes, dtype=np.float32)
        full_solar_profile[:21] = self.solar_episode[self.current_step]
        full_load_profile[:21]  = self.load_episode[self.current_step]

        total_load  = float(np.sum(full_load_profile))
        total_solar = float(np.sum(full_solar_profile))
        hour = (self.current_step % 96) / 4.0

        net_demand        = max(0.0, total_load - total_solar)
        net_solar_surplus = total_solar - total_load
        remaining_solar_surplus  = max(0.0, net_solar_surplus)
        peak_import_target       = self._compute_peak_import_target(hour)

        bess_soc = self.soc[2]

        # --- 2. SOFT ACTION GUIDANCE ---
        action_modified = self._apply_soft_guidance(
            action, hour, net_solar_surplus, net_demand,
            bess_soc, peak_import_target
        )

        # --- 3. PHYSICS: APPLY ACTIONS ---
        remaining_demand, remaining_solar_surplus, prev_batt_power_copy = \
            self._apply_physics_constraints(
                action_modified, hour, net_solar_surplus,
                remaining_solar_surplus, total_solar, total_load
            )

        # --- 4. OpenDSS PHYSICS ---
        converged = self._run_opendss(full_solar_profile, full_load_profile)

        # --- 5. REWARD CALCULATION ---
        reward, reward_info = self._compute_reward(
            hour, net_demand, net_solar_surplus, total_solar,
            total_load, prev_batt_power_copy, peak_import_target,
            converged, remaining_demand
        )

        # --- 6. TRANSITION ---
        self.current_step += 1
        terminated = (self.current_step >= self.max_steps)
        truncated  = False

        # Terminal SoC reward: incentivise leaving batteries near 50% at end-of-day
        if terminated:
            terminal_penalty = 0.0
            for i in range(self.n_storage_units):
                soc_deviation = abs(self.soc[i] - 0.5)
                terminal_penalty += self.R_TERMINAL_SOC_PENALTY * soc_deviation
            reward += terminal_penalty / self.R_NORMALIZER
            reward = float(np.clip(reward, self.R_CLIP_LOW, self.R_CLIP_HIGH))

        obs = self._get_obs() if not terminated else self.state

        info = {
            "hour": hour,
            "net_demand": net_demand,
            "remaining_demand": remaining_demand,
            "max_voltage": float(np.max(self.voltages_max[self.critical_nodes])),
            "min_voltage": float(np.min(self.voltages_min[self.critical_nodes])),
            "violation": reward_info["total_violation"],
            "solar_surplus": net_solar_surplus,
            "total_load": total_load,
            "total_solar": total_solar,
            "revenue": reward_info["revenue"],
            "cost": reward_info["cost"],
            "profit": reward_info["profit"],
            "grid_export_revenue": reward_info["grid_export_revenue"],
            "grid_import_cost": reward_info["grid_import_cost"],
            "bess_discharge_revenue": reward_info["bess_discharge_revenue"],
            "bess_charge_cost": reward_info["bess_charge_cost"],
            "soc_home3": self.soc[0],
            "soc_home5": self.soc[1],
            "soc_bess": self.soc[2],
            "bess_power": self.node_battery_power_kw[self.bess_index],
            "voltage_penalty": reward_info["voltage_penalty"],
            "clean_peak_bonus": reward_info["clean_peak_bonus"],
            "grid_smoothing_penalty": reward_info["grid_smoothing_penalty"],
            "grid_power_change": reward_info["grid_power_change"],
            "current_grid_net": reward_info["current_grid_net"],
            "peak_import_target": reward_info["peak_import_target"],
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Constructs and normalises the 51-value observation vector.

        Raw values are scaled so that typical operating ranges map to [-1, 1].
        The observation space is declared as [-2, 2] to absorb rare extremes
        without triggering SB3 env-checker bounds violations.

        Normalisation per group
        -----------------------
        [0]     Solar  / OBS_SOLAR_MAX          → 0..1  (non-negative)
        [1-21]  Load   / OBS_LOAD_MAX           → 0..1  (non-negative)
        [22-43] (Voltage - 1.0) / OBS_VOLT_RANGE→ ~-1..+1 centred on nominal
        [44-46] (SoC - 0.5) / 0.3               → -1..+1 centred on midpoint
        [47-50] sin/cos features                → already in [-1, 1]
        """
        if self.current_step < self.max_steps:
            common_solar_raw = np.array([self.solar_episode[self.current_step][0]], dtype=np.float32)
            load_step_raw    = self.load_episode[self.current_step]
        else:
            common_solar_raw = np.array([0.0], dtype=np.float32)
            load_step_raw    = np.zeros(21, dtype=np.float32)

        # --- Normalise solar (0 to ~1) ---
        common_solar_norm = common_solar_raw / self.OBS_SOLAR_MAX

        # --- Normalise load (0 to ~1) ---
        load_step_norm = load_step_raw / self.OBS_LOAD_MAX

        # --- Normalise voltages: centre on 1.0, scale by 0.15 ---
        # 0.85 p.u. → -1.0,  1.00 p.u. → 0.0,  1.15 p.u. → +1.0
        volt_norm = (self.voltages - self.OBS_VOLT_NOM) / self.OBS_VOLT_RANGE

        # --- Normalise SoC: [0.2, 0.8] → [-1, +1] ---
        soc_norm = (self.soc - self.OBS_SOC_CENTER) / self.OBS_SOC_RANGE

        # --- Time features: sin/cos already in [-1, 1] ---
        time_angle = (self.current_step / self.max_steps) * 2 * np.pi
        day_angle  = ((self.start_idx // 96) / 365.0) * 2 * np.pi
        date_time_feats = np.array([
            np.sin(time_angle), np.cos(time_angle),
            np.sin(day_angle),  np.cos(day_angle)
        ], dtype=np.float32)

        self.state = np.concatenate([
            common_solar_norm,   # 1   → [0, 1]
            load_step_norm,      # 21  → [0, 1]
            volt_norm,           # 22  → [-1, +1]
            soc_norm,            # 3   → [-1, +1]
            date_time_feats      # 4   → [-1, +1]
        ]).astype(np.float32)

        return self.state