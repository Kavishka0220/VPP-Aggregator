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
    Optimised VPP Environment — Charging Strategy v2
    =================================================

    CHARGING PRIORITY (in order):
    ─────────────────────────────
    1. HOME SOLAR SURPLUS  → home batteries charge from their own local PV surplus.
    2. FEEDER SOLAR SURPLUS → if no local surplus, home batteries charge from the
                               feeder-wide shared solar surplus (split equally).
       BESS also charges from feeder surplus (BESS first, then home batteries).
    3. OFF-PEAK GRID  → charge only when no solar surplus is available (home
                        batteries and BESS). BLOCKED during 22:30–00:15.
    4. DAYTIME TOP-UP → if solar + off-peak still leaves BESS below ~0.75 SoC
                        by 16:00, allow limited daytime grid charging so batteries
                        are full before peak starts at 18:30.

    DISCHARGE:
    ──────────
    • ALL storage discharges ONLY during peak (18:30–22:30).
    • BESS discharge is voltage-aware: throttle if node voltage exceeds 1.03 p.u.
    • Home batteries also voltage-aware: throttle if their node voltage > 1.03 p.u.
    • Peak-shaving target computed by binary search over remaining peak steps.
    • Both BESS and home batteries must reach SoC ≥ 0.2 by end of peak.

    SMOOTH RAMP ENFORCEMENT:
    ────────────────────────
    • Hard ramp limits: BESS 10 kW/step (peak), 5 kW/step (off-peak/daytime charge).
    • Reward *strongly* penalises step changes > SMOOTH_RAMP_KW (2 kW for home,
      5 kW for BESS).  This shapes the RL agent toward gradual transitions even
      within the allowed ramp envelope.
    • Reward *rewards* consistent direction (charge or discharge sustained across
      multiple steps) to discourage oscillation.

    VOLTAGE:
    ────────
    • Hard constraint: all nodes must stay within 0.94–1.06 p.u.
    • Three-tier penalty: soft (0–0.03 violation), hard (0.03–0.06), severe (>0.06).
    • Voltage-aware charge limiting on BESS *and* home batteries.
    • Ideal-node bonus for nodes in 0.98–1.02 p.u.

    KEY CHANGES vs. v1:
    ───────────────────
    A. Late-night no-charge window (22:30–00:15) enforced as hard constraint.
    B. Daytime top-up logic: if pre-peak SoC gap exists, allow limited grid top-up
       after 16:00 to ensure full charge before 18:30.
    C. Smooth-ramp reward is now a *positive* bonus for small steps plus a
       stronger *penalty* for large steps (was only penalty before).
    D. Home batteries now voltage-aware during charging (mirrors BESS logic).
    E. Pre-peak full-charge bonus: reward BESS SoC > 0.75 at 17:00–18:30.
    F. Off-peak solar forecast uses more accurate look-ahead (accounts for
       night hours before dawn when computing expected daytime surplus).
    G. Peak shaving smoothness bonus: reward flat discharge profile.
    """

    metadata = {'render_modes': []}

    # ==================== REWARD TUNING CONSTANTS ====================
    # Economic
    R_ECONOMIC_WEIGHT        = 0.6

    # Daytime solar
    R_SOLAR_CHARGE_BONUS     = 20.0    # per kW of solar charging (any battery)
    R_BESS_SOLAR_CHARGE      = 20.0    # extra per kW of BESS solar charging (priority)
    R_HOME_SOLAR_CHARGE      = 15.0    # per kW of home-battery solar charging
    R_SOLAR_WASTE_PENALTY    = -15.0   # per kW of wasted solar (batteries not full)
    R_BESS_DAYTIME_DISCHARGE = -30.0   # per kW inappropriate BESS discharge outside peak

    # Pre-peak readiness  (NEW)
    # Reward BESS being well-charged during the 2h window before peak (16:30–18:30)
    R_PRE_PEAK_READY_BONUS   = 4.0     # per 0.01 SoC above floor at 05:30–18:30
    R_PRE_PEAK_FLOOR         = 0.20    # SoC floor that earns this bonus

    # Peak hours
    R_PEAK_DISCHARGE_BONUS   = 40.0    # per kW of peak discharge
    R_CLEAN_PEAK_BONUS       = 55.0    # per kW of clean (no-violation) peak discharge
    R_PEAK_CHARGE_PENALTY    = -80.0   # per kW of charging during peak (stronger)

    # Peak shaving smoothness  (NEW)
    # Reward a flat discharge profile: penalise per-kW deviation from the
    # running average discharge power across peak steps so far.
    R_PEAK_SMOOTH_BONUS      = 3.0     # per kW below target variance

    # Off-peak
    R_OFFPEAK_BESS_HEADROOM  = 12.0    # per kW of off-peak BESS charging (headroom-scaled)
    R_HOME_OFFPEAK_SOLAR_OK  = 4.0     # home charging when solar will suffice
    R_HOME_OFFPEAK_NEEDED    = -8.0    # home charging when solar won't suffice

    # Daytime top-up  (NEW)
    # Small bonus for BESS charging from grid after 16:00 to fill gap before peak
    R_DAYTIME_TOPUP_BONUS    = 8.0     # per kW of pre-peak top-up grid charging

    # Smooth ramp  (REDESIGNED)
    R_SMOOTH_RAMP_REWARD     = 1.5     # per kW *below* ramp threshold (positive signal)
    R_RAMP_PENALTY_SOFT      = -0.5    # per kW between 0 and 2× threshold
    R_RAMP_PENALTY_HARD      = -2.0    # per kW above 2× threshold (oscillation)
    R_BESS_SMOOTH_THRESHOLD  = 5.0     # kW; BESS ramp below this earns bonus
    R_HOME_SMOOTH_THRESHOLD  = 2.0     # kW; home-battery ramp below this earns bonus
    R_DIRECTION_BONUS        = 0.5     # per step of sustained same-direction operation

    # Grid smoothing
    R_RAMP_THRESHOLD         = 5.0     # kW change before grid-ramp penalty kicks in
    R_RAMP_PENALTY_RATE      = -0.15   # per kW above threshold

    # Voltage
    R_VOLT_SOFT_PER_PU       = -50.0   # 0 to 0.03 p.u. violation
    R_VOLT_HARD_PER_PU       = -120.0  # 0.03 to 0.06 p.u. violation
    R_VOLT_SEVERE_PER_PU     = -600.0  # beyond 0.06 p.u. violation
    R_IDEAL_NODE_BONUS       = 50.0    # per node in [0.98, 1.02]
    R_ACCEPTABLE_NODE_BONUS  = 6.0     # per node in [0.94, 1.06] but not ideal

    # Battery health
    R_CYCLING_LINEAR         = -0.3
    R_CYCLING_QUADRATIC      = -0.1
    R_SOC_HEALTH_PENALTY     = -50.0

    # Terminal
    R_TERMINAL_SOC_PENALTY   = -2.5    # per unit SoC deviation from 0.5

    # Normalisation
    R_NORMALIZER             = 200.0
    R_CLIP_LOW               = -10.0
    R_CLIP_HIGH              = 10.0

    # Convergence failure
    R_CONVERGENCE_PENALTY    = -5.0
    # =================================================================

    # Guidance blend: 0.4 = 40% rule guidance + 60% agent freedom
    GUIDANCE_ALPHA = 0.4

    # Tiling for episode diversity
    N_TILE_DAYS = 30

    # Float32 SoC comparison tolerance
    SOC_EPS = 1e-4

    # Late-night no-charge window: 22:30–00:15
    NO_CHARGE_START = 22.5   # hour
    NO_CHARGE_END   = 0.25   # hour (00:15)

    def __init__(self, data_path="./data", scenario_name=None, start_index=None, verbose=False):
        super(UrbanVPPEnv, self).__init__()

        self.verbose = verbose
        self.default_start_index = start_index
        self.scenario_name = scenario_name

        # OpenDSS
        dss_file = os.path.join(parent_dir, "openDSS", "feeder_houses.dss")
        self.dss_runner = VPPDSSRunner(dss_file)
        self.dss_runner.compile()

        # System config
        self.n_nodes = 22
        self.solar_indices = [3, 5, 7, 10, 11, 13, 15, 17, 18, 19, 20]

        if not all(0 <= idx < 21 for idx in self.solar_indices):
            raise ValueError(f"Solar indices must be in range [0, 20]. Got: {self.solar_indices}")

        self.home_batt_indices = [3, 5]
        self.bess_index = 21
        self.storage_map = self.home_batt_indices + [self.bess_index]
        self.n_storage_units = len(self.storage_map)
        self.critical_nodes = list(range(21)) + [self.bess_index]

        # Specs
        self.home_batt_cap   = 13.5
        self.bess_cap        = 200.0
        self.home_batt_power = 5.0
        self.bess_power      = 40.0

        # Ramp limits
        self.home_batt_ramp    = 2.0
        self.bess_ramp_peak    = 10.0
        self.bess_ramp_offpeak = 5.0

        # Track discharge direction for direction-bonus
        # +1 = discharging, -1 = charging, 0 = idle
        self._prev_direction   = np.zeros(self.n_storage_units, dtype=np.float32)
        self._direction_streak = np.zeros(self.n_storage_units, dtype=np.int32)

        # Peak discharge tracking for smoothness bonus
        self._peak_discharge_powers = []   # list of total discharge kW per step

        # Action space
        # BESS only — home batteries are rule-based
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: 1(Solar) + 21(Loads) + 22(Voltages) + 3(SoCs) + 4(Time) + 1(FeederSurplus) = 52
        self.obs_size = 52

        self.OBS_SOLAR_MAX  = 10.0
        self.OBS_LOAD_MAX   = 10.0
        self.OBS_VOLT_NOM   = 1.0
        self.OBS_VOLT_RANGE = 0.15
        self.OBS_SOC_CENTER = 0.5
        self.OBS_SOC_RANGE  = 0.3

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.obs_size,), dtype=np.float32
        )

        # State variables
        self.state              = None
        self.current_step       = 0
        self.max_steps          = 96
        self.soc                = np.ones(self.n_storage_units, dtype=np.float32) * 0.5
        self.prev_batt_power    = np.zeros(self.n_storage_units, dtype=np.float32)
        self.prev_grid_net_power= 0.0
        self.voltages           = np.ones(self.n_nodes, dtype=np.float32)
        self.voltages_min       = np.ones(self.n_nodes, dtype=np.float32)
        self.voltages_max       = np.ones(self.n_nodes, dtype=np.float32)

        self._load_data(data_path, scenario_name)

    # ------------------------------------------------------------------
    def _is_no_charge_window(self, hour):
        """Return True if hour falls in the no-charge window (22:30–00:15)."""
        # Window wraps midnight: [22.5, 24) ∪ [0, 0.25)
        return hour >= self.NO_CHARGE_START or hour < self.NO_CHARGE_END

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

            for df_name, df in [("solar", self.solar_df), ("load", self.load_df)]:
                ts_cols = [c for c in df.columns if c.lower() in ['timestamp','time','date','datetime']]
                if ts_cols:
                    df.drop(columns=ts_cols, inplace=True)
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.dropna(axis=1, how='all', inplace=True)
                if df.shape[1] > 21:
                    df = df.iloc[:, :21]
                elif df.shape[1] < 21:
                    if df_name == "solar" and df.shape[1] == 1:
                        # Single irradiance column (W/m²) → convert to per-node kW and broadcast to 21 nodes
                        # Factor calibrated to ~4.3 kW peak per node at STC (1000 W/m²)
                        power = df.iloc[:, 0].values * 0.0043
                        df = pd.DataFrame(np.tile(power.reshape(-1, 1), 21),
                                          columns=[f"Node{i+1}" for i in range(21)])
                    else:
                        raise ValueError(f"{df_name} has only {df.shape[1]} columns, need 21")
                if df_name == "solar":
                    self.solar_df = df
                else:
                    self.load_df = df

            len_s, len_l = len(self.solar_df), len(self.load_df)
            if len_s != len_l:
                print(f"[WARNING] Data length mismatch. Solar: {len_s}, Load: {len_l}")
                min_len = min(len_s, len_l)
                self.solar_df = self.solar_df.iloc[:min_len]
                self.load_df  = self.load_df.iloc[:min_len]

            if len(self.solar_df) < self.max_steps:
                raise ValueError(f"Data must have at least {self.max_steps} rows.")

            min_rows = self.N_TILE_DAYS * self.max_steps
            if len(self.solar_df) < min_rows:
                n_reps = int(np.ceil(min_rows / len(self.solar_df)))
                self.solar_df = pd.concat([self.solar_df] * n_reps, ignore_index=True).iloc[:min_rows]
                self.load_df  = pd.concat([self.load_df]  * n_reps, ignore_index=True).iloc[:min_rows]
                print(f"[INFO] Short data tiled ×{n_reps} → {len(self.solar_df)} rows for episode diversity.")

            print(f"[OK] Data Loaded! Scenario: {scenario_name or 'default'}")

        except FileNotFoundError as e:
            print(f"[WARNING] {e} — using dummy data.")
            self.solar_df = pd.DataFrame(np.random.rand(1000, 21).astype(np.float32) * 5.0)
            self.load_df  = pd.DataFrame(np.random.rand(1000, 21).astype(np.float32) * 3.0)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step        = 0
        self.soc                 = np.full(3, 0.2, dtype=np.float32)
        self.prev_batt_power     = np.zeros(self.n_storage_units, dtype=np.float32)
        self.prev_grid_net_power = 0.0
        self.voltages            = np.ones(self.n_nodes, dtype=np.float32)
        self.voltages_min        = np.ones(self.n_nodes, dtype=np.float32)
        self.voltages_max        = np.ones(self.n_nodes, dtype=np.float32)
        self._prev_direction     = np.zeros(self.n_storage_units, dtype=np.float32)
        self._direction_streak   = np.zeros(self.n_storage_units, dtype=np.int32)
        self._peak_discharge_powers = []

        if options is not None and "start_step" in options:
            self.start_idx = options["start_step"]
            if "episode_len" in options:
                self.max_steps = options["episode_len"]
        elif self.default_start_index is not None:
            self.start_idx = self.default_start_index
        else:
            max_start = len(self.solar_df) - self.max_steps
            self.start_idx = np.random.randint(0, max_start) if max_start > 0 else 0

        self.solar_episode = self.solar_df.iloc[
            self.start_idx:self.start_idx + self.max_steps
        ].values.astype(np.float32)
        self.load_episode = self.load_df.iloc[
            self.start_idx:self.start_idx + self.max_steps
        ].values.astype(np.float32)

        # Apply solar mask (only solar_indices nodes produce solar)
        self.solar_mask = np.zeros(21, dtype=np.float32)
        self.solar_mask[self.solar_indices] = 1.0
        self.solar_episode = self.solar_episode * self.solar_mask

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # HELPER METHODS
    # ------------------------------------------------------------------

    def _max_charge_power_from_headroom(self, soc, cap, p_max):
        if soc >= 0.8:
            return 0.0
        headroom_kwh = (0.8 - soc) * cap
        return min(p_max, headroom_kwh / (0.25 * 0.95))

    def _max_discharge_power_from_soc(self, soc, cap, p_max):
        if soc <= 0.2 + self.SOC_EPS:
            return 0.0
        deliverable_kwh = (soc - 0.2) * cap * 0.95
        return min(p_max, deliverable_kwh / 0.25)

    def _compute_peak_import_target(self, hour):
        """Binary-search optimal peak-import floor for peak shaving."""
        if not (18.5 <= hour < 22.5):
            return None
        peak_end_step = min(self.max_steps, self.current_step + int((22.5 - hour) * 4))
        if peak_end_step <= self.current_step:
            return 0.0

        future_load  = np.sum(self.load_episode[self.current_step:peak_end_step], axis=1)
        future_solar = np.sum(self.solar_episode[self.current_step:peak_end_step], axis=1)
        future_net   = np.maximum(0.0, future_load - future_solar)

        if future_net.size == 0:
            return 0.0

        # Total energy deliverable from all storage (not just BESS)
        deliverable_kwh = 0.0
        for i, node_idx in enumerate(self.storage_map):
            cap = self.bess_cap if node_idx == self.bess_index else self.home_batt_cap
            deliverable_kwh += max(0.0, (self.soc[i] - 0.2) * cap * 0.95)

        if deliverable_kwh <= 0.0:
            return float(np.max(future_net))

        low, high = 0.0, float(np.max(future_net))
        for _ in range(24):
            mid = 0.5 * (low + high)
            # Total storage power needed to shave to 'mid' import level
            max_total_power = self.bess_power + 2 * self.home_batt_power
            shaved = np.minimum(np.maximum(future_net - mid, 0.0), max_total_power)
            if np.sum(shaved) * 0.25 > deliverable_kwh:
                low = mid
            else:
                high = mid
        return high

    def _estimate_daytime_solar_surplus_kwh(self, hour):
        """Estimate kWh of net solar surplus available from now until peak (18:30)."""
        if hour < 5.5:
            day_start_step = self.current_step + int((5.5 - hour) * 4)
        else:
            day_start_step = self.current_step
        day_end_step = min(self.max_steps, self.current_step + int((18.5 - max(hour, 5.5)) * 4))

        if day_end_step <= day_start_step:
            return 0.0

        future_solar = np.sum(self.solar_episode[day_start_step:day_end_step])
        future_load  = np.sum(self.load_episode[day_start_step:day_end_step])
        return max(0.0, (future_solar - future_load)) * 0.25 * 0.95

    def _is_daytime_topup_needed(self, hour):
        """
        Return True if BESS SoC is below R_PRE_PEAK_FLOOR after 16:00 and
        solar surplus alone cannot fill the gap before 18:30.
        This unlocks limited daytime grid-charging for pre-peak readiness.
        """
        if not (8.0 <= hour < 18.5):
            return False
        bess_soc = self.soc[2]
        if bess_soc >= 0.8:
            return False  # already full
        needed_kwh = (0.8 - bess_soc) * self.bess_cap
        surplus_kwh = self._estimate_daytime_solar_surplus_kwh(hour)
        return surplus_kwh < needed_kwh * 1.1  # solar forecast must comfortably exceed need

    # ------------------------------------------------------------------
    # HOME BATTERY RULE-BASED CONTROLLER
    # ------------------------------------------------------------------

    def _home_battery_rules(self, hour, solar_step, load_step):
        """
        Fully rule-based home battery controller.
        Charging priority (daytime):
          1. Home's own local solar surplus.
          2. Feeder-wide solar surplus (shared equally) if no local surplus.
          3. Off-peak grid charging only when neither solar source is available.
        Peak     → discharge to serve local load.
        No-charge window → idle.
        Returns desired powers in kW (negative=charge, positive=discharge).
        """
        powers = np.zeros(len(self.home_batt_indices), dtype=np.float32)

        if self._is_no_charge_window(hour):
            return powers

        if 5.5 <= hour < 18.5:
            feeder_surplus = max(0.0, float(np.sum(solar_step[:21])) - float(np.sum(load_step[:21])))
            n_home = len(self.home_batt_indices)
            for i, node_idx in enumerate(self.home_batt_indices):
                if self.soc[i] >= 0.8:
                    continue
                local_surplus = max(0.0, float(solar_step[node_idx]) - float(load_step[node_idx]))
                if local_surplus > 0:
                    # Priority 1: home's own solar surplus
                    available_surplus = local_surplus
                elif feeder_surplus > 0:
                    # Priority 2: share of feeder-wide surplus
                    available_surplus = feeder_surplus / n_home
                else:
                    available_surplus = 0.0

                if available_surplus > 0:
                    max_charge = self._max_charge_power_from_headroom(
                        self.soc[i], self.home_batt_cap, self.home_batt_power)
                    powers[i] = -min(available_surplus, max_charge)

        elif hour < 5.5:
            # Priority 3: off-peak grid charging when no solar surplus is available
            for i in range(len(self.home_batt_indices)):
                if self.soc[i] < 0.8:
                    max_charge = self._max_charge_power_from_headroom(
                        self.soc[i], self.home_batt_cap, self.home_batt_power)
                    powers[i] = -max_charge

        elif 18.5 <= hour < 22.5:
            # Discharge during peak to serve local demand
            for i in range(len(self.home_batt_indices)):
                max_disch = self._max_discharge_power_from_soc(
                    self.soc[i], self.home_batt_cap, self.home_batt_power)
                powers[i] = max_disch

        return powers

    # ------------------------------------------------------------------
    # BESS SOFT GUIDANCE (RL-controlled)
    # ------------------------------------------------------------------

    def _apply_soft_guidance(self, bess_action_raw, hour, net_solar_surplus, net_demand,
                              bess_soc, peak_import_target):
        """
        BESS-only soft guidance. Returns scalar action in [-1, 1].
        Priority: solar charging > off-peak grid charging > peak discharge.
        """
        if self._is_no_charge_window(hour):
            return 0.0

        def blend(guidance_val, agent_val):
            return self.GUIDANCE_ALPHA * guidance_val + (1.0 - self.GUIDANCE_ALPHA) * agent_val

        # DAYTIME (05:30–18:30): charge from solar or pre-peak top-up
        if 5.5 <= hour < 18.5:
            if net_solar_surplus > 0 and bess_soc < 0.8:
                charge_intensity = min(1.0, net_solar_surplus / self.bess_power)
                return blend(-charge_intensity, bess_action_raw)
            elif self._is_daytime_topup_needed(hour):
                # Urgency-based top-up: compute required power based on hours remaining
                hours_left = max(0.25, 18.5 - hour)
                required_power = max(
                    0.0,
                    ((0.8 - bess_soc) * self.bess_cap) / (hours_left * 0.95)
                )
                required_power = min(required_power, self.bess_power * 0.5)
                topup_frac = min(1.0, required_power / self.bess_power)
                return blend(-topup_frac, bess_action_raw)
            else:
                return blend(0.0, bess_action_raw)

        # OFF-PEAK (00:15–05:30): compute required power to reach SoC 0.8 by 05:30
        elif hour < 5.5:
            if bess_soc < 0.8:
                remaining_energy = max(0.0, (0.8 - bess_soc) * self.bess_cap)
                remaining_offpeak_hours = max(0.25, 5.5 - hour)
                required_power = min(remaining_energy / remaining_offpeak_hours, self.bess_power)
                charge_frac = min(1.0, required_power / self.bess_power)
                guided = -charge_frac
                blended = blend(guided, bess_action_raw)
                # Guidance is a floor: agent can charge MORE aggressively but never less.
                # Prevents agent positive actions from diluting the required charge rate.
                return min(blended, guided)
            return blend(0.0, bess_action_raw)

        # PEAK (18:30–22:30): demand-driven peak shaving
        elif 18.5 <= hour < 22.5:
            bess_deliverable = max(0.0, (bess_soc - 0.2) * self.bess_cap * 0.95)
            if bess_deliverable > 0:
                current_import = net_demand  # already max(0, total_load - total_solar)
                if peak_import_target is not None:
                    needed_discharge = max(0.0, current_import - peak_import_target)
                else:
                    needed_discharge = current_import
                max_discharge = self._max_discharge_power_from_soc(
                    bess_soc, self.bess_cap, self.bess_power
                )
                target_power = min(needed_discharge, max_discharge)
                return blend(min(target_power, self.bess_power) / self.bess_power, bess_action_raw)
            return blend(0.0, bess_action_raw)

        return float(bess_action_raw)

    # ------------------------------------------------------------------
    # PHYSICS CONSTRAINTS
    # ------------------------------------------------------------------

    def _apply_physics_constraints(self, action_modified, hour, net_solar_surplus,
                                    remaining_solar_surplus, total_solar, total_load):
        """
        Apply hard physical constraints and update SoC.
        Returns (remaining_demand, remaining_solar_surplus, prev_batt_power_copy).
        """
        self.node_battery_power_kw = np.zeros(self.n_nodes, dtype=np.float32)
        prev_batt_power_copy = self.prev_batt_power.copy()
        remaining_demand = max(0.0, total_load - total_solar)

        # ── PRE-ALLOCATE SOLAR SURPLUS: BESS first, then home batteries ──
        solar_allocations = {}
        if 5.5 <= hour < 18.5 and remaining_solar_surplus > 0:
            rem = remaining_solar_surplus

            bess_loop_idx = len(self.home_batt_indices)
            if self.soc[bess_loop_idx] < 0.8:
                bess_max = self._max_charge_power_from_headroom(
                    self.soc[bess_loop_idx], self.bess_cap, self.bess_power
                )
                bess_alloc = min(rem, bess_max)
                if bess_alloc > 0:
                    solar_allocations[bess_loop_idx] = bess_alloc
                    rem -= bess_alloc

            for hb_idx in range(len(self.home_batt_indices)):
                if self.soc[hb_idx] < 0.8 and rem > 0:
                    hb_max = self._max_charge_power_from_headroom(
                        self.soc[hb_idx], self.home_batt_cap, self.home_batt_power
                    )
                    hb_alloc = min(rem, hb_max)
                    if hb_alloc > 0:
                        solar_allocations[hb_idx] = hb_alloc
                        rem -= hb_alloc

        for i, node_idx in enumerate(self.storage_map):
            is_bess = (node_idx == self.bess_index)
            p_max   = self.bess_power      if is_bess else self.home_batt_power
            cap     = self.bess_cap        if is_bess else self.home_batt_cap

            # Time-aware ramp for BESS
            if is_bess:
                ramp = self.bess_ramp_peak if 18.5 <= hour < 22.5 else self.bess_ramp_offpeak
            else:
                ramp = self.home_batt_ramp

            desired_power = action_modified[i] * p_max

            # ── HARD CONSTRAINT 1: No-charge window (22:30–00:15) ────────
            # Both charging AND discharging blocked (system is idle).
            if self._is_no_charge_window(hour):
                desired_power = 0.0

            # ── HARD CONSTRAINT 2: SoC floor / ceiling ───────────────────
            if self.soc[i] <= 0.2 + self.SOC_EPS and desired_power > 0:
                desired_power = 0.0
            if self.soc[i] >= 0.8 and desired_power < 0:
                desired_power = 0.0

            # ── HARD CONSTRAINT 3: Discharge only during peak ─────────────
            # Exception: BESS may discharge during daytime for undervoltage support
            if desired_power > 0 and not (18.5 <= hour < 22.5):
                if is_bess and np.min(self.voltages[self.critical_nodes]) < 0.94:
                    pass  # Allow BESS discharge to raise sagging voltage
                else:
                    desired_power = 0.0

            # ── HARD CONSTRAINT 4: No charging during peak ────────────────
            if desired_power < 0 and 18.5 <= hour < 22.5:
                desired_power = 0.0

            # ── HARD CONSTRAINT 5: Voltage-safety on BESS discharge ───────
            if is_bess and desired_power > 0:
                if total_solar > 0.8 * remaining_demand:
                    desired_power = min(desired_power, 0.2 * p_max)

            # ── HARD CONSTRAINT 6: Cap injection to avoid overvoltage ─────
            if total_solar + max(0, desired_power) > total_load + 1.5:
                excess = total_solar + max(0, desired_power) - total_load - 1.5
                if is_bess and desired_power > 0:
                    if 5.5 <= hour < 18.5 and excess > 0.5:
                        desired_power = 0.0
                    else:
                        desired_power = max(0.0, desired_power - excess * 0.8)

            # ── HARD CONSTRAINT 7: BESS daytime solar-only charging ────────
            # Solar allocation is a CAP, not a force: agent can charge anywhere
            # from 0 to solar_allocations[i].  This gives the RL agent real
            # control (how aggressively to absorb solar) while still blocking
            # daytime grid-charging (no allocation → desired_power forced to 0).
            if is_bess and desired_power < 0 and 5.5 <= hour < 18.5:
                if i in solar_allocations and self.soc[i] < 0.8:
                    # Clamp: agent cannot charge MORE than the available surplus,
                    # but CAN choose to charge less (desired_power is negative).
                    desired_power = max(desired_power, -solar_allocations[i])
                elif self._is_daytime_topup_needed(hour):
                    # Allow full power grid top-up so BESS can reach 0.8 SoC
                    # before peak even when solar is insufficient.
                    max_topup = self._max_charge_power_from_headroom(
                        self.soc[i], cap, p_max
                    )
                    desired_power = max(desired_power, -max_topup)
                else:
                    # Solar forecast is expected to cover the need — block grid
                    # charging now to avoid paying for grid power unnecessarily.
                    desired_power = 0.0

            # ── BESS HARD SOLAR BIAS ──────────────────────────────────────
            # During daytime, if feeder has surplus > 2 kW and BESS has room,
            # force at least 50% of possible solar charging regardless of action.
            # VOLTAGE GUARD: skip the bias if node voltage is already sagging
            # (< 0.97 p.u.) — aggressive charging would worsen the sag.
            if is_bess and 5.5 <= hour < 18.5 and self.soc[i] < 0.8:
                node_v_bias = self.voltages[node_idx]
                feeder_surplus_now = max(0.0, net_solar_surplus)
                if feeder_surplus_now > 2.0 and node_v_bias >= 0.97:
                    min_charge_kw = min(
                        feeder_surplus_now,
                        self._max_charge_power_from_headroom(self.soc[i], cap, p_max)
                    )
                    # Push desired_power more negative if needed (more charging)
                    desired_power = min(desired_power, -0.5 * min_charge_kw)

            # ── HARD CONSTRAINT 8: Home battery daytime charging ──────────
            # Rules already enforce local-solar-only; just enforce SoC headroom.
            if not is_bess and desired_power < 0 and 5.5 <= hour < 18.5:
                max_charge = self._max_charge_power_from_headroom(
                    self.soc[i], self.home_batt_cap, self.home_batt_power)
                desired_power = max(desired_power, -max_charge)

            # ── HARD CONSTRAINT 9: Discharge ≤ remaining demand ───────────
            if desired_power > 0:
                desired_power = min(desired_power, remaining_demand)
                remaining_demand -= desired_power

            # ── RAMP RATE ─────────────────────────────────────────────────
            delta_p     = np.clip(desired_power - self.prev_batt_power[i], -ramp, ramp)
            final_power = np.clip(self.prev_batt_power[i] + delta_p, -p_max, p_max)

            # ── VOLTAGE-AWARE DISCHARGE THROTTLE (BESS + home batteries) ──
            # Both can cause overvoltage when exporting during low-load periods.
            if final_power > 0:
                node_v = self.voltages[node_idx]
                if node_v > 1.03:
                    voltage_factor = max(0.2, min(1.0, (1.06 - node_v) / (1.06 - 1.03)))
                    final_power = final_power * voltage_factor
                    if self.verbose:
                        label = "BESS" if is_bess else f"HOME{node_idx}"
                        print(f"[VOLT_DISCHARGE_LIMIT] Hour {hour:.1f}: {label} node "
                              f"{node_v:.4f} p.u. → discharge ×{voltage_factor:.2f}")

            # ── VOLTAGE-AWARE CHARGE LIMITING (BESS + home batteries) ──────
            # All charging draws current from the feeder and sags node voltages.
            # Threshold is time-aware:
            #   Daytime  (05:30-18:30): 0.97 p.u. — solar causes bigger swings;
            #     proactive throttle prevents daytime undervoltage violations.
            #   Off-peak (00:15-05:30): 0.95 p.u. — no solar, feeder voltages are
            #     naturally ~0.95-0.96 at night; using 0.97 would throttle to
            #     33-67% power all night and prevent BESS reaching SoC 0.8 by dawn.
            if final_power < 0:
                node_v = self.voltages[node_idx]
                volt_threshold = 0.97 if 5.5 <= hour < 18.5 else 0.95
                if node_v < volt_threshold:
                    voltage_factor = max(0.0, (node_v - 0.94) / (volt_threshold - 0.94))
                    final_power = final_power * voltage_factor
                    if self.verbose and voltage_factor < 1.0:
                        label = "BESS" if is_bess else f"HOME{node_idx}"
                        print(f"[VOLT_CHARGE_LIMIT] Hour {hour:.1f}: {label} node "
                              f"{node_v:.4f} p.u. -> charge x{voltage_factor:.2f}")


            # ── UPDATE SOC ────────────────────────────────────────────────
            eff = 0.95
            if final_power >= 0:
                self.soc[i] -= (final_power * 0.25) / eff / cap
            else:
                self.soc[i] -= (final_power * 0.25 * eff) / cap
            self.soc[i] = np.clip(self.soc[i], 0.2, 0.8)

            if 5.5 <= hour < 18.5 and final_power < 0 and remaining_solar_surplus > 0:
                remaining_solar_surplus = max(
                    0.0, remaining_solar_surplus - min(-final_power, remaining_solar_surplus)
                )

            self.node_battery_power_kw[node_idx] = final_power
            self.prev_batt_power[i] = final_power

        return remaining_demand, remaining_solar_surplus, prev_batt_power_copy

    # ------------------------------------------------------------------
    # OPENDSS
    # ------------------------------------------------------------------

    def _run_opendss(self, full_solar_profile, full_load_profile):
        """Run OpenDSS power flow. Returns True if converged."""
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

        if not step_res.converged:
            if self.verbose:
                print(f"[WARNING] OpenDSS did not converge at step {self.current_step}")
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
    # REWARD
    # ------------------------------------------------------------------

    def _compute_ramp_reward(self, final_powers, prev_powers):
        """
        Smooth-ramp reward component.

        For each storage unit:
        • Bonus  if |Δpower| < smooth threshold (RL is rewarded for being smooth).
        • Soft penalty if threshold < |Δpower| < 2× threshold.
        • Hard penalty if |Δpower| > 2× threshold (oscillation / chatter).
        • Direction bonus: sustained same direction → small additional bonus.

        Returns scalar reward contribution (unnormalised).
        """
        ramp_reward = 0.0
        for i in range(self.n_storage_units):
            is_bess   = (self.storage_map[i] == self.bess_index)
            threshold = self.R_BESS_SMOOTH_THRESHOLD if is_bess else self.R_HOME_SMOOTH_THRESHOLD
            delta     = abs(final_powers[i] - prev_powers[i])

            if delta < threshold:
                ramp_reward += self.R_SMOOTH_RAMP_REWARD * (threshold - delta)
            elif delta < 2.0 * threshold:
                excess = delta - threshold
                ramp_reward += self.R_RAMP_PENALTY_SOFT * excess
            else:
                excess = delta - 2.0 * threshold
                ramp_reward += self.R_RAMP_PENALTY_SOFT * threshold
                ramp_reward += self.R_RAMP_PENALTY_HARD * excess

            # Direction streak
            cur_dir = np.sign(final_powers[i])
            if cur_dir != 0 and cur_dir == self._prev_direction[i]:
                self._direction_streak[i] = min(self._direction_streak[i] + 1, 12)
                ramp_reward += self.R_DIRECTION_BONUS * self._direction_streak[i]
            else:
                self._direction_streak[i] = 0
            self._prev_direction[i] = cur_dir

        return ramp_reward

    # ------------------------------------------------------------------

    def _compute_reward(self, hour, net_demand, net_solar_surplus, total_solar,
                        total_load, prev_batt_power_copy, peak_import_target,
                        converged, remaining_demand):
        """Shaped reward signal. Returns (reward, info_dict)."""

        # Non-convergence short-circuit
        if not converged:
            return self.R_CONVERGENCE_PENALTY, {
                "revenue": 0.0, "cost": 0.0, "profit": 0.0,
                "grid_export_revenue": 0.0, "grid_import_cost": 0.0,
                "bess_discharge_revenue": 0.0, "bess_charge_cost": 0.0,
                "voltage_penalty": self.R_CONVERGENCE_PENALTY * self.R_NORMALIZER,
                "clean_peak_bonus": 0.0, "grid_smoothing_penalty": 0.0,
                "grid_power_change": 0.0, "current_grid_net": 0.0,
                "peak_import_target": peak_import_target,
                "total_violation": 0.0, "remaining_demand": remaining_demand,
                "pre_peak_bonus": 0.0, "ramp_reward": 0.0,
            }

        # ── TIME-OF-USE PRICING (LKR) ──────────────────────────────────
        if 5.5 <= hour < 18.5:
            buy_price, sell_price = 35, 19
        elif 18.5 <= hour < 22.5:
            buy_price, sell_price = 67, 45
        else:
            buy_price, sell_price = 21, 0

        grid_export = np.maximum(0,  self.net_injection)
        grid_import = np.maximum(0, -self.net_injection)

        grid_export_revenue  = np.sum(grid_export * sell_price)
        grid_import_cost     = np.sum(grid_import * buy_price)
        bess_power_val       = self.node_battery_power_kw[self.bess_index]
        bess_discharge_rev   = bess_power_val * sell_price if bess_power_val > 0 else 0.0
        bess_charge_cost     = (abs(bess_power_val) * buy_price
                                if bess_power_val < 0 and net_solar_surplus <= 0 else 0.0)

        revenue = grid_export_revenue
        cost    = grid_import_cost

        # ── A. DAYTIME SOLAR BONUSES ──────────────────────────────────
        daytime_solar_bonus = 0.0
        solar_waste_penalty = 0.0

        if 5.5 <= hour < 18.5:
            total_charge_power = np.sum(np.minimum(0.0, self.node_battery_power_kw))

            if net_solar_surplus > 0:
                # General solar-charging bonus
                daytime_solar_bonus += self.R_SOLAR_CHARGE_BONUS * (-total_charge_power)

                # Extra bonus for BESS specifically
                bess_charge_pw = np.minimum(0.0, self.node_battery_power_kw[self.bess_index])
                surplus_factor = min(1.0, net_solar_surplus / 20.0)
                daytime_solar_bonus += self.R_BESS_SOLAR_CHARGE * (-bess_charge_pw) * (1.0 + surplus_factor)

                # Home battery solar-charge bonus
                home_charge = np.sum([
                    np.minimum(0.0, self.node_battery_power_kw[idx])
                    for idx in self.home_batt_indices
                ])
                if home_charge < 0:
                    daytime_solar_bonus += self.R_HOME_SOLAR_CHARGE * (-home_charge)

                # Waste penalty: surplus not absorbed when batteries have room
                avg_soc = np.mean(self.soc)
                if avg_soc < 0.8:
                    solar_wasted = net_solar_surplus - (-total_charge_power)
                    if solar_wasted > 0.2:
                        solar_waste_penalty = self.R_SOLAR_WASTE_PENALTY * solar_wasted
            else:
                # Penalise BESS discharge outside peak during daytime
                bess_disc = np.maximum(0.0, self.node_battery_power_kw[self.bess_index])
                daytime_solar_bonus += self.R_BESS_DAYTIME_DISCHARGE * bess_disc

        # ── B. PRE-PEAK READINESS BONUS (05:30–18:30) ─────────────────────────
        pre_peak_bonus = 0.0
        bess_soc = self.soc[2]
        if 5.5 <= hour < 18.5:
            soc_above_floor = max(0.0, bess_soc - self.R_PRE_PEAK_FLOOR)
            pre_peak_bonus = self.R_PRE_PEAK_READY_BONUS * soc_above_floor * 100.0
        elif 18.5 <= hour < 18.75:
            # One-time bonus at peak start: strong signal for how full BESS is
            soc_progress = max(0.0, bess_soc - 0.2) / 0.6
            pre_peak_bonus = self.R_PRE_PEAK_READY_BONUS * soc_progress * 100.0

        # ── C. PEAK BONUSES ───────────────────────────────────────────
        peak_bonus          = 0.0
        peak_charge_penalty = 0.0
        clean_peak_bonus    = 0.0
        peak_smooth_bonus   = 0.0

        if 18.5 <= hour < 22.5:
            total_discharge = np.sum(np.maximum(0.0, self.node_battery_power_kw))

            if np.mean(self.soc) > 0.25:
                peak_bonus = self.R_PEAK_DISCHARGE_BONUS * total_discharge

            # Charge penalty
            for i, node_idx in enumerate(self.storage_map):
                charge_pw = np.minimum(0.0, self.node_battery_power_kw[node_idx])
                if charge_pw < 0:
                    peak_charge_penalty += self.R_PEAK_CHARGE_PENALTY * (-charge_pw)

            # Clean peak bonus (no violations)
            if (np.all(self.voltages_min[self.critical_nodes] >= 0.94) and
                    np.all(self.voltages_max[self.critical_nodes] <= 1.06)):
                clean_peak_bonus = self.R_CLEAN_PEAK_BONUS * total_discharge

            # Peak shaving smoothness bonus:
            # Reward staying close to the average discharge power so far this peak.
            self._peak_discharge_powers.append(total_discharge)
            if len(self._peak_discharge_powers) > 1:
                avg_discharge = np.mean(self._peak_discharge_powers)
                deviation     = abs(total_discharge - avg_discharge)
                # Bonus for being within 5 kW of average, penalty beyond
                if deviation < 5.0:
                    peak_smooth_bonus = self.R_PEAK_SMOOTH_BONUS * (5.0 - deviation)
                else:
                    peak_smooth_bonus = -self.R_PEAK_SMOOTH_BONUS * (deviation - 5.0)

            # Near-end-of-peak: penalise remaining SoC to encourage full discharge
            if 22.0 <= hour < 22.5:
                remaining_soc = max(0.0, self.soc[2] - 0.2)
                peak_bonus += -100.0 * remaining_soc

        # ── D. OFF-PEAK BONUSES ───────────────────────────────────────
        offpeak_bonus = 0.0
        # True off-peak: 00:15–05:30 (no-charge window ends at 00:15)
        if (not self._is_no_charge_window(hour)) and hour < 5.5:
            home_batt_soc = [self.soc[0], self.soc[1]]
            home_charge_pw = min(0.0,
                self.node_battery_power_kw[3] + self.node_battery_power_kw[5]
            )

            if np.mean(home_batt_soc) < 0.8 and home_charge_pw < 0:
                # Check if solar will be sufficient during next daytime
                steps_ahead = min(96, self.max_steps - self.current_step)
                solar_will_be_sufficient = False
                if steps_ahead > 24:
                    daylight_start = max(0, int((5.5 - hour) * 4))
                    daylight_end   = min(steps_ahead, daylight_start + 52)
                    if daylight_end > daylight_start:
                        fs_chunk = self.solar_episode[self.current_step:self.current_step + steps_ahead]
                        fl_chunk = self.load_episode[self.current_step:self.current_step + steps_ahead]
                        exp_surplus = (
                            np.sum(fs_chunk[daylight_start:daylight_end])
                            - np.sum(fl_chunk[daylight_start:daylight_end])
                        )
                        energy_needed = 2 * self.home_batt_cap * (0.8 - np.mean(home_batt_soc))
                        if exp_surplus > energy_needed * 0.9:
                            solar_will_be_sufficient = True

                if solar_will_be_sufficient:
                    offpeak_bonus = self.R_HOME_OFFPEAK_SOLAR_OK * home_charge_pw
                else:
                    offpeak_bonus = self.R_HOME_OFFPEAK_NEEDED * home_charge_pw

            # BESS off-peak charging
            bess_soc_now    = self.soc[2]
            bess_charge_pwr = self.node_battery_power_kw[self.bess_index]
            if bess_charge_pwr < 0 and bess_soc_now < 0.8:
                soc_headroom    = max(0.0, 0.8 - bess_soc_now)
                headroom_factor = 1.0 + (soc_headroom / 0.6) * 1.5
                offpeak_bonus  += -headroom_factor * self.R_OFFPEAK_BESS_HEADROOM * bess_charge_pwr

                # Deficit bonus: if solar won't suffice, reward more urgently
                expected_solar = self._estimate_daytime_solar_surplus_kwh(hour)
                if expected_solar < (0.8 - bess_soc_now) * self.bess_cap:
                    offpeak_bonus += -6.0 * bess_charge_pwr

        # ── E. DAYTIME TOP-UP BONUS ────────────────────────────────────
        daytime_topup_bonus = 0.0
        if 14.0 <= hour < 18.5:
            bess_charge_pwr = self.node_battery_power_kw[self.bess_index]
            if bess_charge_pwr < 0 and self._is_daytime_topup_needed(hour):
                daytime_topup_bonus = self.R_DAYTIME_TOPUP_BONUS * (-bess_charge_pwr)

        # ── F. SMOOTH RAMP REWARD ──────────────────────────────────────
        final_powers = np.array(
            [self.node_battery_power_kw[n] for n in self.storage_map], dtype=np.float32
        )
        ramp_reward = self._compute_ramp_reward(final_powers, prev_batt_power_copy)

        # ── G. GRID SMOOTHING PENALTY ──────────────────────────────────
        current_grid_net   = float(np.sum(self.net_injection))
        grid_power_change  = abs(current_grid_net - self.prev_grid_net_power)
        if grid_power_change > self.R_RAMP_THRESHOLD:
            excess_ramp          = grid_power_change - self.R_RAMP_THRESHOLD
            grid_smoothing_penalty = self.R_RAMP_PENALTY_RATE * excess_ramp
        else:
            grid_smoothing_penalty = 0.0
        self.prev_grid_net_power = current_grid_net

        # ── H. VOLTAGE ────────────────────────────────────────────────
        min_voltages = self.voltages_min[self.critical_nodes]
        max_voltages = self.voltages_max[self.critical_nodes]

        under_voltage   = np.maximum(0.0, 0.94 - min_voltages)
        over_voltage    = np.maximum(0.0, max_voltages - 1.06)
        total_violation = float(np.sum(over_voltage + under_voltage))
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

        ideal_nodes      = int(np.sum((min_voltages >= 0.98) & (max_voltages <= 1.02)))
        acceptable_nodes = int(np.sum((min_voltages >= 0.94) & (max_voltages <= 1.06)))
        voltage_stability_bonus = (
            self.R_IDEAL_NODE_BONUS      * ideal_nodes
            + self.R_ACCEPTABLE_NODE_BONUS * (acceptable_nodes - ideal_nodes)
        )

        # ── I. BATTERY CYCLING & SOC HEALTH ───────────────────────────
        power_changes = final_powers - prev_batt_power_copy
        cycling_cost  = (
            self.R_CYCLING_LINEAR    * float(np.sum(np.abs(power_changes)))
            + self.R_CYCLING_QUADRATIC * float(np.sum(power_changes ** 2))
        )

        soc_health_penalty = 0.0
        for i in range(len(self.soc)):
            if self.soc[i] < 0.2:
                soc_health_penalty += self.R_SOC_HEALTH_PENALTY * (0.2 - self.soc[i]) ** 2
            elif self.soc[i] > 0.8:
                soc_health_penalty += self.R_SOC_HEALTH_PENALTY * (self.soc[i] - 0.8) ** 2

        # ── J. TOTAL REWARD ───────────────────────────────────────────
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
            + peak_smooth_bonus
            + offpeak_bonus
            + daytime_topup_bonus
            + pre_peak_bonus
            + ramp_reward
            + grid_smoothing_penalty
            + cycling_cost
            + soc_health_penalty
        )

        reward = float(np.clip(reward / self.R_NORMALIZER, self.R_CLIP_LOW, self.R_CLIP_HIGH))

        if self.verbose and self.current_step % 24 == 0:
            print(f"[REWARD_DEBUG] Hour {hour:.1f}:")
            print(f"  Economic : revenue={revenue:.0f}, cost={cost:.0f}, "
                  f"net_scaled={(revenue-cost)*self.R_ECONOMIC_WEIGHT:.0f}")
            print(f"  Voltages : min={np.min(min_voltages):.4f}, max={np.max(max_voltages):.4f}, "
                  f"penalty={voltage_penalty:.0f}, bonus={voltage_stability_bonus:.0f}")
            print(f"  Battery  : cycling={cycling_cost:.0f}, soc_health={soc_health_penalty:.0f}")
            print(f"  Bonuses  : daytime={daytime_solar_bonus:.0f}, peak={peak_bonus:.0f}, "
                  f"clean_peak={clean_peak_bonus:.0f}, peak_smooth={peak_smooth_bonus:.0f}")
            print(f"  Off-peak : {offpeak_bonus:.0f}, topup={daytime_topup_bonus:.0f}, "
                  f"pre_peak={pre_peak_bonus:.0f}")
            print(f"  Ramp     : {ramp_reward:.0f}")
            print(f"  Total (after norm+clip): {reward:.4f}")

        reward_info = {
            "revenue": revenue,
            "cost": cost,
            "profit": revenue - cost,
            "grid_export_revenue": grid_export_revenue,
            "grid_import_cost": grid_import_cost,
            "bess_discharge_revenue": bess_discharge_rev,
            "bess_charge_cost": bess_charge_cost,
            "voltage_penalty": float(voltage_penalty),
            "clean_peak_bonus": float(clean_peak_bonus),
            "grid_smoothing_penalty": float(grid_smoothing_penalty),
            "grid_power_change": float(grid_power_change),
            "current_grid_net": float(current_grid_net),
            "peak_import_target": peak_import_target,
            "total_violation": total_violation,
            "remaining_demand": float(remaining_demand),
            "pre_peak_bonus": float(pre_peak_bonus),
            "ramp_reward": float(ramp_reward),
        }

        return reward, reward_info

    # ------------------------------------------------------------------
    # MAIN STEP
    # ------------------------------------------------------------------

    def step(self, action):
        """Run one environment step."""

        # 1. GET DATA
        full_solar_profile = np.zeros(self.n_nodes, dtype=np.float32)
        full_load_profile  = np.zeros(self.n_nodes, dtype=np.float32)
        full_solar_profile[:21] = self.solar_episode[self.current_step]
        full_load_profile[:21]  = self.load_episode[self.current_step]

        total_load   = float(np.sum(full_load_profile))
        total_solar  = float(np.sum(full_solar_profile))
        hour         = (self.current_step % 96) / 4.0

        net_demand              = max(0.0, total_load - total_solar)
        net_solar_surplus       = total_solar - total_load
        remaining_solar_surplus = max(0.0, net_solar_surplus)
        peak_import_target      = self._compute_peak_import_target(hour)
        bess_soc                = self.soc[2]

        # 2. HOME BATTERY RULE-BASED CONTROL
        home_powers = self._home_battery_rules(
            hour, self.solar_episode[self.current_step], self.load_episode[self.current_step]
        )

        # 3. BESS SOFT GUIDANCE (RL action[0])
        bess_guided = self._apply_soft_guidance(
            float(action[0]), hour, net_solar_surplus, net_demand, bess_soc, peak_import_target
        )

        # Combine into 3-element vector for physics (normalised to [-1, 1])
        action_modified = np.array([
            home_powers[0] / self.home_batt_power,
            home_powers[1] / self.home_batt_power,
            bess_guided
        ], dtype=np.float32)

        # 4. PHYSICS
        remaining_demand, remaining_solar_surplus, prev_batt_power_copy = \
            self._apply_physics_constraints(
                action_modified, hour, net_solar_surplus,
                remaining_solar_surplus, total_solar, total_load
            )

        # 5. OPENDSS
        converged = self._run_opendss(full_solar_profile, full_load_profile)

        # 6. REWARD
        reward, reward_info = self._compute_reward(
            hour, net_demand, net_solar_surplus, total_solar, total_load,
            prev_batt_power_copy, peak_import_target, converged, remaining_demand
        )

        # 7. TRANSITION
        self.current_step += 1
        terminated = (self.current_step >= self.max_steps)
        truncated  = False

        if terminated:
            # Terminal SoC reward: incentivise 50% end-of-day SoC
            terminal_penalty = sum(
                self.R_TERMINAL_SOC_PENALTY * abs(self.soc[i] - 0.5)
                for i in range(self.n_storage_units)
            )
            reward += terminal_penalty / self.R_NORMALIZER
            reward  = float(np.clip(reward, self.R_CLIP_LOW, self.R_CLIP_HIGH))

        obs = self._get_obs() if not terminated else self.state

        info = {
            "hour":                   hour,
            "net_demand":             net_demand,
            "remaining_demand":       remaining_demand,
            "max_voltage":            float(np.max(self.voltages_max[self.critical_nodes])),
            "min_voltage":            float(np.min(self.voltages_min[self.critical_nodes])),
            "violation":              reward_info["total_violation"],
            "solar_surplus":          net_solar_surplus,
            "total_load":             total_load,
            "total_solar":            total_solar,
            "revenue":                reward_info["revenue"],
            "cost":                   reward_info["cost"],
            "profit":                 reward_info["profit"],
            "grid_export_revenue":    reward_info["grid_export_revenue"],
            "grid_import_cost":       reward_info["grid_import_cost"],
            "bess_discharge_revenue": reward_info["bess_discharge_revenue"],
            "bess_charge_cost":       reward_info["bess_charge_cost"],
            "soc_home3":              float(self.soc[0]),
            "soc_home5":              float(self.soc[1]),
            "soc_bess":               float(self.soc[2]),
            "bess_power":             float(self.node_battery_power_kw[self.bess_index]),
            "voltage_penalty":        reward_info["voltage_penalty"],
            "clean_peak_bonus":       reward_info["clean_peak_bonus"],
            "grid_smoothing_penalty": reward_info["grid_smoothing_penalty"],
            "grid_power_change":      reward_info["grid_power_change"],
            "current_grid_net":       reward_info["current_grid_net"],
            "peak_import_target":     reward_info["peak_import_target"],
            "pre_peak_bonus":         reward_info["pre_peak_bonus"],
            "ramp_reward":            reward_info["ramp_reward"],
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        """51-value normalised observation vector."""
        if self.current_step < self.max_steps:
            # Mean solar across actual solar nodes (node 0 has no panel — never use index [0])
            common_solar_raw = np.array([
                np.mean(self.solar_episode[self.current_step][self.solar_indices])
            ], dtype=np.float32)
            load_step_raw = self.load_episode[self.current_step]
        else:
            common_solar_raw = np.array([0.0], dtype=np.float32)
            load_step_raw    = np.zeros(21, dtype=np.float32)

        common_solar_norm = common_solar_raw / self.OBS_SOLAR_MAX
        load_step_norm    = load_step_raw    / self.OBS_LOAD_MAX
        volt_norm         = (self.voltages - self.OBS_VOLT_NOM) / self.OBS_VOLT_RANGE
        soc_norm          = (self.soc - self.OBS_SOC_CENTER)    / self.OBS_SOC_RANGE

        time_angle = (self.current_step / self.max_steps) * 2 * np.pi
        day_angle  = ((self.start_idx // 96) / 365.0) * 2 * np.pi
        date_time_feats = np.array([
            np.sin(time_angle), np.cos(time_angle),
            np.sin(day_angle),  np.cos(day_angle)
        ], dtype=np.float32)

        # Feeder-wide surplus observation (52nd element)
        # Helps PPO anticipate solar availability across the whole feeder.
        if self.current_step < self.max_steps:
            solar_raw_step = self.solar_episode[self.current_step]
            load_raw_step  = self.load_episode[self.current_step]
            feeder_surplus_raw = float(np.sum(solar_raw_step) - np.sum(load_raw_step))
        else:
            feeder_surplus_raw = 0.0
        surplus_norm = np.array([feeder_surplus_raw / 100.0], dtype=np.float32)

        self.state = np.concatenate([
            common_solar_norm,
            load_step_norm,
            volt_norm,
            soc_norm,
            date_time_feats,
            surplus_norm
        ]).astype(np.float32)

        return self.state