# run_opendss.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import opendssdirect as dss

DSS_PATH = Path(r"D:\UoM\FYP\VPP-Aggregator\openDSS\feeder_houses.dss")

SOLAR_INDICES = [0, 1, 2, 4, 6, 8]
LOAD_COUNT = 10
BUS_NAMES_ORDERED = [f"N{i}" for i in range(10)] + ["NBESS"]  # fixed order for RL


@dataclass
class StepResult:
    converged: bool
    bus_names: List[str]
    vmin_pu_by_bus: List[float]                 # one number per bus (min phase)
    vabc_pu_by_bus: List[Tuple[float, float, float]]  # 3-phase magnitudes (NaN if missing phase)
    total_losses_w_var: Tuple[float, float]
    voltage_violation: float


class VPPDSSRunner:
    """
    RL runner for 11 buses:
      N0..N9 houses, NBESS storage bus.

    Sign convention for setpoints:
      +kw => discharge (inject to grid)
      -kw => charge    (consume from grid)
    """

    def __init__(self, dss_file: Path, vmin_pu: float = 0.9, vmax_pu: float = 1.1):
        self.dss_file = Path(dss_file)
        self.vmin_pu = float(vmin_pu)
        self.vmax_pu = float(vmax_pu)

        if not self.dss_file.exists():
            raise FileNotFoundError(f"DSS file not found: {self.dss_file}")

        self._compiled = False

    def compile(self) -> None:
        dss.Basic.ClearAll()
        dss.Command(f'compile "{self.dss_file}"')

        # RL-friendly settings (safe even if already in DSS)
        dss.Command("set mode=snapshot")
        dss.Command("set controlmode=off")
        dss.Command("set maxiterations=50")
        dss.Command("set tolerance=0.0001")

        self._compiled = True

    def solve(self) -> bool:
        dss.Solution.Solve()
        return bool(dss.Solution.Converged())

    # ---------- update elements ----------
    def set_loads(self, loads_kw: List[float], loads_kvar: Optional[List[float]] = None) -> None:
        if len(loads_kw) != LOAD_COUNT:
            raise ValueError(f"loads_kw must be length {LOAD_COUNT}, got {len(loads_kw)}")

        if loads_kvar is not None and len(loads_kvar) != LOAD_COUNT:
            raise ValueError(f"loads_kvar must be length {LOAD_COUNT}, got {len(loads_kvar)}")

        for i in range(LOAD_COUNT):
            kw = float(loads_kw[i])
            if loads_kvar is None:
                dss.Command(f"edit load.Load{i} kw={kw}")
            else:
                kvar = float(loads_kvar[i])
                dss.Command(f"edit load.Load{i} kw={kw} kvar={kvar}")

    def set_pv_kw(self, pv_kw: Dict[int, float]) -> None:
        for idx in SOLAR_INDICES:
            p = float(pv_kw.get(idx, 0.0))
            # keep kvar=0 unless you intentionally want Volt-VAR control etc.
            dss.Command(f"edit generator.PV{idx} kw={p} kvar=0")

    def set_storage_kw(self, batt0_kw: float, batt2_kw: float, bess_kw: float) -> None:
        self._set_one_storage("Batt0", batt0_kw)
        self._set_one_storage("Batt2", batt2_kw)
        self._set_one_storage("BESS",  bess_kw)

    def _set_one_storage(self, name: str, p_kw: float) -> None:
        p_kw = float(p_kw)

        if abs(p_kw) < 1e-6:
            dss.Command(f"edit storage.{name} state=idl kw=0")
            return

        if p_kw > 0:
            # discharge: inject
            dss.Command(f"edit storage.{name} state=discharging kw={p_kw}")
        else:
            # charge: consume
            dss.Command(f"edit storage.{name} state=charging kw={abs(p_kw)}")

    # ---------- read outputs ----------
    def get_total_losses(self) -> Tuple[float, float]:
        w_var = dss.Circuit.Losses()  # [W, var]
        return float(w_var[0]), float(w_var[1])

    def get_bus_v_pu(self, bus_name: str) -> Tuple[float, Tuple[float, float, float]]:
        """
        Returns:
          (vmin_pu, (Va, Vb, Vc)) where Va/Vb/Vc are magnitudes in pu.
        If a phase is missing, it will be NaN.
        """
        import math

        dss.Circuit.SetActiveBus(bus_name)
        arr = dss.Bus.puVmagAngle()  # [mag1, ang1, mag2, ang2, mag3, ang3] (if 3 phases exist)

        mags = arr[0::2]  # extract magnitudes only

        # Pad to 3 phases
        vabc = [math.nan, math.nan, math.nan]
        for i in range(min(3, len(mags))):
            vabc[i] = float(mags[i])

        # vmin across available phases (ignore NaNs)
        avail = [v for v in vabc if not (isinstance(v, float) and math.isnan(v))]
        vmin = float(min(avail)) if avail else float("nan")

        return vmin, (vabc[0], vabc[1], vabc[2])

    def voltage_violation(self, vmin_by_bus: List[float]) -> float:
        viol = 0.0
        for v in vmin_by_bus:
            if v != v:  # NaN check
                continue
            if v < self.vmin_pu:
                viol += (self.vmin_pu - v)
            elif v > self.vmax_pu:
                viol += (v - self.vmax_pu)
        return float(viol)

    # ---------- single RL-like step ----------
    def step(
        self,
        loads_kw: List[float],
        pv_kw: Dict[int, float],
        batt_home0_kw: float,
        batt_home2_kw: float,
        bess_kw: float,
        loads_kvar: Optional[List[float]] = None,
        auto_compile: bool = True,
    ) -> StepResult:

        if auto_compile and not self._compiled:
            self.compile()

        self.set_loads(loads_kw, loads_kvar=loads_kvar)
        self.set_pv_kw(pv_kw)
        self.set_storage_kw(batt_home0_kw, batt_home2_kw, bess_kw)

        converged = self.solve()

        vmin_list: List[float] = []
        vabc_list: List[Tuple[float, float, float]] = []

        for b in BUS_NAMES_ORDERED:
            vmin, vabc = self.get_bus_v_pu(b)
            vmin_list.append(vmin)
            vabc_list.append(vabc)

        losses = self.get_total_losses()
        viol = self.voltage_violation(vmin_list)

        return StepResult(
            converged=converged,
            bus_names=BUS_NAMES_ORDERED,
            vmin_pu_by_bus=vmin_list,
            vabc_pu_by_bus=vabc_list,
            total_losses_w_var=losses,
            voltage_violation=viol,
        )


if __name__ == "__main__":
    runner = VPPDSSRunner(DSS_PATH)

    loads_kw = [5, 4, 6, 3, 5, 4, 6, 3, 4, 5]
    pv_kw = {0: 2.0, 1: 1.5, 2: 2.2, 4: 1.0, 6: 1.8, 8: 1.2}

    out = runner.step(loads_kw, pv_kw, batt_home0_kw=+1.0, batt_home2_kw=-0.5, bess_kw=+3.0)

    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"\nConverged: {out.converged}")
    print(f"\n--- Topology ---")
    print(f"Bus order: {out.bus_names}")
    print(f"\n--- System State ---")
    print(f"Vmin per bus (pu): {out.vmin_pu_by_bus}")
    print(f"Losses (W,var): {out.total_losses_w_var}")
    print(f"\n--- Metrics ---")
    print(f"Voltage Violation: {out.voltage_violation}")
    print("="*50 + "\n")



