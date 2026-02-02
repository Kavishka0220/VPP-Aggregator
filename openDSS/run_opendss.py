# run_vpp_opendss.py
# Complete Python runner + step function for your 11-node RL mapping
# Nodes: 0..9 houses (buses N0..N9), node 10 = BESS bus (NBESS)
# PV at indices: 0,1,2,4,6,8  => generator.PV0,PV1,PV2,PV4,PV6,PV8
# Storage at: Batt0, Batt2, BESS

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import opendssdirect as dss


# ----------------------------
# Configuration
# ----------------------------
dss_path = Path(r"D:\UoM\FYP\VPP-Aggregator\openDSS\feeder_houses.dss")
#dss_path = Path(r"D:\UoM\FYP\VPP-Aggregator\openDSS\main.dss")

SOLAR_INDICES = [0, 1, 2, 4, 6, 8]
HOME_BATT_INDICES = [0, 2]
BESS_BUS_NAME = "NBESS"


@dataclass
class StepResult:
    converged: bool
    buses: List[str]
    vmag_pu: List[float]
    total_losses_w_var: Tuple[float, float]
    voltage_violation: float


class VPPDSSRunner:
    """
    A small helper that compiles a DSS model once and then repeatedly:
      - edits loads
      - edits PV generators
      - edits storage power setpoints (charge/discharge)
      - solves
      - reads voltages and losses

    Sign conventions used here:
      batt_kw > 0  => discharge (inject power to grid)
      batt_kw < 0  => charge (abs(batt_kw) is charging power)
    """

    def __init__(
        self,
        dss_file: Path,
        vmin_pu: float = 0.9,
        vmax_pu: float = 1.1,
    ):
        self.dss_file = Path(dss_file)
        self.vmin_pu = float(vmin_pu)
        self.vmax_pu = float(vmax_pu)

        if not self.dss_file.exists():
            raise FileNotFoundError(f"DSS file not found: {self.dss_file}")

        self._compiled = False

    # ---------- core engine ----------
    def compile(self) -> None:
        """Clear OpenDSS state and compile the DSS model."""
        dss.Basic.ClearAll()
        dss.Command(f'compile "{self.dss_file}"')
        self._compiled = True

    def solve(self) -> bool:
        """Solve power flow and return convergence."""
        dss.Solution.Solve()
        return bool(dss.Solution.Converged())

    # ---------- update elements ----------
    def set_loads_kw(self, loads_kw: List[float]) -> None:
        """
        Update 10 house loads: load.Load0 ... load.Load9
        loads_kw must be length 10.
        """
        if len(loads_kw) != 10:
            raise ValueError(f"loads_kw must be length 10, got {len(loads_kw)}")

        for i, p in enumerate(loads_kw):
            dss.Command(f"edit load.Load{i} kw={float(p)}")

    def set_pv_kw(self, pv_kw: Dict[int, float]) -> None:
        """
        Update PV outputs for indices in SOLAR_INDICES:
          generator.PV0, PV1, PV2, PV4, PV6, PV8
        pv_kw dict should map {index: kw}.
        Missing indices will be set to 0.
        """
        for idx in SOLAR_INDICES:
            p = float(pv_kw.get(idx, 0.0))
            dss.Command(f"edit generator.PV{idx} kw={p}")

    def set_storage_kw(self, storage_kw: Dict[str, float]) -> None:
        """
        Update storage dispatch:
          storage.Batt0, storage.Batt2, storage.BESS
        storage_kw keys must be among {"Batt0","Batt2","BESS"}.

        Sign convention:
          +kw => discharge  (state=dis, kw=+)
          -kw => charge     (state=chg, kw=abs)
           0  => idle       (state=idl, kw=0)
        """
        for name, p in storage_kw.items():
            name = str(name)
            p = float(p)

            if abs(p) < 1e-6:
                dss.Command(f"edit storage.{name} state=idl kw=0")
            elif p > 0:
                dss.Command(f"edit storage.{name} state=dis kw={p}")
            else:
                dss.Command(f"edit storage.{name} state=chg kw={abs(p)}")

    # ---------- read outputs ----------
    def get_bus_voltages_pu(self) -> Tuple[List[str], List[float]]:
        """Return (bus_names, voltage_magnitudes_pu)."""
        buses = list(dss.Circuit.AllBusNames())
        vpu = list(dss.Circuit.AllBusMagPu())
        return buses, vpu

    def get_total_losses(self) -> Tuple[float, float]:
        """Return total losses (W, var)."""
        w_var = dss.Circuit.Losses()
        # Losses returns [W, var]
        return float(w_var[0]), float(w_var[1])

    def voltage_violation(self, vpu: List[float]) -> float:
        """
        Sum of per-bus violation outside [vmin_pu, vmax_pu].
        0 => no violation.
        """
        viol = 0.0
        for v in vpu:
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
        auto_compile: bool = True,
    ) -> StepResult:
        """
        One simulation step:
          1) compile if not compiled (optional)
          2) edit loads
          3) edit PV
          4) edit storage
          5) solve
          6) read voltages, losses, compute violation
        """
        if auto_compile and not self._compiled:
            self.compile()

        self.set_loads_kw(loads_kw)
        self.set_pv_kw(pv_kw)
        self.set_storage_kw({
            "Batt0": batt_home0_kw,
            "Batt2": batt_home2_kw,
            "BESS": bess_kw,
        })

        converged = self.solve()
        buses, vpu = self.get_bus_voltages_pu()
        losses = self.get_total_losses()
        viol = self.voltage_violation(vpu)

        return StepResult(
            converged=converged,
            buses=buses,
            vmag_pu=vpu,
            total_losses_w_var=losses,
            voltage_violation=viol,
        )


# ----------------------------
# Example usage (test run)
# ----------------------------
if __name__ == "__main__":
    runner = VPPDSSRunner(dss_path, vmin_pu=0.9, vmax_pu=1.1)

    # Example inputs (replace with your data for each time step)
    loads_kw = [5, 4, 6, 3, 5, 4, 6, 3, 4, 5]  # 10 houses
    pv_kw = {0: 2.0, 1: 1.5, 2: 2.2, 4: 1.0, 6: 1.8, 8: 1.2}

    # Battery dispatch setpoints (kW)
    # + => discharge, - => charge
    batt0 = +1.0
    batt2 = -0.5
    bess = +3.0

    out = runner.step(
        loads_kw=loads_kw,
        pv_kw=pv_kw,
        batt_home0_kw=batt0,
        batt_home2_kw=batt2,
        bess_kw=bess,
    )

    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)

    print(f"\nConverged: {out.converged}")

    print(f"\n--- Topology ---")
    print(f"Total Buses: {len(out.buses)}")
    print(f"Buses: {out.buses}")

    print(f"\n--- System State ---")
    print(f"Voltages (pu): {out.vmag_pu}")
    print(f"Total losses (W, var): {out.total_losses_w_var}")

    print(f"\n--- Metrics ---")
    print(f"Voltage Violation: {out.voltage_violation}")
    print("="*50 + "\n")



