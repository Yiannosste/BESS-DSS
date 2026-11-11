"""Battery as a physical system: not a perfect bucket.

Contents
  * BatterySpec: nameplate, usable SoC window, efficiencies, auxiliary load, warranty limits.
  * Degradation models (state of health, SoH, as a fraction of nameplate energy):
      - LinearDegradation: constant calendar fade per year plus fade per full equivalent cycle (FEC).
        This is the simple, well-understood baseline.
      - BlastLfpGr250Ah: the open NREL BLAST-Lite life model for a large-format LFP/graphite cell
        (calendar fade ~ t^0.526 with temperature and SoC dependence, cycle fade ~ EFC^0.828 with
        depth of discharge and average C-rate), with the published coefficients.
  * rainflow_cycles: half-cycle depth extraction from a SoC trace (for depth-dependent models).

Units: energy MWh, power MW, temperature degC, time in years unless stated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BatterySpec:
    power_mw: float
    energy_mwh: float  # nameplate (beginning of life) energy
    soc_min: float = 0.05
    soc_max: float = 0.95
    eta_charge: float = 0.93  # one-way, battery bus to cell including PCS and transformer
    eta_discharge: float = 0.93
    self_discharge_per_h: float = 0.0
    aux_kw_per_mwh: float = 1.5  # average auxiliary (thermal management, BMS) load per nameplate MWh
    coupling: str = "ac"
    max_cycles_per_year: float | None = 365.0  # warranty full equivalent cycles per year
    eol_soh: float = 0.6  # end-of-life / end-of-warranty state of health

    @property
    def duration_h(self) -> float:
        return self.energy_mwh / self.power_mw if self.power_mw > 0 else 0.0

    @property
    def round_trip_efficiency(self) -> float:
        return self.eta_charge * self.eta_discharge

    def usable_energy(self, soh: float = 1.0) -> float:
        """Usable energy at a given state of health (MWh)."""
        return self.energy_mwh * soh * (self.soc_max - self.soc_min)

    def aux_energy_mwh_per_year(self) -> float:
        return self.aux_kw_per_mwh * self.energy_mwh * 8760.0 / 1000.0


# --------------------------------------------------------------------------- degradation models
@dataclass(frozen=True)
class YearUsage:
    """Summary of one year of operation, the interface between dispatch and degradation."""

    fec: float  # full equivalent cycles in the year, on NAMEPLATE energy (sum |dSOC| / 2)
    mean_soc: float = 0.5  # time-average SoC as a fraction of nameplate
    mean_dod: float = 0.8  # throughput-weighted cycle depth (fraction of nameplate)
    mean_c_rate: float = 0.3  # throughput-weighted C-rate (1/h)
    cell_temp_c: float = 25.0  # average cell temperature (liquid cooled containers hold about 25 degC)


class LinearDegradation:
    """SoH falls by a fixed amount per calendar year and per full equivalent cycle."""

    name = "linear"

    def __init__(self, calendar_fade_per_year: float = 0.010, cycle_fade_per_fec: float = 0.00004):
        self.kcal = calendar_fade_per_year
        self.kcyc = cycle_fade_per_fec

    def simulate(self, usage: list[YearUsage]) -> np.ndarray:
        """Return SoH at the START of each year (length len(usage)+1)."""
        soh = [1.0]
        for u in usage:
            soh.append(max(0.0, soh[-1] - self.kcal - self.kcyc * u.fec))
        return np.array(soh)

    def marginal_fade_per_fec(self, usage: list[YearUsage], year: int) -> float:
        return self.kcyc


class BlastLfpGr250Ah:
    """Semi-empirical LFP/graphite life model of NREL BLAST-Lite (`Lfp_Gr_250AhPrismatic`).

    Source: github.com/NREL/BLAST-Lite, file blast/models/lfp_gr_250AhPrismatic_2019.py (BSD-3),
    fitted on ageing data of a large-format (>250 Ah) prismatic LFP-Gr cell, Gasper et al.,
    J. Energy Storage 2023, doi 10.1016/j.est.2023.109042. Coefficients are copied from that file.

        q = 1 - qLoss_t - qLoss_EFC
        qLoss_t   = kcal * t_days^pcal,   kcal = |p1| exp(p2/T) exp(p3 Ua(SOC)/T)
        qLoss_EFC = kcyc * EFC^pcyc,      kcyc = (p4 + p5 DOD + p6 Crate) (exp(p7/T) + exp(-p8/T))

    T in kelvin. Crate is the TIME-AVERAGED |dSOC/dt| in 1/h including rests (about 0.083 for one
    full cycle per day), not the nameplate C-rate. Valid for 10 to 45 degC and DOD 0.8 to 1; outside
    that range it extrapolates. It is a CELL model: the authors advise 20 to 30 % shorter life at
    pack level, which is applied here as `pack_life_factor` (time and throughput run faster).

    Conditions change from year to year, so fade is accumulated in the "virtual time" domain:
    y_new = k * ((y/k)^(1/p) + dx)^p, which is the exact integral of the incremental form in BLAST.
    """

    name = "blast_lfp_gr_250ah"
    P1, P2, P3, PCAL = 8.37e4, -5.21e3, -3.56e3, 0.526
    P4, P5, P6, P7, P8, PCYC = 4.38e-8, 1.55e-8, 1.68e-7, 2.19e3, 1.55e5, 0.828

    def __init__(self, pack_life_factor: float = 0.775, degradation_scalar: float = 1.0):
        self.pack_life_factor = pack_life_factor  # 1.0 = pure cell model
        self.scalar = degradation_scalar  # sensitivity knob, 1.0 = as published

    @staticmethod
    def anode_potential(soc: float) -> float:
        xa = 8.5e-3 + soc * (0.78 - 8.5e-3)
        return float(
            0.6379
            + 0.5416 * np.exp(-305.5309 * xa)
            + 0.044 * np.tanh(-(xa - 0.1958) / 0.1088)
            - 0.1978 * np.tanh((xa - 1.0571) / 0.0854)
            - 0.6875 * np.tanh((xa + 0.0117) / 0.0529)
            - 0.0175 * np.tanh((xa - 0.5692) / 0.0875)
        )

    def _kcal(self, temp_c: float, soc: float) -> float:
        T = temp_c + 273.15
        return abs(self.P1) * np.exp(self.P2 / T) * np.exp(self.P3 * self.anode_potential(soc) / T)

    def _kcyc(self, temp_c: float, dod: float, crate: float) -> float:
        T = temp_c + 273.15
        return (self.P4 + self.P5 * dod + self.P6 * crate) * (np.exp(self.P7 / T) + np.exp(-self.P8 / T))

    @staticmethod
    def _advance(y: float, k: float, p: float, dx: float) -> float:
        if k <= 0 or dx <= 0:
            return y
        return k * ((y / k) ** (1.0 / p) + dx) ** p

    def simulate(self, usage: list[YearUsage]) -> np.ndarray:
        """SoH at the START of each year (length len(usage)+1). usage.fec is in nameplate full cycles."""
        qt, qe = 0.0, 0.0
        soh = [1.0]
        f = 1.0 / self.pack_life_factor
        for u in usage:
            crate = 2.0 * u.fec / 8760.0  # time-averaged |dSOC/dt|, 1/h
            qt = self._advance(qt, self._kcal(u.cell_temp_c, u.mean_soc), self.PCAL, 365.0 * f)
            qe = self._advance(qe, self._kcyc(u.cell_temp_c, u.mean_dod, crate), self.PCYC, u.fec * f)
            soh.append(max(0.0, 1.0 - self.scalar * (qt + qe)))
        return np.array(soh)

    def marginal_fade_per_fec(self, usage: list[YearUsage], year: int, eps: float = 1.0) -> float:
        """Extra end-of-horizon fade caused by one more full cycle in a given year (numerical)."""
        base = self.simulate(usage)[-1]
        bumped = list(usage)
        u = bumped[year]
        bumped[year] = YearUsage(u.fec + eps, u.mean_soc, u.mean_dod, u.mean_c_rate, u.cell_temp_c)
        return float(base - self.simulate(bumped)[-1]) / eps


def rainflow_cycles(soc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rainflow counting (ASTM E1049 three-point method) on a SoC trace.

    Returns (depths, weights): cycle depths as SoC fractions and weights 1.0 (full) or 0.5 (half).
    """
    x = np.asarray(soc, float)
    # keep turning points only
    d = np.diff(x)
    keep = np.ones(len(x), bool)
    keep[1:-1] = (d[:-1] * d[1:]) < 0
    tp = x[keep]
    stack: list[float] = []
    depths: list[float] = []
    weights: list[float] = []
    for v in tp:
        stack.append(v)
        while len(stack) >= 3:
            a, b, c = stack[-3], stack[-2], stack[-1]
            r1, r2 = abs(b - a), abs(c - b)
            if r2 < r1:
                break
            if len(stack) == 3:
                depths.append(r1)
                weights.append(0.5)
                stack.pop(0)
            else:
                depths.append(r1)
                weights.append(1.0)
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for i in range(len(stack) - 1):
        depths.append(abs(stack[i + 1] - stack[i]))
        weights.append(0.5)
    return np.array(depths), np.array(weights)


def usage_from_trace(soc_frac: np.ndarray, power_frac: np.ndarray, dt_h: float, spec: BatterySpec) -> YearUsage:
    """Summarise a simulated year (SoC and power as fractions of nameplate) for the degradation model."""
    fec = np.clip(power_frac, 0, None).sum() * dt_h  # discharged energy in units of nameplate energy
    depths, weights = rainflow_cycles(soc_frac)
    if len(depths) and (depths * weights).sum() > 0:
        mean_dod = float((depths**2 * weights).sum() / (depths * weights).sum())
    else:
        mean_dod = 0.0
    thr = np.abs(power_frac)
    mean_c = float((thr**2).sum() / thr.sum()) if thr.sum() > 0 else 0.0
    return YearUsage(fec=float(fec), mean_soc=float(np.mean(soc_frac)), mean_dod=mean_dod, mean_c_rate=mean_c)
