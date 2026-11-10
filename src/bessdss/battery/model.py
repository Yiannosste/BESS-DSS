"""Battery as a physical system: not a perfect bucket.

Contents
  * BatterySpec: nameplate, usable SoC window, efficiencies, auxiliary load, warranty limits.
  * Degradation models (state of health, SoH, as a fraction of nameplate energy):
      - LinearDegradation: constant calendar fade per year plus fade per full equivalent cycle (FEC).
        This is the simple, well-understood baseline.
      - SemiEmpiricalLFP: calendar fade ~ t^0.5 with Arrhenius temperature and SoC dependence,
        cycle fade ~ FEC^0.5 scaled by depth of discharge and C-rate. Functional forms follow
        Naumann et al. (2018, 2020) for an LFP/graphite cell. Coefficients are loaded from the
        assumptions register so that every number has a source.
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

    fec: float  # full equivalent cycles in the year (discharged energy / nameplate usable energy)
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


class SemiEmpiricalLFP:
    """Square-root-of-time calendar fade and square-root-of-throughput cycle fade for LFP.

    Calendar:  q_cal(t) = k_cal(T, SoC) * sqrt(t_seconds)
               k_cal = k_ref * exp(-Ea/R * (1/T - 1/T_ref)) * (c * (SoC - 0.5)^3 + d)
    Cycle:     q_cyc(FEC) = k_cyc(DoD, C) * sqrt(FEC * 100) / 100 ... see _kcyc
    Total capacity loss = q_cal + q_cyc (superposition, as in Naumann et al. 2020).

    Because both terms are concave in their driver, fade must be accumulated in the "virtual
    time / virtual throughput" domain when conditions change from year to year. That is what
    simulate() does.

    Default coefficients are the published fits for a Sony/Murata US26650FTC1 LFP/graphite cell
    (Naumann et al. 2018 J. Energy Storage 17:153; Naumann et al. 2020 J. Power Sources 451:227666).
    They are replaced at run time by the values in registers/assumptions.yaml when present.
    """

    name = "semi_empirical_lfp"
    R = 8.314462618  # J/mol/K

    def __init__(
        self,
        k_ref: float = 1.2571e-5,  # 1/sqrt(s) fraction per sqrt(second) at T_ref, SoC reference shape
        ea_j_per_mol: float = 17126.0,
        t_ref_k: float = 298.15,
        c_soc: float = 2.8575,
        d_soc: float = 0.60225,
        a_crate: float = 0.0630,
        b_crate: float = 0.0971,
        c_dod: float = 4.0253,
        d_dod: float = 1.0923,
        cycle_scale: float = 1.0,
        calendar_scale: float = 1.0,
    ):
        self.k_ref, self.ea, self.t_ref = k_ref, ea_j_per_mol, t_ref_k
        self.c_soc, self.d_soc = c_soc, d_soc
        self.a_c, self.b_c, self.c_d, self.d_d = a_crate, b_crate, c_dod, d_dod
        self.cycle_scale, self.calendar_scale = cycle_scale, calendar_scale

    def _kcal(self, temp_c: float, soc: float) -> float:
        T = temp_c + 273.15
        arr = np.exp(-self.ea / self.R * (1.0 / T - 1.0 / self.t_ref))
        return self.calendar_scale * self.k_ref * arr * (self.c_soc * (soc - 0.5) ** 3 + self.d_soc)

    def _kcyc(self, dod: float, c_rate: float) -> float:
        # Naumann 2020: Q_loss,cyc [%] = (a*C + b) * (c*(DoD-0.6)^3 + d) * sqrt(FEC)
        return self.cycle_scale * (self.a_c * c_rate + self.b_c) * (self.c_d * (dod - 0.6) ** 3 + self.d_d) / 100.0

    def simulate(self, usage: list[YearUsage]) -> np.ndarray:
        sec_per_year = 365.25 * 24 * 3600.0
        qcal, qcyc = 0.0, 0.0
        soh = [1.0]
        for u in usage:
            kc = self._kcal(u.cell_temp_c, u.mean_soc)
            t_virtual = (qcal / kc) ** 2 if kc > 0 else 0.0
            qcal = kc * np.sqrt(t_virtual + sec_per_year)
            ky = self._kcyc(u.mean_dod, u.mean_c_rate)
            fec_virtual = (qcyc / ky) ** 2 if ky > 0 else 0.0
            qcyc = ky * np.sqrt(fec_virtual + u.fec)
            soh.append(max(0.0, 1.0 - qcal - qcyc))
        return np.array(soh)

    def marginal_fade_per_fec(self, usage: list[YearUsage], year: int, eps: float = 1.0) -> float:
        """Extra end-of-horizon fade caused by one more FEC in a given year (numerical)."""
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
    discharged = np.clip(power_frac, 0, None).sum() * dt_h  # in units of nameplate energy per 1 C
    window = spec.soc_max - spec.soc_min
    fec = discharged / window if window > 0 else 0.0
    depths, weights = rainflow_cycles(soc_frac)
    if len(depths) and (depths * weights).sum() > 0:
        mean_dod = float((depths**2 * weights).sum() / (depths * weights).sum())
    else:
        mean_dod = 0.0
    thr = np.abs(power_frac)
    mean_c = float((thr**2).sum() / thr.sum()) if thr.sum() > 0 else 0.0
    return YearUsage(fec=float(fec), mean_soc=float(np.mean(soc_frac)), mean_dod=mean_dod, mean_c_rate=mean_c)
