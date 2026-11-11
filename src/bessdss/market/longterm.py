"""Long-term market scenario engine for Greece.

It does NOT forecast prices. For a scenario (capacities, demand, fuel prices, export saturation)
and a historical weather year it builds one coherent year of:
    * hourly and 15-minute day-ahead prices,
    * system PV curtailment share per interval,
    * the monthly PV reference market price (needed for sliding feed-in premium contracts).

Chain of steps for one simulated year
    1. Weather year w gives the national load shape, PV and wind capacity factors of that year,
       so a sunny year means more PV, lower noon prices and more curtailment at the same time.
    2. Potential net residual load  z = load - potential PV - wind  (GW).
    3. The storage fleet (batteries by duration, pumped hydro) is dispatched by a planner LP
       against the expected price curve. This is the competitive equilibrium of price-taking
       storage and it is where the feedback "more storage, lower spread" comes from.
    4. Price = M(year, month) x Q(u | z after storage - saturation shift), with ranks u resampled
       as whole historical days (CQR-rank model, see cqr.py).
    5. PV curtailment in zero-price hours from the surplus depth; 15-minute texture from history.

Free-entry storage equilibrium (optional per scenario)
    Battery capacity is not only an input. Above the committed floor, merchants keep entering
    while a marginal battery earns its annualised cost. The capacity where the two meet is found
    by bisection for every anchor year (capacity never decreases). Cheaper batteries therefore
    lead to more batteries and lower spreads: the rebound is computed, not assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bessdss.dispatch.lp import DispatchLP, DispatchParams
from bessdss.market.cqr import QuantileSurface
from bessdss.market.fleet import FleetDispatch, StorageUnit

EF_GAS_T_PER_MWH_TH = 0.202  # tCO2 per MWh thermal of natural gas (IPCC default 56.1 kg/GJ)
CCGT_EFFICIENCY = 0.55


def gas_marginal_cost(gas_eur_mwh_th: float | np.ndarray, co2_eur_t: float | np.ndarray) -> np.ndarray:
    """Short-run marginal cost of a combined-cycle gas plant, EUR/MWh electric."""
    return (np.asarray(gas_eur_mwh_th, float) + EF_GAS_T_PER_MWH_TH * np.asarray(co2_eur_t, float)) / CCGT_EFFICIENCY


@dataclass
class MarketCalibration:
    surface: QuantileSurface
    rank_days: np.ndarray  # (n_days, 24) ranks of historical days
    rank_month: np.ndarray  # (n_days,)
    rank_weekend: np.ndarray  # (n_days,) bool
    texture_days: np.ndarray  # (n_tex, 96) normalised 15-min deviation from the hourly mean
    texture_month: np.ndarray
    weather: dict[int, pd.DataFrame]  # year -> hourly [load_shape, pv_cf, wind_cf], UTC index
    level_intercept: float  # M = intercept + slope * gas marginal cost
    level_slope: float
    level_seasonal: np.ndarray  # (12,) multiplicative monthly index of M, mean 1
    curt_a_gw: float  # PV curtailment in zero-price hours = max(0, a - b * z)  (GW)
    curt_b: float
    zero_price_eps: float = 0.5
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ persistence
    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "market_calibration.npz",
            z_grid=self.surface.z_grid, u_grid=self.surface.u_grid, q=self.surface.q,
            rank_days=self.rank_days.astype(np.float32), rank_month=self.rank_month, rank_weekend=self.rank_weekend,
            texture_days=self.texture_days.astype(np.float32), texture_month=self.texture_month,
            level_seasonal=self.level_seasonal,
        )
        wx = pd.concat({y: d for y, d in self.weather.items()}, names=["weather_year"])
        wx.astype(np.float32).to_parquet(folder / "market_weather_years.parquet")
        scal = dict(
            level_intercept=self.level_intercept, level_slope=self.level_slope,
            curt_a_gw=self.curt_a_gw, curt_b=self.curt_b, zero_price_eps=self.zero_price_eps, meta=self.meta,
        )
        (folder / "market_calibration.json").write_text(json.dumps(scal, indent=1, default=str), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path) -> MarketCalibration:
        folder = Path(folder)
        z = np.load(folder / "market_calibration.npz")
        scal = json.loads((folder / "market_calibration.json").read_text(encoding="utf-8"))
        wx = pd.read_parquet(folder / "market_weather_years.parquet")
        weather = {int(y): d.droplevel(0).astype(float) for y, d in wx.groupby(level=0)}
        return cls(
            surface=QuantileSurface(z["z_grid"], z["u_grid"], z["q"]),
            rank_days=z["rank_days"].astype(float), rank_month=z["rank_month"], rank_weekend=z["rank_weekend"],
            texture_days=z["texture_days"].astype(float), texture_month=z["texture_month"],
            weather=weather, level_intercept=scal["level_intercept"], level_slope=scal["level_slope"],
            level_seasonal=z["level_seasonal"], curt_a_gw=scal["curt_a_gw"], curt_b=scal["curt_b"],
            zero_price_eps=scal.get("zero_price_eps", 0.5), meta=scal.get("meta", {}),
        )


@dataclass
class YearDrivers:
    """Scenario values for one target year (after interpolation between anchor years)."""

    year: int
    pv_gw: float
    wind_gw: float
    load_twh: float
    flat_load_gw: float  # new baseload demand (data centres, electrolysers) added on top of the shape
    bess_gw: float
    bess_duration_h: float
    pumped_gw: float
    pumped_duration_h: float
    gas_eur_mwh: float
    co2_eur_t: float
    saturation_shift_gw: float  # how far the zero-price threshold has moved up versus the calibration period
    spread_scale: float = 1.0  # structural model error knob on rank dispersion (1 = as calibrated)


@dataclass
class Scenario:
    name: str
    label: str
    description: str
    anchors: pd.DataFrame  # index: year, columns as YearDrivers fields
    endogenous_storage_entry: bool = False
    entry: dict = field(default_factory=dict)  # parameters of the free-entry condition
    weight: float = 1.0

    def drivers(self, year: int) -> YearDrivers:
        a = self.anchors
        y = float(np.clip(year, a.index.min(), a.index.max()))
        vals = {c: float(np.interp(y, a.index.to_numpy(float), a[c].to_numpy(float))) for c in a.columns}
        return YearDrivers(year=year, **vals)


def load_scenarios(path: str | Path) -> dict[str, Scenario]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = {}
    for s in raw["scenarios"]:
        anchors = pd.DataFrame(s["anchors"]).set_index("year").sort_index()
        out[s["name"]] = Scenario(
            name=s["name"], label=s.get("label", s["name"]), description=s.get("description", ""),
            anchors=anchors, endogenous_storage_entry=bool(s.get("endogenous_storage_entry", False)),
            entry=s.get("entry", {}), weight=float(s.get("weight", 1.0)),
        )
    return out


@dataclass
class MarketYear:
    """One simulated market year on a UTC grid (365 days)."""

    year: int
    weather_year: int
    price_h: np.ndarray  # (8760,) EUR/MWh
    price_qh: np.ndarray  # (35040,) EUR/MWh
    pv_curtail_frac_h: np.ndarray  # (8760,) share of potential PV shed system-wide
    pv_cf_h: np.ndarray  # (8760,) national potential PV capacity factor (per GW installed)
    z_before_gw: np.ndarray
    z_after_gw: np.ndarray
    month_h: np.ndarray  # (8760,) month of each hour
    drivers: YearDrivers

    @property
    def pv_ref_price_monthly(self) -> np.ndarray:
        """Monthly PV-weighted market price (proxy of the ΕΤΑ of the PV technology), per hour."""
        out = np.zeros(8760)
        w = self.pv_cf_h * (1.0 - self.pv_curtail_frac_h)
        for m in range(1, 13):
            k = self.month_h == m
            out[k] = (self.price_h[k] * w[k]).sum() / max(w[k].sum(), 1e-9)
        return out

    def stats(self) -> dict:
        p = self.price_h.reshape(365, 24)
        srt = np.sort(p, axis=1)
        w = self.pv_cf_h
        return dict(
            mean_price=float(self.price_h.mean()),
            tb2=float((srt[:, -2:].mean(1) - srt[:, :2].mean(1)).mean()),
            tb4=float((srt[:, -4:].mean(1) - srt[:, :4].mean(1)).mean()),
            zero_hours=int((self.price_h <= 0.5).sum()),
            pv_capture_price=float((self.price_h * w).sum() / w.sum()),
            pv_curtailment_share=float((self.pv_curtail_frac_h * w).sum() / w.sum()),
        )


class MarketSimulator:
    def __init__(self, cal: MarketCalibration):
        self.cal = cal
        zg = cal.surface.z_grid
        self._mean_curve = cal.surface.mean_curve()
        self._fleet_cache: dict = {}
        self._zmid = 0.5 * (zg[1:] + zg[:-1])
        self._seg_price_norm = np.maximum.accumulate(0.5 * (self._mean_curve[1:] + self._mean_curve[:-1]))

    # ------------------------------------------------------------------ pieces
    def level(self, d: YearDrivers) -> np.ndarray:
        """Monthly thermal price level M for the year, (12,) EUR/MWh."""
        base = self.cal.level_intercept + self.cal.level_slope * float(gas_marginal_cost(d.gas_eur_mwh, d.co2_eur_t))
        return base * self.cal.level_seasonal

    def residual_load(self, d: YearDrivers, weather_year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wx = self.cal.weather[weather_year]
        n = 8760
        shape = wx["load_shape"].to_numpy()[:n]
        mean_load_gw = d.load_twh * 1000.0 / 8760.0
        load = shape / shape.mean() * mean_load_gw + d.flat_load_gw
        pv_cf = wx["pv_cf"].to_numpy()[:n]
        z = load - pv_cf * d.pv_gw - wx["wind_cf"].to_numpy()[:n] * d.wind_gw
        return z, pv_cf, wx.index[:n].month.to_numpy()

    def _units(self, d: YearDrivers, bess_gw: float | None = None) -> list[StorageUnit]:
        b = d.bess_gw if bess_gw is None else bess_gw
        return [
            StorageUnit("bess", max(b, 1e-6), max(b, 1e-6) * d.bess_duration_h, 0.93, 0.93),
            StorageUnit("pumped", max(d.pumped_gw, 1e-6), max(d.pumped_gw, 1e-6) * d.pumped_duration_h, 0.87, 0.87),
        ]

    def dispatch_fleet(self, z: np.ndarray, M_h: np.ndarray, d: YearDrivers, bess_gw: float | None = None) -> np.ndarray:
        """Net residual load after the storage fleet (GW), day by day with SoC carried over."""
        zg = self.cal.surface.z_grid + d.saturation_shift_gw
        key = (len(zg), round(float(zg[0]), 6))
        if key not in self._fleet_cache:
            self._fleet_cache[key] = FleetDispatch(24, 1.0, zg, 2)
        fd = self._fleet_cache[key]
        fd.b = zg
        units = self._units(d, bess_gw)
        out = np.empty_like(z)
        soc = np.zeros(2)
        for k in range(365):
            sl = slice(24 * k, 24 * k + 24)
            zz = np.clip(z[sl], zg[0] + 1e-6, None)
            seg = M_h[sl][:, None] * self._seg_price_norm[None, :]
            r = fd.solve(zz, seg, units, soc)
            net = r["net_load_gw"] - r["curtail_gw"] + (z[sl] - zz)
            out[sl] = net
            soc = r["soc_gwh"][:, -1]
        return out

    def sample_ranks(self, months_by_day: np.ndarray, weekend_by_day: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cal
        out = np.empty((len(months_by_day), 24))
        for i, (m, we) in enumerate(zip(months_by_day, weekend_by_day, strict=True)):
            near = np.isin((c.rank_month - m) % 12, (0, 1, 11))
            cand = np.where(near & (c.rank_weekend == we))[0]
            if len(cand) == 0:
                cand = np.where(near)[0]
            if len(cand) == 0:
                cand = np.arange(len(c.rank_month))
            out[i] = c.rank_days[cand[rng.integers(len(cand))]]
        return out

    def sample_texture(self, months_by_day: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cal
        if len(c.texture_days) == 0:
            return np.zeros((len(months_by_day), 96))
        out = np.empty((len(months_by_day), 96))
        for i, m in enumerate(months_by_day):
            cand = np.where(np.isin((c.texture_month - m) % 12, (0, 1, 11)))[0]
            if len(cand) == 0:
                cand = np.arange(len(c.texture_month))
            out[i] = c.texture_days[cand[rng.integers(len(cand))]]
        return out

    # ------------------------------------------------------------------ one year
    def simulate_year(
        self, d: YearDrivers, weather_year: int, seed: int = 0, bess_gw: float | None = None
    ) -> MarketYear:
        rng = np.random.default_rng([seed, d.year, weather_year])
        z, pv_cf, month_h = self.residual_load(d, weather_year)
        M12 = self.level(d)
        M_h = M12[month_h - 1]
        z_after = self.dispatch_fleet(z, M_h, d, bess_gw)
        month_d = month_h.reshape(365, 24)[:, 12]
        dow = (np.arange(365) + pd.Timestamp(f"{d.year}-01-01").dayofweek) % 7
        u = self.sample_ranks(month_d, dow >= 5, rng).reshape(-1)
        if d.spread_scale != 1.0:
            u = np.clip(0.5 + (u - 0.5) * d.spread_scale, 1e-3, 1 - 1e-3)
        z_eff = z_after - d.saturation_shift_gw
        price_h = M_h * self.cal.surface.price(z_eff, u)
        zero = price_h <= self.cal.zero_price_eps
        curt_gw = np.where(zero, np.clip(self.cal.curt_a_gw - self.cal.curt_b * z_eff, 0.0, None), 0.0)
        pot_gw = pv_cf * d.pv_gw
        frac = np.divide(curt_gw, pot_gw, out=np.zeros_like(curt_gw), where=pot_gw > 1e-6)
        frac = np.clip(frac, 0.0, 1.0)
        tex = self.sample_texture(month_d, rng).reshape(365, 24, 4) * M_h.reshape(365, 24)[:, :, None]
        qh = np.repeat(price_h, 4).reshape(365, 24, 4)
        # texture only where the hour is not in the zero-price regime, never pushing a positive hour below zero
        live = (~zero).reshape(365, 24)[:, :, None]
        qh = np.where(live, np.maximum(qh + tex, 0.0), qh)
        return MarketYear(
            year=d.year, weather_year=weather_year, price_h=price_h, price_qh=qh.reshape(-1),
            pv_curtail_frac_h=frac, pv_cf_h=pv_cf, z_before_gw=z, z_after_gw=z_after, month_h=month_h, drivers=d,
        )

    # ------------------------------------------------------------------ free-entry equilibrium
    def marginal_battery_margin(self, my: MarketYear, duration_h: float, cycle_cost: float = 5.0) -> float:
        """Annual arbitrage margin (EUR per MW) of one small price-taking standalone battery."""
        dp = DispatchParams(1.0, duration_h, 0.93, 0.93, degradation_cost_eur_mwh=cycle_cost)
        lp = DispatchLP(24, 1.0, dp)
        tot, s0 = 0.0, 0.0
        zeros, cap = np.zeros(24), np.ones(24)
        for k in range(365):
            p = my.price_h[24 * k : 24 * k + 24]
            r = lp.solve(zeros, cap, zeros, p, p, cap, soc0=s0)
            tot += float(((r.d - r.cg) * p).sum())
            s0 = float(r.soc[-1])
        return tot

    def equilibrium_bess(
        self,
        d: YearDrivers,
        weather_years: list[int],
        annual_cost_eur_per_mw: float,
        other_revenue_eur_per_mw: float = 0.0,
        k_floor: float = 0.0,
        k_max: float = 20.0,
        tol_gw: float = 0.05,
        seed: int = 0,
    ) -> tuple[float, list[tuple[float, float]]]:
        """Battery fleet (GW) at which a marginal entrant just covers its annualised cost.

        Returns the capacity and the (capacity, margin) points evaluated on the way.
        """
        trace: list[tuple[float, float]] = []

        def margin(k: float) -> float:
            m = np.mean(
                [self.marginal_battery_margin(self.simulate_year(d, w, seed, bess_gw=k), d.bess_duration_h) for w in weather_years]
            )
            trace.append((k, float(m)))
            return float(m) + other_revenue_eur_per_mw - annual_cost_eur_per_mw

        lo, hi = max(k_floor, 0.0), k_max
        if margin(lo) <= 0:
            return lo, trace
        if margin(hi) >= 0:
            return hi, trace
        while hi - lo > tol_gw:
            mid = 0.5 * (lo + hi)
            if margin(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), trace


def annualised_cost(capex_eur_per_mw: float, fom_frac: float, rate: float, life: int) -> float:
    """Capital recovery factor times CAPEX plus fixed O&M (EUR per MW per year)."""
    crf = rate * (1 + rate) ** life / ((1 + rate) ** life - 1)
    return capex_eur_per_mw * (crf + fom_frac)
