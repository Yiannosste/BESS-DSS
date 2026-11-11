"""Scenario library: resolves the endogenous storage path of each scenario and serves simulated
market years with an on-disk cache, so that the sizing loop never recomputes the market."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from bessdss.market.longterm import (
    MarketCalibration,
    MarketSimulator,
    MarketYear,
    Scenario,
    YearDrivers,
    annualised_cost,
    load_scenarios,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAL_DIR = ROOT / "data" / "processed" / "market_calibration"
DEFAULT_SCENARIOS = ROOT / "registers" / "scenarios.yaml"
DEFAULT_CACHE = ROOT / "data" / "cache" / "market"
EQUILIBRIUM_YEARS = (2027, 2029, 2031, 2034, 2038, 2042)
EQUILIBRIUM_WEATHER = (2017, 2021)


class ScenarioLibrary:
    def __init__(
        self,
        cal: MarketCalibration | None = None,
        scenarios: dict[str, Scenario] | None = None,
        cache_dir: str | Path | None = None,
        seed: int = 0,
    ):
        self.cal = cal or MarketCalibration.load(DEFAULT_CAL_DIR)
        self.scenarios = scenarios or load_scenarios(DEFAULT_SCENARIOS)
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from bessdss import assumptions as A

        self.sim = MarketSimulator(self.cal, fleet_arbitrage_share=A.central("fleet_arbitrage_share"))
        self.seed = seed
        self._paths: dict[str, dict] = {}

    # ------------------------------------------------------------------ helpers
    def _scen_key(self, scen: Scenario) -> str:
        blob = json.dumps(
            dict(a=scen.anchors.reset_index().to_dict("list"), e=scen.entry, endo=scen.endogenous_storage_entry,
                 cal=self.cal.meta, q=float(self.cal.surface.q.sum()), seed=self.seed, share=self.sim.fleet_arbitrage_share),
            sort_keys=True, default=str,
        )
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    @property
    def weather_years(self) -> list[int]:
        """Weather years usable together with PVGIS park data (PVGIS ends in 2023)."""
        return sorted(y for y in self.cal.weather if y <= 2023)

    # ------------------------------------------------------------------ endogenous storage
    def storage_path(self, name: str) -> dict:
        """Battery fleet (GW) per equilibrium year, with the margin at the solution."""
        if name in self._paths:
            return self._paths[name]
        scen = self.scenarios[name]
        f = self.cache_dir / f"storage_path_{name}_{self._scen_key(scen)}.json"
        if f.exists():
            self._paths[name] = json.loads(f.read_text())
            return self._paths[name]
        out = {"years": [], "bess_gw": [], "floor_gw": [], "annual_cost_eur_per_mw": [], "margin_at_floor": [], "margin_at_solution": []}
        k_prev, y_prev = None, None
        for y in EQUILIBRIUM_YEARS:
            d = scen.drivers(y)
            floor = d.bess_gw if k_prev is None else max(d.bess_gw, k_prev)
            if not scen.endogenous_storage_entry:
                k, m_floor, m_sol, cost = floor, None, None, None
            else:
                e = scen.entry
                capex_mw = e["capex_eur_per_kwh"] * (1 - e["capex_decline"]) ** (y - 2026) * 1000.0 * d.bess_duration_h
                cost = annualised_cost(capex_mw, e["fom_frac"], e["wacc"], e["life"])
                pts = np.array(e["other_revenue_eur_per_mw"], float)
                other = float(np.interp(y, pts[:, 0], pts[:, 1]))
                k_max = floor + e["max_build_gw_per_year"] * ((y - y_prev) if y_prev else 1.0)
                k, trace = self.sim.equilibrium_bess(
                    d, list(EQUILIBRIUM_WEATHER), cost, other, k_floor=floor, k_max=k_max, tol_gw=0.1, seed=self.seed
                )
                m_floor = trace[0][1]
                m_sol = min(trace, key=lambda t: abs(t[0] - k))[1]
            out["years"].append(y)
            out["bess_gw"].append(round(float(k), 3))
            out["floor_gw"].append(round(float(d.bess_gw), 3))
            out["annual_cost_eur_per_mw"].append(cost)
            out["margin_at_floor"].append(m_floor)
            out["margin_at_solution"].append(m_sol)
            k_prev, y_prev = k, y
        f.write_text(json.dumps(out, indent=1))
        self._paths[name] = out
        return out

    def drivers(self, name: str, year: int) -> YearDrivers:
        scen = self.scenarios[name]
        d = scen.drivers(year)
        p = self.storage_path(name)
        k = float(np.interp(year, p["years"], p["bess_gw"]))
        return replace(d, bess_gw=max(d.bess_gw, k))

    # ------------------------------------------------------------------ market years
    def market_year(self, name: str, year: int, weather_year: int) -> MarketYear:
        scen = self.scenarios[name]
        f = self.cache_dir / f"my_{name}_{self._scen_key(scen)}_{year}_{weather_year}.npz"
        d = self.drivers(name, year)
        if f.exists():
            z = np.load(f)
            return MarketYear(
                year=year, weather_year=weather_year, price_h=z["price_h"].astype(float), price_qh=z["price_qh"].astype(float),
                pv_curtail_frac_h=z["curt"].astype(float), pv_cf_h=z["pv_cf"].astype(float), z_before_gw=z["zb"].astype(float),
                z_after_gw=z["za"].astype(float), month_h=z["month"], drivers=d,
            )
        my = self.sim.simulate_year(d, weather_year, seed=self.seed)
        np.savez_compressed(
            f, price_h=my.price_h.astype(np.float32), price_qh=my.price_qh.astype(np.float32),
            curt=my.pv_curtail_frac_h.astype(np.float32), pv_cf=my.pv_cf_h.astype(np.float32),
            zb=my.z_before_gw.astype(np.float32), za=my.z_after_gw.astype(np.float32), month=my.month_h,
        )
        return my

    def summary(self, years: tuple[int, ...] = (2027, 2030, 2035, 2040), weather_years: list[int] | None = None) -> list[dict]:
        """Headline market statistics per scenario and year, averaged over weather years."""
        rows = []
        wys = weather_years or self.weather_years
        for name in self.scenarios:
            for y in years:
                st = [self.market_year(name, y, w).stats() for w in wys]
                d = self.drivers(name, y)
                row = {"scenario": name, "year": y, "bess_gw": round(d.bess_gw, 2), "pv_gw": round(d.pv_gw, 1)}
                row.update({k: round(float(np.mean([s[k] for s in st])), 3 if "share" in k else 1) for k in st[0]})
                rows.append(row)
        return rows


def drivers_dict(d: YearDrivers) -> dict:
    return asdict(d)
