"""Product-level checks: input schema, legal candidate set, compliance, speed, reproducibility,
physical invariants on real data (REQ-01, REQ-06, REQ-09, REQ-12, REQ-16, REQ-18)."""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from bessdss.config import Park, RegulatoryConfig, Study, SupportScheme
from bessdss.dispatch.lp import DispatchParams
from bessdss.dispatch.simulate import YearInputs, greek_local_hour, simulate_pv_only, simulate_with_battery, static_cap_mw
from bessdss.regulatory.compliance import check, summary_status
from bessdss.sizing.search import candidate_grid, legal_minimum_energy_mwh
from bessdss.study import PathSet

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "parks"


def test_study_schema_validation():
    for f in EX.glob("*.yaml"):
        Study.from_yaml(f)
    with pytest.raises(ValidationError):
        Study.model_validate({"park": {"name": "x", "lat": 38, "lon": 23, "pv_dc_kwp": 1000, "pv_ac_kw": 1000, "grid_export_limit_kw": 1000, "support": {"kind": "fit"}}})
    with pytest.raises(ValidationError):
        Study.model_validate({"park": {"name": "x", "lat": 38, "lon": 23, "pv_dc_kwp": -5, "pv_ac_kw": 1000, "grid_export_limit_kw": 1000, "support": {"kind": "merchant"}}})


def test_candidate_grid_is_legal():
    for f in EX.glob("*.yaml"):
        st = Study.from_yaml(f)
        e_min = legal_minimum_energy_mwh(st)
        grid = candidate_grid(st)
        assert grid
        for cat, p, e in grid:
            assert cat in ("11A", "11B")
            assert e >= e_min - 1e-9
            assert p <= st.park.grid_export_limit_kw / 1000.0 + 1e-9
            usable = e * (st.battery.soc_max - st.battery.soc_min)
            assert usable >= st.regulatory.min_storage_hours * st.park.installed_power_mw(st.regulatory.installed_power_basis) - 1e-9


def test_compliance_cases():
    old = Study.from_yaml(EX / "park_b_fit_500kw.yaml")
    e_ok = legal_minimum_energy_mwh(old)
    res = {c["id"]: c for c in check(old, "11A", 0.25, e_ok)}
    assert res["CHK-01"]["status"] == "fail"  # commissioned before 04.07.2019 with a live FiT
    assert res["CHK-02"]["status"] == "pass"
    assert summary_status(list(res.values())) == "fail"
    new = Study.from_yaml(EX / "park_a_fip_1mw.yaml")
    res = {c["id"]: c for c in check(new, "11A", 0.5, legal_minimum_energy_mwh(new) * 0.5)}
    assert res["CHK-01"]["status"] == "pass"
    assert res["CHK-02"]["status"] == "fail"  # below one hour of installed PV power
    res = {c["id"]: c for c in check(new, "11A", 5.0, legal_minimum_energy_mwh(new))}
    assert res["CHK-03"]["status"] == "fail"  # battery power above the export limit
    for c in res.values():
        assert c["rules"] and all(r["verified_on"] for r in c["rules"])


def _synthetic_year(dt=0.25, seed=0):
    n = int(365 * 24 / dt)
    tod = (np.arange(n) * dt) % 24
    rng = np.random.default_rng(seed)
    price = 80 - 70 * np.exp(-((tod - 10.5) / 2.5) ** 2) + 70 * np.exp(-((tod - 18) / 1.5) ** 2) + rng.normal(0, 8, n)
    pv = np.clip(np.sin((tod - 4) / 13 * np.pi), 0, None) * 1.1
    return YearInputs(dt, pv, price, np.zeros(n), local_hour=greek_local_hour(2027, n, dt))


def test_year_simulation_speed():
    park = Park(name="t", lat=39, lon=22, pv_dc_kwp=1100, pv_ac_kw=1000, grid_export_limit_kw=1000, support=SupportScheme(kind="merchant"))
    y = _synthetic_year()
    dp = DispatchParams(0.5, 1.1, 0.94, 0.94, degradation_cost_eur_mwh=10, max_cycles_per_day=1.5)
    t0 = time.perf_counter()
    simulate_with_battery(park, RegulatoryConfig(), y, 2027, dp)
    assert time.perf_counter() - t0 < 2.0


def test_physical_invariants_on_real_year():
    """Real 2026 prices (15 min) and PVGIS production of Larissa 2023: limits are never violated."""
    f_price = ROOT / "data" / "processed" / "dam_price_gr.parquet"
    f_pv = ROOT / "data" / "processed" / "pvgis_national_points.parquet"
    if not (f_price.exists() and f_pv.exists()):
        pytest.skip("real data not in this checkout")
    p = pd.read_parquet(f_price).set_index("ts_utc")["price_eur_mwh"]
    p = p[(p.index >= "2025-11-01") & (p.index < "2026-11-01")].resample("15min").ffill()
    pv_h = pd.read_parquet(f_pv)["thessaly_larissa"]
    pv_h = pv_h[pv_h.index.year == 2023]
    n = min(len(p), len(pv_h) * 4) // 96 * 96
    pv = np.repeat(pv_h.to_numpy(), 4)[:n] * 1.1
    park = Park(name="t", lat=39.64, lon=22.42, pv_dc_kwp=1100, pv_ac_kw=1000, grid_export_limit_kw=1000, support=SupportScheme(kind="merchant"))
    for cat in ("11A", "11B"):
        reg = RegulatoryConfig(station_category=cat)
        y = YearInputs(0.25, pv, p.to_numpy()[:n], np.zeros(n), local_hour=greek_local_hour(2026, n, 0.25))
        dp = DispatchParams(0.5, 1.1, 0.94, 0.94, degradation_cost_eur_mwh=10, max_cycles_per_day=1.5)
        base = simulate_pv_only(park, reg, y, 2026)
        r = simulate_with_battery(park, reg, y, 2026, dp, keep_traces=True)
        cap = static_cap_mw(park, reg, y.local_hour, True)
        assert np.all(r.net_export_mw <= cap + 1e-6)  # shared injection limit, hour by hour
        assert r.soc.min() >= -1e-6 and r.soc.max() <= dp.energy_usable_mwh + 1e-6
        assert np.all(r.c + r.d <= dp.power_mw + 1e-6)  # never more than rated power in total
        assert (r.d * 0.25).reshape(-1, 96).sum(axis=1).max() <= 1.5 * dp.energy_usable_mwh + 1e-6  # warranty cycles
        if cat == "11A":
            assert r.charged_grid_mwh == 0.0
            assert np.all(r.net_export_mw >= -1e-9)
        # sanity bound from the teams' risk list: gain <= spread x energy x cycles
        gain = r.revenue_eur - base.revenue_eur
        pr = y.price.reshape(-1, 96)
        bound = ((pr.max(axis=1) - np.minimum(pr.min(axis=1), 0)) * dp.energy_usable_mwh * 1.5).sum()
        assert gain <= bound


def test_reproducible_paths():
    ctx = SimpleNamespace(scen_names=["a", "b"], scen_weights=np.array([0.5, 0.5]), weather_years=list(range(9)), study=SimpleNamespace(life_years=15))
    a, b = PathSet.sample(ctx, 500, seed=7), PathSet.sample(ctx, 500, seed=7)
    assert np.array_equal(a.scen_idx, b.scen_idx) and np.array_equal(a.weather_idx, b.weather_idx)
    assert np.allclose(a.capture, b.capture) and np.allclose(a.model_err, b.model_err) and np.allclose(a.capex_mult, b.capex_mult)
    assert not np.array_equal(a.scen_idx, PathSet.sample(ctx, 500, seed=8).scen_idx)
