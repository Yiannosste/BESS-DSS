"""Experiments behind the innovation log. Each one compares an idea with a simple baseline and
writes its numbers to docs/validation/experiments.json. Run after the market cache exists:

    python scripts/run_experiments.py            # all
    python scripts/run_experiments.py duals      # one of: duals, cycle_cost, degradation, rebound, forecast, replay
"""
import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "validation" / "experiments.json"

from bessdss.config import Park, RegulatoryConfig, Study, SupportScheme  # noqa: E402
from bessdss.dispatch.lp import DispatchParams  # noqa: E402
from bessdss.dispatch.simulate import (  # noqa: E402
    YearInputs,
    greek_local_hour,
    simulate_pv_only,
    simulate_with_battery,
)
from bessdss.study import PathSet, StudyContext, economics, marginal_cycle_cost, run_design  # noqa: E402

WYS = [2018, 2022, 2023]


def _ctx(park="park_c_merchant_5mw"):
    st = Study.from_yaml(ROOT / "examples" / "parks" / f"{park}.yaml")
    ctx = StudyContext.build(st)
    return st, ctx, PathSet.sample(ctx, 2000)


def exp_duals():
    """INN-03: are the LP duals the true marginal value of power and energy? Compare with finite differences
    obtained by re-simulating the whole design (one scenario, three weather years, hourly)."""
    st, ctx, _ = _ctx()
    cat, p, e = "11A", 2.1, 8.3
    kw = dict(resolution="hourly", weather_years=WYS, scen_names=["central"])
    t0 = time.time()
    d0 = run_design(ctx, cat, p, e, **kw)
    t_one = time.time() - t0
    window = st.battery.soc_max - st.battery.soc_min
    hp, he = 0.02, 0.05
    dP = (run_design(ctx, cat, p + hp, e, **kw).gross.mean(2)[0] - run_design(ctx, cat, p - hp, e, **kw).gross.mean(2)[0]) / (2 * hp)
    dE = (run_design(ctx, cat, p, e + he, **kw).gross.mean(2)[0] - run_design(ctx, cat, p, e - he, **kw).gross.mean(2)[0]) / (2 * he)
    dual_p = d0.mv_power[0]
    soh_rep = np.array([d0.soh[0, ctx.years.index(y)] for y in ctx.rep_years])
    dual_e = d0.mv_energy[0] * window * soh_rep
    return dict(
        design=[cat, p, e], rep_years=ctx.rep_years,
        marginal_power_fd=dP.round(0).tolist(), marginal_power_dual=dual_p.round(0).tolist(),
        marginal_energy_fd=dE.round(0).tolist(), marginal_energy_dual=dual_e.round(0).tolist(),
        rel_err_power=float(np.abs(dual_p - dP).sum() / np.abs(dP).sum()), rel_err_energy=float(np.abs(dual_e - dE).sum() / np.abs(dE).sum()),
        seconds_per_design=round(t_one, 1), note="finite differences need 4 extra simulations, the duals need none",
    )


def exp_cycle_cost():
    """INN-04: which degradation cost per MWh should the dispatch see? Baseline: the marginal fade value from
    the life model. Alternative: search the scalar that maximises expected NPV (a constant shadow price)."""
    st, ctx, paths = _ctx()
    cat, p, e = "11A", 2.1, 5.6
    base = marginal_cycle_cost(st, ctx.fin)
    rows = []
    for c in [0.0, 0.5 * base, base, 2 * base, 4 * base, 8 * base]:
        d = run_design(ctx, cat, p, e, resolution="hourly", weather_years=WYS, cycle_cost_eur_mwh=c)
        m = economics(ctx, d, paths).metrics()
        rows.append(dict(cycle_cost=round(c, 1), npv_mean=round(m["npv_mean"]), fec_year1=round(float(d.fec[:, 0].mean())), soh_end=round(float(d.soh[:, -1].mean()), 3),
                         revenue_y1=round(m["revenue_y1_p50"])))
    best = max(rows, key=lambda r: r["npv_mean"])
    return dict(design=[cat, p, e], baseline_cycle_cost=round(base, 1), rows=rows, best_cycle_cost=best["cycle_cost"],
                npv_gain_vs_baseline=best["npv_mean"] - next(r for r in rows if r["cycle_cost"] == round(base, 1))["npv_mean"])


def exp_degradation():
    """How detailed must degradation be? Linear model versus the BLAST LFP model: SoH path and NPV."""
    st, ctx, paths = _ctx()
    cat, p, e = "11A", 2.1, 5.6
    out = {}
    for name in ("blast_lfp_gr_250ah", "linear"):
        st2 = st.model_copy(update={"battery": st.battery.model_copy(update={"degradation_model": name})})
        ctx2 = replace(ctx, study=st2)
        d = run_design(ctx2, cat, p, e, resolution="hourly", weather_years=WYS)
        m = economics(ctx2, d, paths).metrics()
        out[name] = dict(npv_mean=round(m["npv_mean"]), soh_year5=round(float(d.soh[:, 5].mean()), 3), soh_year10=round(float(d.soh[:, 10].mean()), 3),
                         soh_end=round(float(d.soh[:, -1].mean()), 3), revenue_last_year=round(float(economics(ctx2, d, paths).revenue_real[:, -1].mean())))
    out["npv_difference"] = out["linear"]["npv_mean"] - out["blast_lfp_gr_250ah"]["npv_mean"]
    return out


def exp_rebound():
    """INN-01: how much does endogenous storage entry change the market and the value of the park's battery?
    Central scenario with the committed floor only (exogenous) versus free entry above the floor."""
    from bessdss.market.library import ScenarioLibrary
    from bessdss.market.longterm import Scenario

    st, ctx, _ = _ctx()
    lib = ctx.lib
    sc = lib.scenarios["central"]
    lib.scenarios["central_floor_only"] = Scenario(name="central_floor_only", label="floor", description="", anchors=sc.anchors, endogenous_storage_entry=False, weight=0.0)
    rows = []
    for y in (2027, 2030, 2035, 2041):
        r = {"year": y}
        for name in ("central_floor_only", "central"):
            stt = [lib.market_year(name, y, w).stats() for w in WYS]
            d = lib.drivers(name, y)
            r[name] = dict(bess_gw=round(d.bess_gw, 2), tb4=round(float(np.mean([s["tb4"] for s in stt])), 1),
                           zero_hours=round(float(np.mean([s["zero_hours"] for s in stt]))), capture=round(float(np.mean([s["pv_capture_price"] for s in stt])), 1))
        rows.append(r)
    cat, p, e = "11A", 2.1, 5.6
    val = {}
    for name in ("central_floor_only", "central"):
        d = run_design(ctx, cat, p, e, resolution="hourly", weather_years=WYS, scen_names=[name])
        val[name] = [round(float(x)) for x in d.gross.mean(2)[0]]
    assert isinstance(lib, ScenarioLibrary)
    return dict(market=rows, battery_gross_value_by_rep_year=val, rep_years=ctx.rep_years, storage_path=lib.storage_path("central"))


def _real_year(dt=1.0):
    p = pd.read_parquet(ROOT / "data/processed/dam_price_gr.parquet").set_index("ts_utc")["price_eur_mwh"]
    p = p[(p.index >= "2025-11-01") & (p.index < "2026-11-01")]
    p = p.resample("1h").mean() if dt == 1.0 else p.resample("15min").ffill()
    return p


def exp_forecast():
    """INN-05: how much of the perfect-foresight value survives a real day-ahead forecast?
    Real hourly prices Nov 2025 to Oct 2026. Forecasters: yesterday (naive), same weekday last week, and a
    ridge regression on lagged prices and ADMIE's day-ahead load and RES forecasts (LEAR-like)."""
    from sklearn.linear_model import RidgeCV

    p = _real_year(1.0)
    idx = p.index
    days = len(p) // 24
    P = p.to_numpy()[: days * 24].reshape(days, 24)
    full = pd.read_parquet(ROOT / "data/processed/dam_price_gr.parquet").set_index("ts_utc")["price_eur_mwh"].resample("1h").mean()
    hist = full[(full.index >= "2025-06-01") & (full.index < idx[0])]
    H = hist.to_numpy()[-(len(hist) // 24 * 24):].reshape(-1, 24)
    allP = np.vstack([H, P])
    n0 = len(H)
    try:
        fc = pd.read_parquet(ROOT / "data/processed/res_forecast_gr.parquet").set_index("ts_utc")
        cols = ["load_da_fc_mw", "pv_da_fc_mw", "wind_da_fc_mw"]
        start = idx[0] - pd.Timedelta(days=n0)
        F = fc[cols].reindex(pd.date_range(start, periods=len(allP) * 24, freq="1h", tz="UTC")).ffill().bfill().to_numpy().reshape(len(allP), 24, 3)
        resid = (F[:, :, 0] - F[:, :, 1] - F[:, :, 2]) / 1000.0  # day-ahead forecast of residual load, GW
    except Exception:
        cols, resid = [], np.zeros_like(allP)
    naive = allP[n0 - 1 : -1]
    weekly = allP[n0 - 7 : -7]
    ridge = np.zeros_like(P)
    for k in range(days):  # expanding window, refit weekly
        i = n0 + k
        if k % 7 == 0:
            X = np.hstack([allP[6 : i - 1], allP[5 : i - 2], allP[0 : i - 7], resid[7:i]])
            Y = allP[7:i]
            model = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(X, Y)
        x = np.hstack([allP[i - 1], allP[i - 2], allP[i - 7], resid[i]])[None, :]
        ridge[k] = model.predict(x)[0]
    pv = pd.read_parquet(ROOT / "data/processed/pvgis_national_points.parquet")["thessaly_larissa"]
    pv = pv[pv.index.year == 2023].to_numpy()[:8760]
    pv = np.roll(pv, -304 * 24)[: days * 24] * 1.1
    park = Park(name="t", lat=39.64, lon=22.42, pv_dc_kwp=1100, pv_ac_kw=1000, grid_export_limit_kw=1000, support=SupportScheme(kind="merchant"))
    reg = RegulatoryConfig()
    eta = np.sqrt(0.88)
    dp = DispatchParams(0.5, 1.1, eta, eta, degradation_cost_eur_mwh=14.0, max_cycles_per_day=1.5)
    actual = P.reshape(-1)
    y = YearInputs(1.0, pv, actual, np.zeros(len(actual)), local_hour=greek_local_hour(2026, len(actual), 1.0), steps_per_day=24)
    base = _conv(park, reg, y)
    out = {"mae": {}, "gain": {}, "capture": {}}
    perfect = simulate_with_battery(park, reg, y, 2026, dp, lookahead_days=0).revenue_eur - base
    out["gain"]["perfect"] = round(perfect)
    for name, f in (("naive_yesterday", naive), ("same_weekday_last_week", weekly), ("ridge_lagged_prices_and_admie_forecasts", ridge)):
        out["mae"][name] = round(float(np.abs(f - P).mean()), 1)
        g = _plan_and_settle(park, reg, y, dp, f.reshape(-1)) - base
        out["gain"][name] = round(g)
        out["capture"][name] = round(g / perfect, 3)
    r = simulate_with_battery(park, reg, y, 2026, dp, policy="rules").revenue_eur - base
    out["gain"]["clock_rule"] = round(r)
    out["capture"]["clock_rule"] = round(r / perfect, 3)
    out["features"] = cols
    out["note"] = "11A station, PV charging only, 0.5 MW / 1.22 MWh on a 1.1 MWp park, real prices"
    return out


def _conv(park, reg, y):
    from bessdss.study import _converted_no_battery

    return _converted_no_battery(park, reg, y, 2026, "ac").revenue_eur


def _plan_and_settle(park, reg, y, dp, forecast):
    """Schedule with forecast prices, settle at the real prices (same physics, same caps)."""
    from bessdss.dispatch.lp import DispatchLP
    from bessdss.dispatch.simulate import _pv_available, static_cap_mw

    pv_avail, _ = _pv_available(park, "ac", y.pv_dc_mw)
    cap = static_cap_mw(park, reg, y.local_hour, True)
    lp = DispatchLP(24, 1.0, dp, cycle_block_steps=24)
    rev = 0.0
    for k in range(len(forecast) // 24):
        sl = slice(24 * k, 24 * k + 24)
        r = lp.solve(pv_avail[sl], cap[sl], forecast[sl], forecast[sl])
        rev += float(((r.e + r.d) * y.price[sl]).sum())
    return rev


def exp_replay():
    """History replay versus scenarios for a merchant 1 MW park: yearly battery value at real 2025-26 prices
    (15 minutes) compared with the simulated central scenario."""
    p = _real_year(0.25)
    n = len(p) // 96 * 96
    pv_h = pd.read_parquet(ROOT / "data/processed/pvgis_national_points.parquet")["thessaly_larissa"]
    pv = np.repeat(np.roll(pv_h[pv_h.index.year == 2023].to_numpy()[:8760], -304 * 24), 4)[:n] * 1.1
    park = Park(name="t", lat=39.64, lon=22.42, pv_dc_kwp=1100, pv_ac_kw=1000, grid_export_limit_kw=1000, support=SupportScheme(kind="merchant"))
    eta = np.sqrt(0.88)
    out = {}
    for cat in ("11A", "11B"):
        reg = RegulatoryConfig(station_category=cat)
        y = YearInputs(0.25, pv, p.to_numpy()[:n], np.zeros(n), local_hour=greek_local_hour(2026, n, 0.25))
        dp = DispatchParams(0.5, 1.1, eta, eta, degradation_cost_eur_mwh=14.0, max_cycles_per_day=1.5)
        base = simulate_pv_only(park, reg, y, 2026).revenue_eur
        conv = _conv(park, reg, y)
        r = simulate_with_battery(park, reg, y, 2026, dp, lookahead_days=1)
        out[cat] = dict(pv_only=round(base), conversion_penalty=round(conv - base), battery_gross=round(r.revenue_eur - conv), net=round(r.revenue_eur - base),
                        cycles=round(r.fec), eur_per_mw_year=round((r.revenue_eur - conv) / 0.5))
    out["note"] = "real 15-minute prices Nov 2025 to Oct 2026, perfect foresight, 0.5 MW / 1.22 MWh, merchant 1 MW park"
    return out


EXPERIMENTS = dict(duals=exp_duals, cycle_cost=exp_cycle_cost, degradation=exp_degradation, rebound=exp_rebound, forecast=exp_forecast, replay=exp_replay)

if __name__ == "__main__":
    which = sys.argv[1:] or list(EXPERIMENTS)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in which:
        t0 = time.time()
        try:
            res[name] = EXPERIMENTS[name]()
            res[name]["_seconds"] = round(time.time() - t0, 1)
        except Exception as exc:  # keep the failure visible in the log instead of hiding it
            res[name] = {"error": repr(exc)}
        print(name, json.dumps(res[name], indent=1, default=float)[:3000], flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1, default=float))
