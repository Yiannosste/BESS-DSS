"""End-to-end run of a study: market, sizing, risk, compliance, regulatory sensitivity, report data."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bessdss import assumptions as A
from bessdss.config import Study
from bessdss.dispatch.lp import DispatchParams
from bessdss.dispatch.simulate import simulate_pv_only, simulate_with_battery
from bessdss.market.library import ScenarioLibrary
from bessdss.regulatory.compliance import check, summary_status
from bessdss.risk.analysis import full_risk_report
from bessdss.sizing.search import legal_minimum_energy_mwh, screen_and_refine
from bessdss.study import (
    PathSet,
    StudyContext,
    _converted_no_battery,
    economics,
    marginal_cycle_cost,
    run_design,
)


def _market_job(args):
    name, year, wy = args
    lib = ScenarioLibrary()
    lib.market_year(name, year, wy)
    return 1


def precompute_market(ctx: StudyContext, extra_scenarios: tuple[str, ...] = (), n_workers: int = 4) -> None:
    """Resolve storage paths in this process, then simulate missing market years in parallel."""
    names = list(ctx.scen_names) + [s for s in extra_scenarios if s in ctx.lib.scenarios]
    for n in names:
        ctx.lib.storage_path(n)
    jobs = [(s, y, w) for s in names for y in ctx.rep_years for w in ctx.weather_years]
    if n_workers <= 1:
        for j in jobs:
            ctx.lib.market_year(*j)
        return
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_market_job, jobs, chunksize=8))


REG_VARIANTS = [
    ("cap_40", "Όριο έγχυσης 11Α στο 40% αντί για 60%", {"cap_11a_frac_of_installed": 0.40}, ("11A",)),
    ("neg_strict", "Καμία ενίσχυση σε κάθε μη θετική τιμή (αυστηρή ανάγνωση)", {"negative_price_rule": "nonpositive_all"}, ("11A",)),
    ("stored_support", "Η ενέργεια της μπαταρίας παίρνει την τιμή ενίσχυσης (αντίθετη ανάγνωση)", {"stored_energy_support": "support_price"}, ("11A",)),
    ("fip_one_way", "Προσαύξηση μονής κατεύθυνσης (χωρίς επιστροφή)", {"fip_two_way": False}, ("11A",)),
    ("basis_ac", "«Εγκατεστημένη ισχύς» = ισχύς μετατροπέων (AC)", {"installed_power_basis": "pv_ac_kw"}, ("11A", "11B")),
    ("pre2019_ok", "Παλιό πάρκο κρατά την ενίσχυση ως 11Α", {"pre2019_supported_park_eligible_11a": True}, ("11A",)),
]


def regulatory_sensitivity(ctx: StudyContext, paths: PathSet, category: str, p: float, e: float, wys: list[int]) -> list[dict]:
    """Expected NPV of the chosen design under each alternative legal reading (hourly screening fidelity)."""
    park = ctx.study.park
    base = economics(ctx, run_design(ctx, category, p, e, resolution="hourly", weather_years=wys), paths).metrics()
    rows = [dict(key="base", label="Βασική ανάγνωση (προεπιλογές του μητρώου)", npv_mean=base["npv_mean"], npv_p90=base["npv_p90"], revenue_y1=base["revenue_y1_p50"])]
    for key, label, upd, cats in REG_VARIANTS:
        if category not in cats:
            continue
        if key in ("neg_strict", "fip_one_way") and park.support.kind != "fip":
            continue
        if key == "stored_support" and park.support.kind not in ("fit", "fip"):
            continue
        if key == "pre2019_ok" and not (park.support.kind in ("fit", "fip") and park.commissioned <= ctx.study.regulatory.eligibility_commissioning_after):
            continue
        reg = ctx.study.regulatory.model_copy(update=upd)
        m = economics(ctx, run_design(ctx, category, p, e, resolution="hourly", weather_years=wys, reg=reg), paths).metrics()
        rows.append(dict(key=key, label=label, npv_mean=m["npv_mean"], npv_p90=m["npv_p90"], revenue_y1=m["revenue_y1_p50"]))
    return rows


def frozen_contrast(ctx: StudyContext, paths: PathSet, category: str, p: float, e: float, wys: list[int]) -> dict | None:
    """Result if the market stayed as in 2026 forever (history replay), to show the overestimation."""
    if "frozen_2026" not in ctx.lib.scenarios:
        return None
    d = run_design(ctx, category, p, e, resolution="hourly", weather_years=wys, scen_names=["frozen_2026"])
    fixed = PathSet(
        scen_idx=np.zeros(len(paths.capture), int), weather_idx=paths.weather_idx, capture=paths.capture,
        model_err=paths.model_err, capex_mult=paths.capex_mult,
    )
    m = economics(ctx, d, fixed).metrics()
    return dict(npv_mean=m["npv_mean"], npv_p90=m["npv_p90"], irr_p50=m["irr_p50"], revenue_y1=m["revenue_y1_p50"], payback_p50=m["payback_p50"])


def operation_example(ctx: StudyContext, category: str, p: float, e: float) -> dict:
    """One simulated year at 15 minutes for the chosen design (central scenario, median weather year):
    traces for typical days and the value under the three operating policies."""
    study = ctx.study
    b, park = study.battery, study.park
    s = "central" if "central" in ctx.scen_names else ctx.scen_names[0]
    y = ctx.rep_years[min(1, len(ctx.rep_years) - 1)]
    yields = {w: ctx.pv_h[w].sum() for w in ctx.weather_years}
    w = sorted(yields, key=yields.get)[len(yields) // 2]
    reg = study.regulatory.model_copy(update={"station_category": category})
    yi = ctx.inputs(s, y, w, "15min")
    eta = float(np.sqrt(b.round_trip_efficiency_ac))
    dp = DispatchParams(p, e * (b.soc_max - b.soc_min), eta, eta, coupling=b.coupling, eta_inverter=park.eta_inverter,
                        degradation_cost_eur_mwh=marginal_cycle_cost(study, ctx.fin),
                        max_cycles_per_day=(b.max_cycles_per_year / 365.0) if b.max_cycles_per_year else None)
    conv = _converted_no_battery(park, reg, yi, y, b.coupling).revenue_eur
    base = simulate_pv_only(park, reg, yi, y)
    out = {"scenario": s, "year": y, "weather_year": w, "policies": {}}
    for pol in ("perfect", "persistence", "rules"):
        r = simulate_with_battery(park, reg, yi, y, dp, policy=pol, lookahead_days=1 if pol != "rules" else 0, keep_traces=(pol == "perfect"))
        out["policies"][pol] = dict(gross_value=r.revenue_eur - conv, discharged_mwh=r.discharged_mwh, fec=r.fec, cycled_in=r.cycled_within_interval_mwh)
        if pol == "perfect":
            tr = r
    pv_ac = np.minimum(yi.pv_dc_mw * park.eta_inverter, park.pv_ac_kw / 1000.0)
    # two typical days: the spring day with most zero-price quarter hours and a median July day
    day = lambda k: slice(96 * k, 96 * k + 96)  # noqa: E731
    # spring day: many zero-price quarter hours at noon AND a normal evening price (a typical surplus day,
    # not the most extreme one)
    zero_cnt = np.array([(yi.price[day(k)][36:64] <= 0.5).sum() for k in range(365)])
    evening = np.array([yi.price[day(k)][64:84].mean() for k in range(365)])
    score = np.where(evening > np.median(evening), zero_cnt, 0)
    spring = int(60 + np.argmax(score[60:150]))
    summer = 196
    days = {}
    for label, k in (("spring_surplus", spring), ("summer", summer), ("winter", 20)):
        sl = day(k)
        days[label] = dict(
            day_of_year=k, local_hour=yi.local_hour[sl].tolist(), price=yi.price[sl].tolist(), pv=pv_ac[sl].tolist(),
            export=tr.net_export_mw[sl].tolist(), charge=tr.c[sl].tolist(), discharge=tr.d[sl].tolist(), soc=tr.soc[sl].tolist(),
            base_export=base.net_export_mw[sl].tolist(),
        )
    out["days"] = days
    return out


def run_study(study: Study, out_dir: str | Path, n_workers: int = 4, n_paths: int = 2000, log=print) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = StudyContext.build(study)
    log(f"[1/6] market scenarios ({len(ctx.scen_names)} scenarios x {len(ctx.rep_years)} years x {len(ctx.weather_years)} weather years)")
    precompute_market(ctx, extra_scenarios=("frozen_2026",), n_workers=n_workers)
    paths = PathSet.sample(ctx, n_paths)
    log("[2/6] sizing")
    sizing = screen_and_refine(study, ctx, paths, n_workers=n_workers, log=log)
    best = sizing["best"]
    d = best.design
    log(f"[3/6] risk analysis of {d.category} {d.power_mw} MW / {d.energy_mwh} MWh")
    risk = full_risk_report(ctx, best, paths)
    log("[4/6] compliance and regulatory sensitivity")
    comp = check(study, d.category, d.power_mw, d.energy_mwh)
    wys = sizing["screening_weather_years"]
    reg_sens = regulatory_sensitivity(ctx, paths, d.category, d.power_mw, d.energy_mwh, wys)
    frozen = frozen_contrast(ctx, paths, d.category, d.power_mw, d.energy_mwh, wys)
    log("[5/6] operation example")
    oper = operation_example(ctx, d.category, d.power_mw, d.energy_mwh)
    market = ctx.lib.summary(years=tuple(ctx.rep_years), weather_years=wys)
    market = [m for m in market if m["scenario"] in ctx.scen_names or m["scenario"] == "frozen_2026"]
    f = best.fin
    res = dict(
        meta=dict(tool="bessdss", version="0.1.0", run_seconds=None, base_year=study.base_year, cod_year=study.cod_year,
                  life_years=study.life_years, n_paths=n_paths, weather_years=ctx.weather_years, rep_years=ctx.rep_years,
                  scenarios={n: dict(label=ctx.lib.scenarios[n].label, weight=float(w), description=ctx.lib.scenarios[n].description)
                             for n, w in zip(ctx.scen_names, ctx.scen_weights, strict=True)}),
        study=json.loads(study.model_dump_json()),
        pv=ctx.pv_report,
        legal_minimum_energy_mwh=legal_minimum_energy_mwh(study),
        verdict=sizing["verdict"],
        best=sizing["best_metrics"],
        marginal=sizing["marginal"],
        screening=sizing["screening"],
        finals=sizing["finals"],
        risk=risk,
        compliance=comp,
        compliance_status=summary_status(comp),
        regulatory_sensitivity=reg_sens,
        frozen_2026=frozen,
        operation=oper,
        market=market,
        storage_paths={n: ctx.lib.storage_path(n) for n in ctx.scen_names},
        design_detail=dict(
            years=d.years, rep_years=d.rep_years, scen_names=d.scen_names,
            gross=d.gross.mean(axis=2).tolist(), penalty=d.penalty.mean(axis=2).tolist(), base_rev=d.base_rev.mean(axis=2).tolist(),
            fec=d.fec.mean(axis=2).tolist(), discharged=d.discharged.mean(axis=2).tolist(),
            lost_base=d.lost_base.mean(axis=2).tolist(), lost_with=d.lost_with.mean(axis=2).tolist(),
            cycled_in=d.cycled_in.mean(axis=2).tolist(), soh=d.soh.tolist(), mv_power=d.mv_power.tolist(), mv_energy=d.mv_energy.tolist(),
        ),
        distributions=dict(
            npv=np.round(f.npv).tolist(),
            revenue_p10=np.percentile(best.revenue_real, 10, axis=0).tolist(), revenue_p50=np.percentile(best.revenue_real, 50, axis=0).tolist(),
            revenue_p90=np.percentile(best.revenue_real, 90, axis=0).tolist(),
            dscr_p10=np.nanpercentile(f.dscr, 10, axis=0).tolist(), dscr_p50=np.nanpercentile(f.dscr, 50, axis=0).tolist(),
            target_dscr=ctx.fin.target_dscr,
        ),
        finance=dict(ctx.fin.__dict__),
        cycle_cost_eur_mwh=marginal_cycle_cost(study, ctx.fin),
        assumptions=A.all_entries(),
    )
    res["meta"]["run_seconds"] = round(time.time() - t0, 1)
    (out_dir / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log("[6/6] report")
    from bessdss.report.build import build_report

    build_report(res, out_dir)
    log(f"done in {res['meta']['run_seconds']} s: {out_dir / 'report.html'}")
    return res
