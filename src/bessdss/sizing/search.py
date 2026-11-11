"""Battery sizing: the size is an output of the study, not an input.

Method
    1. Legal feasible set: usable energy >= minimum hours x installed PV power (art. 11Γ), battery
       power <= allowed share of the export limit, both station categories (11A and 11B).
    2. Screening of a grid of (category, power, energy) at hourly resolution on three weather years
       (dry, median, sunny), in parallel.
    3. Full evaluation of the best candidates at 15-minute resolution on all weather years.
    4. Marginal value of one more MW and one more MWh from the LP duals (envelope theorem): shows
       whether the chosen size sits at an optimum or on a legal bound, at no extra simulation cost.

"No battery" is always a candidate: if no design has positive expected NPV the answer is "do not invest".
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bessdss.config import Study
from bessdss.finance.model import evaluate
from bessdss.study import DesignEconomics, DesignResult, PathSet, StudyContext, economics, run_design

_CTX: StudyContext | None = None


def _init(study_json: str) -> None:
    global _CTX
    _CTX = StudyContext.build(Study.model_validate_json(study_json))


def _work(args: tuple) -> DesignResult:
    category, p, e, kw = args
    assert _CTX is not None
    reg = None
    if kw.get("reg_updates"):
        reg = _CTX.study.regulatory.model_copy(update=kw["reg_updates"])
    return run_design(
        _CTX, category, p, e, resolution=kw.get("resolution", "hourly"), policy=kw.get("policy", "perfect"),
        lookahead_days=kw.get("lookahead_days", 0), weather_years=kw.get("weather_years"), reg=reg,
        cycle_cost_eur_mwh=kw.get("cycle_cost_eur_mwh"),
    )


def evaluate_many(study: Study, designs: list[tuple[str, float, float]], n_workers: int = 4, **kw) -> list[DesignResult]:
    """Simulate many designs in parallel processes. Market years must already be cached on disk."""
    jobs = [(c, p, e, kw) for c, p, e in designs]
    if n_workers <= 1 or len(jobs) == 1:
        _init(study.model_dump_json())
        return [_work(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs)), initializer=_init, initargs=(study.model_dump_json(),)) as ex:
        return list(ex.map(_work, jobs))


def legal_minimum_energy_mwh(study: Study) -> float:
    """Smallest NAMEPLATE energy that satisfies the minimum usable capacity rule (REG-A-007)."""
    reg, b, park = study.regulatory, study.battery, study.park
    if not reg.min_capacity_rule_active:
        return 0.0
    usable = reg.min_storage_hours * park.installed_power_mw(reg.installed_power_basis)
    return usable / (b.soc_max - b.soc_min)


def candidate_grid(
    study: Study,
    categories: tuple[str, ...] = ("11A", "11B"),
    power_fracs: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    energy_mults: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    duration_range: tuple[float, float] = (1.0, 6.5),
) -> list[tuple[str, float, float]]:
    park, reg, b = study.park, study.regulatory, study.battery
    export = park.grid_export_limit_kw / 1000.0
    e_min = legal_minimum_energy_mwh(study)
    window = b.soc_max - b.soc_min
    if e_min <= 0:  # rule switched off: explore from half an hour of the export limit upwards
        e_min = 0.5 * export / window
    out = []
    for cat in categories:
        for pf in power_fracs:
            p = round(pf * export * reg.max_battery_power_frac_of_export, 4)
            for m in energy_mults:
                e = round(m * e_min, 4)
                dur = e * window / p
                if duration_range[0] <= dur <= duration_range[1]:
                    out.append((cat, p, e))
    return out


def screening_weather_years(ctx: StudyContext) -> list[int]:
    """Dry, median and sunny year of the park (by annual PV yield)."""
    y = {w: float(ctx.pv_h[w].sum()) for w in ctx.weather_years}
    order = sorted(y, key=y.get)
    return sorted({order[0], order[len(order) // 2], order[-1]})


def marginal_npv(ctx: StudyContext, econ: DesignEconomics, paths: PathSet) -> dict:
    """Expected NPV gain of one more MW of power and one more MWh of nameplate energy.

    The LP duals give the exact marginal revenue of power and of usable energy in every simulated
    year (envelope theorem). They are pushed through the same cash-flow model as the design itself.
    """
    d, fin = econ.design, ctx.fin
    b = ctx.study.battery
    window = b.soc_max - b.soc_min
    n, N = paths.weather_idx.shape
    mvp = d.interp_years(d.mv_power)[paths.scen_idx]  # (n, N) EUR per MW per year
    soh = d.soh[:, :-1][paths.scen_idx]
    mve = d.interp_years(d.mv_energy)[paths.scen_idx] * window * soh  # EUR per nameplate MWh per year
    k = (paths.capture * paths.model_err)[:, None]
    base = econ.fin.npv.mean()
    disch = d.interp_years(d.discharged)[paths.scen_idx[:, None], np.arange(N)[None, :], paths.weather_idx % d.gross.shape[2]]
    aux = d.interp_years(d.aux_cost)[paths.scen_idx]
    out = {}
    for name, dp, de, drev in (("power", 1.0, 0.0, mvp * k), ("energy", 0.0, 1.0, mve * k)):
        h = 0.01  # 10 kW or 10 kWh step
        res = evaluate(
            fin, d.power_mw + h * dp, d.energy_mwh + h * de, econ.revenue_real + h * drev, discharged_mwh=disch,
            extra_opex_real=aux, capex_multiplier=paths.capex_mult, extra_capex_eur=econ.extra_capex,
        )
        out[f"dnpv_d{name}"] = float((res.npv.mean() - base) / h)
    return out


def rank_designs(metrics: list[dict], target_dscr: float) -> list[dict]:
    """Order designs by expected NPV and attach a bankability flag."""
    for m in metrics:
        m["bankable"] = bool(np.isfinite(m["dscr_min_p90"]) and m["dscr_min_p90"] >= target_dscr)
    return sorted(metrics, key=lambda m: -m["npv_mean"])


def verdict(best: dict) -> str:
    """Three-way answer to 'is a battery worth it': yes / only_with_risk / no."""
    if best["npv_mean"] <= 0:
        return "no"
    if best["npv_p90"] > 0:
        return "yes"
    return "only_with_risk"


def screen_and_refine(
    study: Study, ctx: StudyContext, paths: PathSet, n_workers: int = 4, n_final: int = 3, log=print
) -> dict:
    """Full sizing run. Returns screening table, final designs with economics, and the recommendation."""
    grid = candidate_grid(study)
    wys = screening_weather_years(ctx)
    log(f"screening {len(grid)} designs at hourly resolution on weather years {wys}")
    screened = evaluate_many(study, grid, n_workers=n_workers, resolution="hourly", weather_years=wys)
    scr = [economics(ctx, d, paths) for d in screened]
    table = rank_designs([e.metrics() for e in scr], ctx.fin.target_dscr)
    # finalists: best overall, plus the best of each category so that both legal routes are shown
    chosen: list[tuple[str, float, float]] = []
    for m in table:
        key = (m["category"], m["power_mw"], m["energy_mwh"])
        if len(chosen) < n_final and key not in chosen:
            chosen.append(key)
    for cat in ("11A", "11B"):
        best_cat = next((m for m in table if m["category"] == cat), None)
        if best_cat:
            key = (best_cat["category"], best_cat["power_mw"], best_cat["energy_mwh"])
            if key not in chosen:
                chosen.append(key)
    log(f"full evaluation of {len(chosen)} designs at 15-minute resolution on all weather years")
    finals = evaluate_many(study, chosen, n_workers=n_workers, resolution="15min", lookahead_days=1)
    fin_econ = [economics(ctx, d, paths) for d in finals]
    fin_table = rank_designs([e.metrics() for e in fin_econ], ctx.fin.target_dscr)
    best_m = fin_table[0]
    best = next(e for e in fin_econ if (e.design.category, e.design.power_mw, e.design.energy_mwh) == (best_m["category"], best_m["power_mw"], best_m["energy_mwh"]))
    return dict(
        screening=table, finals=fin_table, final_econ=fin_econ, best=best, best_metrics=best_m,
        verdict=verdict(best_m), marginal=marginal_npv(ctx, best, paths), screening_weather_years=wys,
    )
