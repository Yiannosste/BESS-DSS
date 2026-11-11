"""Risk analysis of one design: what drives the spread of outcomes and what drives the downside.

    * first-order sensitivity indices  S_i = Var(E[NPV | factor i]) / Var(NPV)  estimated on the
      Monte Carlo sample (binning for continuous factors),
    * downside profile: how each factor looks in the worst 10 % of outcomes,
    * tornado: one factor at a time moved to the ends of its range, everything else as sampled,
    * per-scenario results, break-even CAPEX and the subsidy that would bring expected NPV to zero.
"""

from __future__ import annotations

import numpy as np

from bessdss import assumptions as A
from bessdss.finance.model import FinanceParams, evaluate
from bessdss.study import DesignEconomics, PathSet, StudyContext, economics

FACTOR_LABELS = {
    "scenario": "Σενάριο αγοράς",
    "weather": "Καιρός (ηλιοφάνεια)",
    "model_err": "Σφάλμα μοντέλου τιμών στη διαφορά",
    "capture": "Ποιότητα πρόβλεψης στη λειτουργία",
    "capex": "Κόστος εγκατάστασης",
}


def path_factors(ctx: StudyContext, paths: PathSet) -> dict[str, np.ndarray]:
    """One number per path and factor. Weather is the average PV yield of the sampled years."""
    yields = np.array([ctx.pv_h[w].sum() for w in ctx.weather_years])
    yields = yields / yields.mean()
    wi = paths.weather_idx % len(yields)
    return {
        "scenario": paths.scen_idx.astype(float),
        "weather": yields[wi].mean(axis=1),
        "model_err": paths.model_err,
        "capture": paths.capture,
        "capex": paths.capex_mult,
    }


def sensitivity_indices(y: np.ndarray, factors: dict[str, np.ndarray], bins: int = 10) -> dict[str, float]:
    var = float(np.var(y))
    out = {}
    for name, x in factors.items():
        if var <= 0:
            out[name] = 0.0
            continue
        if name == "scenario":
            groups = [y[x == v] for v in np.unique(x)]
        else:
            edges = np.quantile(x, np.linspace(0, 1, bins + 1))
            idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
            groups = [y[idx == b] for b in range(bins)]
        groups = [g for g in groups if len(g)]
        mean = y.mean()
        between = sum(len(g) * (g.mean() - mean) ** 2 for g in groups) / len(y)
        out[name] = float(between / var)
    return out


def downside_profile(y: np.ndarray, factors: dict[str, np.ndarray], scen_names: list[str], q: float = 0.10) -> dict:
    """In the worst q share of outcomes: scenario mix and the mean shift of each factor in standard deviations."""
    cut = np.quantile(y, q)
    bad = y <= cut
    prof = {}
    for name, x in factors.items():
        if name == "scenario":
            prof[name] = {scen_names[int(v)]: float((x[bad] == v).mean()) for v in np.unique(x)}
        else:
            sd = float(np.std(x)) or 1.0
            prof[name] = float((x[bad].mean() - x.mean()) / sd)
    return prof


def tornado(ctx: StudyContext, econ: DesignEconomics, paths: PathSet) -> list[dict]:
    """Expected NPV when one factor is fixed at the low or high end of its range."""
    base = float(econ.fin.npv.mean())
    rows = []

    def run(p: PathSet | None = None, fin: FinanceParams | None = None) -> float:
        return float(economics(ctx, econ.design, p or paths, fin).fin.npv.mean())

    def with_paths(**kw) -> PathSet:
        d = dict(scen_idx=paths.scen_idx, weather_idx=paths.weather_idx, capture=paths.capture, model_err=paths.model_err, capex_mult=paths.capex_mult)
        d.update(kw)
        return PathSet(**d)

    n = len(paths.capture)
    lo, hi = A.low_high("capex_uncertainty_multiplier")
    rows.append(dict(factor="Κόστος εγκατάστασης (x%.2f έως x%.2f)" % (lo, hi), low=run(with_paths(capex_mult=np.full(n, hi))), high=run(with_paths(capex_mult=np.full(n, lo)))))
    sig = A.central("spread_model_error_sigma")
    rows.append(dict(factor="Σφάλμα μοντέλου τιμών (±1 τυπική απόκλιση)", low=run(with_paths(model_err=np.full(n, np.exp(-sig)))), high=run(with_paths(model_err=np.full(n, np.exp(sig))))))
    lo, hi = A.low_high("forecast_capture_factor")
    rows.append(dict(factor="Ποιότητα πρόβλεψης (%.2f έως %.2f)" % (lo, hi), low=run(with_paths(capture=np.full(n, lo))), high=run(with_paths(capture=np.full(n, hi)))))
    # weather: every year is the driest or the sunniest weather year of the park
    y = np.array([ctx.pv_h[w].sum() for w in ctx.weather_years])
    W = econ.design.gross.shape[2]
    order = np.argsort(y)[:W] if W < len(y) else np.argsort(y)
    dry, sunny = int(np.argmin(y)) % W, int(np.argmax(y)) % W
    del order
    rows.append(dict(factor="Καιρός (όλα τα έτη ξηρά ή όλα ηλιόλουστα)", low=min(run(with_paths(weather_idx=np.full_like(paths.weather_idx, dry))), run(with_paths(weather_idx=np.full_like(paths.weather_idx, sunny)))),
                     high=max(run(with_paths(weather_idx=np.full_like(paths.weather_idx, dry))), run(with_paths(weather_idx=np.full_like(paths.weather_idx, sunny))))))
    sc = [run(with_paths(scen_idx=np.full(n, i))) for i in range(len(ctx.scen_names))]
    rows.append(dict(factor="Σενάριο αγοράς (χειρότερο έως καλύτερο)", low=min(sc), high=max(sc)))
    for name, label in (("discount_rate_nominal", "Επιτόκιο προεξόφλησης"), ("opex_fixed_frac_of_capex", "Ετήσια συντήρηση")):
        lo, hi = A.low_high(name)
        f_lo, f_hi = FinanceParams(**{**ctx.fin.__dict__, name: lo}), FinanceParams(**{**ctx.fin.__dict__, name: hi})
        rows.append(dict(factor=f"{label} ({lo:g} έως {hi:g})", low=run(fin=f_hi), high=run(fin=f_lo)))
    lo, hi = A.low_high("aggregator_fee_frac")
    f_lo = FinanceParams(**{**ctx.fin.__dict__, "aggregator_fee_frac_of_incremental_revenue": lo})
    f_hi = FinanceParams(**{**ctx.fin.__dict__, "aggregator_fee_frac_of_incremental_revenue": hi})
    rows.append(dict(factor=f"Αμοιβή ΦοΣΕ ({lo:g} έως {hi:g})", low=run(fin=f_hi), high=run(fin=f_lo)))
    for r in rows:
        r["base"] = base
        r["swing"] = r["high"] - r["low"]
    return sorted(rows, key=lambda r: -r["swing"])


def per_scenario(ctx: StudyContext, econ: DesignEconomics, paths: PathSet) -> list[dict]:
    out = []
    for i, name in enumerate(ctx.scen_names):
        m = paths.scen_idx == i
        if not m.any():
            continue
        f = econ.fin
        out.append(dict(
            scenario=name, label=ctx.lib.scenarios[name].label, weight=float(ctx.scen_weights[i]),
            npv_mean=float(f.npv[m].mean()), npv_p90=float(np.percentile(f.npv[m], 10)),
            irr_p50=float(np.nanpercentile(f.irr_project[m], 50)), dscr_min_p90=float(np.nanpercentile(f.dscr_min[m], 10)),
            revenue_y1=float(econ.revenue_real[m, 0].mean()), revenue_last=float(econ.revenue_real[m, -1].mean()),
        ))
    return out


def break_even(ctx: StudyContext, econ: DesignEconomics, paths: PathSet) -> dict:
    """CAPEX multiplier and subsidy share at which the expected NPV is zero (NPV is linear in both)."""
    d = econ.design
    N = ctx.study.life_years
    yy = np.arange(N)[None, :]
    wi = paths.weather_idx % d.gross.shape[2]
    disch = d.interp_years(d.discharged)[paths.scen_idx[:, None], yy, wi]
    aux = d.interp_years(d.aux_cost)[paths.scen_idx]

    def npv(mult: float = 1.0, subsidy: float = 0.0) -> float:
        fin = FinanceParams(**{**ctx.fin.__dict__, "capex_subsidy_frac": subsidy})
        r = evaluate(fin, d.power_mw, d.energy_mwh, econ.revenue_real, discharged_mwh=disch, extra_opex_real=aux,
                     capex_multiplier=paths.capex_mult * mult, extra_capex_eur=econ.extra_capex)
        return float(r.npv.mean())

    n1, n0 = npv(1.0), npv(0.5)
    slope = (n1 - n0) / 0.5
    mult = 1.0 - n1 / slope if slope != 0 else float("nan")
    s1 = npv(1.0, 0.5)
    slope_s = (s1 - n1) / 0.5
    subsidy = -n1 / slope_s if slope_s != 0 else float("nan")
    base_capex = float(np.mean(econ.fin.capex_nominal))
    return dict(
        capex_multiplier_for_zero_npv=float(mult), break_even_capex_eur=float(base_capex * mult),
        break_even_capex_eur_per_kwh=float(base_capex * mult / (d.energy_mwh * 1000.0)),
        subsidy_share_for_zero_npv=float(subsidy),
    )


def full_risk_report(ctx: StudyContext, econ: DesignEconomics, paths: PathSet) -> dict:
    fac = path_factors(ctx, paths)
    npv = econ.fin.npv
    return dict(
        sensitivity_npv=sensitivity_indices(npv, fac),
        sensitivity_dscr=sensitivity_indices(np.nan_to_num(econ.fin.dscr_min, nan=0.0), fac),
        downside=downside_profile(npv, fac, ctx.scen_names),
        tornado=tornado(ctx, econ, paths),
        per_scenario=per_scenario(ctx, econ, paths),
        break_even=break_even(ctx, econ, paths),
        labels=FACTOR_LABELS,
    )
