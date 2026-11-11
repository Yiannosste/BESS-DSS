"""Project-finance model for the battery investment (incremental to the existing PV park).

Conventions (see Decision Log D-014):
  * Market scenarios are produced in REAL euros of the base year. This module converts every
    cash flow to NOMINAL euros with one explicit inflation index, computes tax and debt in
    nominal terms, and discounts with a NOMINAL rate. Real and nominal are never mixed.
  * The investment case is INCREMENTAL: cash flow of (PV + battery) minus cash flow of (PV alone).
  * Year 0 is the investment year, operation runs in years 1..N.
  * All functions are vectorised over scenario paths: inputs of shape (n_paths, N).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FinanceParams:
    life_years: int = 15
    inflation: float = 0.02  # general price inflation per year
    discount_rate_nominal: float = 0.075  # owner's opportunity cost of capital (WACC), nominal
    tax_rate: float = 0.22
    depreciation_years: int = 10
    tax_shield_usable: bool = True  # losses offset the park's other taxable profit in the same year
    # CAPEX in base-year euros
    capex_energy_eur_per_kwh: float = 160.0
    capex_power_eur_per_kw: float = 70.0
    capex_fixed_eur: float = 40_000.0
    capex_subsidy_frac: float = 0.0
    # OPEX in base-year euros
    opex_fixed_frac_of_capex: float = 0.015
    insurance_frac_of_capex: float = 0.005
    aggregator_fee_frac_of_incremental_revenue: float = 0.07
    # augmentation / replacement: list of (year, fraction of nameplate energy added)
    augmentation: list[tuple[int, float]] = field(default_factory=list)
    battery_cost_decline_per_year: float = 0.03  # real decline of the energy block cost
    augmentation_cost_share_of_energy_capex: float = 0.7
    residual_value_frac_of_capex: float = 0.0
    decommissioning_frac_of_capex: float = 0.02
    # debt
    gearing: float = 0.6
    debt_tenor_years: int = 10
    debt_rate: float = 0.055
    target_dscr: float = 1.25


def capex_base(p: FinanceParams, power_mw: float, energy_mwh: float) -> float:
    """Total installed cost in base-year euros, before subsidy."""
    if power_mw <= 0 or energy_mwh <= 0:
        return 0.0
    return (
        p.capex_energy_eur_per_kwh * energy_mwh * 1000.0
        + p.capex_power_eur_per_kw * power_mw * 1000.0
        + p.capex_fixed_eur
    )


def irr(cashflows: np.ndarray, lo: float = -0.99, hi: float = 1.0, iters: int = 80) -> np.ndarray:
    """Vectorised IRR by bisection. cashflows: (n_paths, N+1). NaN when no sign change."""
    cf = np.atleast_2d(np.asarray(cashflows, float))
    t = np.arange(cf.shape[1])

    def npv(r):
        return (cf / (1.0 + r[:, None]) ** t).sum(axis=1)

    a = np.full(cf.shape[0], lo)
    b = np.full(cf.shape[0], hi)
    fa, fb = npv(a), npv(b)
    ok = np.sign(fa) != np.sign(fb)
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = npv(m)
        left = np.sign(fm) == np.sign(fa)
        a = np.where(left, m, a)
        fa = np.where(left, fm, fa)
        b = np.where(left, b, m)
    out = 0.5 * (a + b)
    out[~ok] = np.nan
    return out


@dataclass
class FinanceResult:
    capex_nominal: np.ndarray  # (n_paths,) net of subsidy
    npv: np.ndarray  # (n_paths,) project NPV, nominal discounting, EUR of year 0
    irr_project: np.ndarray
    irr_equity: np.ndarray
    payback_years: np.ndarray  # simple payback on unlevered after-tax cash flow, NaN if never
    dscr: np.ndarray  # (n_paths, tenor) incremental DSCR per debt year
    dscr_min: np.ndarray
    debt_amount: np.ndarray  # (n_paths,)
    debt_service: np.ndarray  # (n_paths, tenor)
    fcf_unlevered: np.ndarray  # (n_paths, N+1) nominal
    cfads: np.ndarray  # (n_paths, N)
    ebitda: np.ndarray  # (n_paths, N) nominal
    lcos_eur_mwh: np.ndarray  # levelised cost of storage per MWh discharged


def evaluate(
    p: FinanceParams,
    power_mw: float,
    energy_mwh: float,
    incremental_revenue_real: np.ndarray,
    discharged_mwh: np.ndarray | None = None,
    extra_opex_real: np.ndarray | None = None,
    capex_multiplier: np.ndarray | float = 1.0,
    extra_capex_eur: float = 0.0,
) -> FinanceResult:
    """Cash-flow waterfall for one battery size over many scenario paths.

    incremental_revenue_real: (n_paths, N) revenue of PV+battery minus PV alone, base-year euros,
        before aggregator fee. Must already contain grid-charging costs.
    discharged_mwh: (n_paths, N) energy discharged per year (for LCOS). Optional.
    extra_opex_real: (n_paths, N) or (N,) further operating cost in base-year euros (e.g. auxiliary energy).
    capex_multiplier: scalar or (n_paths,) uncertainty factor on the installed cost.
    extra_capex_eur: one-off cost not proportional to size (e.g. guarantee letter fees, meters).
    """
    rev_real = np.atleast_2d(np.asarray(incremental_revenue_real, float))
    n_paths, N = rev_real.shape
    if N != p.life_years:
        raise ValueError(f"revenue has {N} years but life_years={p.life_years}")
    years = np.arange(1, N + 1)
    infl = (1.0 + p.inflation) ** years
    mult = np.broadcast_to(np.asarray(capex_multiplier, float), (n_paths,)).copy()
    capex0 = (capex_base(p, power_mw, energy_mwh) + extra_capex_eur) * mult  # (n_paths,)
    capex_net = capex0 * (1.0 - p.capex_subsidy_frac)

    revenue = rev_real * infl
    opex = capex0[:, None] * (p.opex_fixed_frac_of_capex + p.insurance_frac_of_capex) * infl[None, :]
    opex = opex + p.aggregator_fee_frac_of_incremental_revenue * np.maximum(revenue, 0.0)
    if extra_opex_real is not None:
        opex = opex + np.atleast_2d(extra_opex_real) * infl
    ebitda = revenue - opex

    # augmentation capex (nominal), depreciated straight-line from the following year
    aug = np.zeros((n_paths, N))
    for yr, frac in p.augmentation:
        if 1 <= yr <= N:
            real_cost = (
                frac * energy_mwh * 1000.0 * p.capex_energy_eur_per_kwh * p.augmentation_cost_share_of_energy_capex
                * (1.0 - p.battery_cost_decline_per_year) ** yr
            )
            aug[:, yr - 1] += real_cost * (1.0 + p.inflation) ** yr * mult

    dep = np.zeros((n_paths, N))
    nd = min(p.depreciation_years, N)
    dep[:, :nd] += capex_net[:, None] / p.depreciation_years
    for yr, _ in p.augmentation:
        if 1 <= yr <= N and aug[:, yr - 1].any():
            n_left = max(1, min(p.depreciation_years, N - yr))
            dep[:, yr : yr + n_left] += aug[:, yr - 1 : yr] / n_left

    # debt: annuity on a share of net capex
    debt = p.gearing * capex_net  # (n_paths,)
    tenor = min(p.debt_tenor_years, N)
    interest = np.zeros((n_paths, N))
    principal = np.zeros((n_paths, N))
    if p.gearing > 0 and tenor > 0:
        ann = debt * p.debt_rate / (1.0 - (1.0 + p.debt_rate) ** (-tenor))
        bal = debt.copy()
        for k in range(tenor):
            interest[:, k] = bal * p.debt_rate
            principal[:, k] = ann - interest[:, k]
            bal = bal - principal[:, k]
    debt_service = interest + principal

    def tax_on(taxable):
        if p.tax_shield_usable:
            return p.tax_rate * taxable
        # stand-alone: losses carried forward 5 years (Greek rule), simple running pool
        tax = np.zeros_like(taxable)
        for i in range(taxable.shape[0]):
            pool: list[list[float]] = []
            for y in range(N):
                inc = taxable[i, y]
                pool = [q for q in pool if y - q[0] <= 5]
                if inc < 0:
                    pool.append([y, -inc])
                else:
                    for q in pool:
                        use = min(q[1], inc)
                        q[1] -= use
                        inc -= use
                    tax[i, y] = p.tax_rate * inc
        return tax

    tax_unlev = tax_on(ebitda - dep)
    end_flow = np.zeros((n_paths, N))
    end_flow[:, -1] = (p.residual_value_frac_of_capex - p.decommissioning_frac_of_capex) * capex0 * infl[-1]
    fcf_ops = ebitda - tax_unlev - aug + end_flow
    fcf = np.concatenate([-capex_net[:, None], fcf_ops], axis=1)

    t = np.arange(N + 1)
    disc = (1.0 + p.discount_rate_nominal) ** t
    npv = (fcf / disc).sum(axis=1)
    irr_p = irr(fcf)

    tax_lev = tax_on(ebitda - dep - interest)
    cfads = ebitda - tax_lev - aug
    eq = np.concatenate([-(capex_net - debt)[:, None], cfads - debt_service + end_flow], axis=1)
    irr_e = irr(eq)
    if tenor > 0 and p.gearing > 0:
        dscr = cfads[:, :tenor] / debt_service[:, :tenor]
        dscr_min = dscr.min(axis=1)
    else:
        dscr = np.full((n_paths, max(tenor, 1)), np.nan)
        dscr_min = np.full(n_paths, np.nan)
    cum = np.cumsum(fcf, axis=1)
    pos = cum >= 0
    first = np.where(pos.any(axis=1), pos.argmax(axis=1).astype(float), np.nan)
    payback = first.copy()
    ok = ~np.isnan(first) & (first >= 1)
    k = first[ok].astype(int)
    rows = np.where(ok)[0]
    prev, cur = cum[rows, k - 1], cum[rows, k]
    payback[ok] = (k - 1) + np.divide(-prev, cur - prev, out=np.ones_like(prev), where=(cur - prev) != 0)

    if discharged_mwh is not None:
        dis = np.atleast_2d(np.asarray(discharged_mwh, float))
        cost_pv = capex_net + ((opex + aug) / disc[1:]).sum(axis=1)
        e_pv = (dis / disc[1:]).sum(axis=1)
        lcos = np.divide(cost_pv, e_pv, out=np.full(n_paths, np.nan), where=e_pv > 0)
    else:
        lcos = np.full(n_paths, np.nan)

    return FinanceResult(
        capex_nominal=capex_net,
        npv=npv,
        irr_project=irr_p,
        irr_equity=irr_e,
        payback_years=payback,
        dscr=dscr,
        dscr_min=dscr_min,
        debt_amount=debt,
        debt_service=debt_service[:, :tenor],
        fcf_unlevered=fcf,
        cfads=cfads,
        ebitda=ebitda,
        lcos_eur_mwh=lcos,
    )


def max_debt_for_dscr(p: FinanceParams, cfads_downside: np.ndarray, target_dscr: float | None = None) -> float:
    """Largest annuity loan whose debt service is covered target_dscr times in the downside case.

    cfads_downside: (N,) cash flow available for debt service in the bank's bad-year case (nominal).
    This is how a lender sizes debt: on the P90 (or downside) case, not on the average.
    """
    target = target_dscr or p.target_dscr
    tenor = min(p.debt_tenor_years, len(cfads_downside))
    if tenor == 0:
        return 0.0
    max_service = np.min(cfads_downside[:tenor]) / target
    if max_service <= 0:
        return 0.0
    return float(max_service * (1.0 - (1.0 + p.debt_rate) ** (-tenor)) / p.debt_rate)
