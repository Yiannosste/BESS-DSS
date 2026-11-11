"""One year of operation of a PV park with (or without) a battery under Greek settlement rules.

Three operating policies are available, so that the cost of not knowing the future is explicit:
    "perfect"      schedule built with the true prices of the day (upper bound, optimistic),
    "persistence"  schedule built with yesterday's prices, settled at the true prices
                   (a floor for what a day-ahead forecast can do),
    "rules"        fixed clock rule: store PV surplus and midday energy, release in the evening
                   (no optimisation at all, the simplest baseline).

Settlement (remuneration per interval) depends on the support scheme and the regulatory switches,
see `remuneration()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bessdss.config import Park, RegulatoryConfig, SupportScheme
from bessdss.dispatch.lp import DispatchLP, DispatchParams


@dataclass
class YearInputs:
    """Time series of one simulated year on a regular grid (length T, step dt_h)."""

    dt_h: float
    pv_dc_mw: np.ndarray  # PV DC power available at the inverter input, MW
    price: np.ndarray  # day-ahead price EUR/MWh
    curtail_frac: np.ndarray  # share of the park's potential output it is ordered to shed (0..1)
    pv_ref_price_monthly: np.ndarray | None = None  # per interval: monthly PV market reference price (ΕΤΑ), for FiP
    steps_per_day: int | None = None
    local_hour: np.ndarray | None = None  # hour of day in Greek local time at interval start (for legal time windows)

    def __post_init__(self):
        if self.steps_per_day is None:
            self.steps_per_day = int(round(24.0 / self.dt_h))
        if self.local_hour is None:  # default: UTC+2, series starting at 00:00 UTC
            self.local_hour = (np.arange(len(self.price)) * self.dt_h + 2.0) % 24.0


def greek_local_hour(year: int, n_steps: int, dt_h: float) -> np.ndarray:
    """Hour of day in Europe/Athens for a UTC grid starting on 1 January of `year` (handles DST)."""
    import pandas as pd

    idx = pd.date_range(f"{year}-01-01", periods=n_steps, freq=pd.Timedelta(hours=dt_h), tz="UTC").tz_convert("Europe/Athens")
    return (idx.hour + idx.minute / 60.0).to_numpy(float)


@dataclass
class YearResult:
    revenue_eur: float  # total market + support revenue of the park in the year
    pv_direct_mwh: float
    discharged_mwh: float  # at the battery bus
    charged_pv_mwh: float
    charged_grid_mwh: float
    curtailed_mwh: float  # PV energy lost (curtailment orders, negative prices, export limit, clipping not included)
    clipped_mwh: float  # PV energy lost at the inverter limit
    grid_cost_eur: float
    fec: float  # full equivalent cycles on usable energy
    mean_soc_frac: float  # of usable energy
    marginal_value_power_eur_per_mw: float
    marginal_value_energy_eur_per_mwh: float
    soc: np.ndarray | None = None
    net_export_mw: np.ndarray | None = None
    d: np.ndarray | None = None
    c: np.ndarray | None = None
    milp_days: int = 0
    cycled_within_interval_mwh: float = 0.0  # diagnostic: energy cycled inside single intervals (rule exploitation)


def effective_support_kind(park: Park, reg: RegulatoryConfig, year: int, with_battery: bool, coupling: str = "ac") -> str:
    """Which remuneration regime applies to the park in a given year.

    Without a battery: the park's own contract until it expires, then merchant.
    With a battery (art. 11Γ ν. 4685/2020):
      * 11B: operating aid (FiT, FiP) is lost, the park is merchant (REG-A-005). A PPA is a private
        contract and stays.
      * 11A: aid is kept only for parks commissioned after the cut-off date (REG-A-001). For older
        supported parks the law is silent (REG-A-002): by default the aid is treated as lost.
      * DC coupling behind a common inverter: aid treated as lost unless the switch says otherwise
        (REG-A-026, unclear).
    """
    kind = park.support.kind
    if park.support.end_year is not None and year > park.support.end_year:
        kind = "merchant"
    if not with_battery or kind in ("merchant", "ppa"):
        return kind
    if reg.station_category == "11B":
        return "merchant"
    if park.commissioned <= reg.eligibility_commissioning_after and not reg.pre2019_supported_park_eligible_11a:
        return "merchant"
    if coupling == "dc" and not reg.dc_coupling_keeps_support:
        return "merchant"
    return kind


def aid_suspended(price: np.ndarray, dt_h: float, reg: RegulatoryConfig) -> np.ndarray:
    """Intervals in which the sliding premium is not paid because of the negative price rule."""
    p = np.asarray(price, float)
    if reg.negative_price_rule == "none":
        return np.zeros(len(p), bool)
    if reg.negative_price_rule == "nonpositive_all":
        return p <= 0.0
    # negative for more than two consecutive hours, judged on the hourly average price
    sph = int(round(1.0 / dt_h))
    n_h = len(p) // sph
    hourly = p[: n_h * sph].reshape(n_h, sph).mean(axis=1)
    neg = hourly < 0.0
    out_h = np.zeros(n_h, bool)
    i = 0
    while i < n_h:
        if neg[i]:
            j = i
            while j < n_h and neg[j]:
                j += 1
            if j - i > 2:
                start = i if reg.negative_episode_scope == "whole_episode" else i + 2
                out_h[start:j] = True
            i = j
        else:
            i += 1
    out = np.zeros(len(p), bool)
    out[: n_h * sph] = np.repeat(out_h, sph)
    return out


def remuneration(
    scheme: SupportScheme, reg: RegulatoryConfig, y: YearInputs, year: int, kind: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """EUR/MWh paid for (a) PV energy exported directly and (b) energy discharged from the battery.

    merchant  both get the day-ahead price.
    ppa       PV gets the PPA price; battery output gets the PPA price only if the contract says so.
    fit       PV gets the tariff.
    fip       sliding premium: market price + (reference price - monthly PV reference market price).
              The premium is two-way (negative when the market is above the reference price) and
              is suspended by the negative price rule.
    Battery output of a supported park is paid at market prices through its own meter (REG-A-006),
    unless the regulatory switch is set to the alternative reading.
    """
    p = np.asarray(y.price, float)
    if kind is None:
        kind = scheme.kind
        if scheme.end_year is not None and year > scheme.end_year:
            kind = "merchant"
    if kind == "merchant":
        return p.copy(), p.copy()
    if kind == "ppa":
        r_pv = np.full_like(p, scheme.price_eur_mwh)
        r_dis = r_pv.copy() if scheme.ppa_covers_battery_output else p.copy()
        return r_pv, r_dis
    if kind == "fit":
        r_pv = np.full_like(p, scheme.price_eur_mwh)
        if reg.fit_exposed_to_negative_price_rule:
            r_pv = np.where(aid_suspended(p, y.dt_h, reg), np.minimum(p, 0.0), r_pv)
        r_dis = r_pv.copy() if reg.stored_energy_support == "support_price" else p.copy()
        return r_pv, r_dis
    if kind == "fip":
        if y.pv_ref_price_monthly is None:
            raise ValueError("FiP settlement needs the monthly PV reference market price (pv_ref_price_monthly)")
        premium = scheme.price_eur_mwh - np.asarray(y.pv_ref_price_monthly, float)
        if not reg.fip_two_way:
            premium = np.maximum(premium, 0.0)
        with_aid = np.where(aid_suspended(p, y.dt_h, reg), p, p + premium)
        r_pv = with_aid
        r_dis = with_aid.copy() if reg.stored_energy_support == "support_price" else p.copy()
        return r_pv, r_dis
    raise ValueError(kind)


def static_cap_mw(park: Park, reg: RegulatoryConfig, local_hour: np.ndarray, with_battery: bool) -> np.ndarray:
    """Permanent injection limit at the connection point per interval (MW), before curtailment orders.

    No battery: the park's existing cap (REG-A-010).
    11A: share of the installed PV power between 07:00 and 19:00 local time, full power at night (REG-A-009).
    11B: hourly profile with zero injection around noon (REG-A-011).
    """
    export = park.grid_export_limit_kw / 1000.0
    installed = park.installed_power_mw(reg.installed_power_basis)
    h = np.asarray(local_hour, float)
    if not with_battery:
        cap = export if park.static_injection_cap_frac >= 1.0 else min(export, park.static_injection_cap_frac * installed)
        return np.full(len(h), cap)
    if reg.station_category == "11A":
        lo, hi = reg.cap_11a_window_local
        day = (h >= lo) & (h < hi)
        return np.where(day, min(export, reg.cap_11a_frac_of_installed * installed), export)
    cap = np.full(len(h), export)
    for lo, hi, frac in reg.cap_11b_profile_local:
        cap = np.where((h >= lo) & (h < hi), frac * export, cap)
    return cap


def _pv_available(park: Park, coupling: str, pv_dc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PV power the dispatch may use and the power clipped at the inverter (both MW)."""
    p_inv = park.pv_ac_kw / 1000.0
    ac_potential = pv_dc * park.eta_inverter
    ac = np.minimum(ac_potential, p_inv)
    clipped_ac = ac_potential - ac
    if coupling == "dc":
        return pv_dc, clipped_ac
    return ac, clipped_ac


def simulate_pv_only(park: Park, reg: RegulatoryConfig, y: YearInputs, year: int) -> YearResult:
    """The park as it is today, no battery."""
    kind = effective_support_kind(park, reg, year, with_battery=False)
    r_pv, _ = remuneration(park.support, reg, y, year, kind)
    ac, clipped = _pv_available(park, "ac", y.pv_dc_mw)
    cap = np.minimum(static_cap_mw(park, reg, y.local_hour, False), ac * (1.0 - _order(park, reg, y)))
    e = np.where(r_pv >= 0, np.minimum(ac, cap), 0.0)
    lost = ac - e
    rev = float((e * r_pv).sum() * y.dt_h)
    comp = _compensation(park, reg, y, ac, r_pv)
    return YearResult(
        revenue_eur=rev + comp,
        pv_direct_mwh=float(e.sum() * y.dt_h),
        discharged_mwh=0.0,
        charged_pv_mwh=0.0,
        charged_grid_mwh=0.0,
        curtailed_mwh=float(lost.sum() * y.dt_h),
        clipped_mwh=float(clipped.sum() * y.dt_h),
        grid_cost_eur=0.0,
        fec=0.0,
        mean_soc_frac=0.0,
        marginal_value_power_eur_per_mw=0.0,
        marginal_value_energy_eur_per_mwh=0.0,
        net_export_mw=e,
    )


def _order(park: Park, reg: RegulatoryConfig, y: YearInputs) -> np.ndarray:
    """Curtailment order seen by this park as a share of its potential output."""
    if park.support.kind == "fit" and not reg.fit_subject_to_curtailment:
        return np.zeros_like(y.curtail_frac)
    return np.clip(np.asarray(y.curtail_frac, float) * park.curtailment_multiplier, 0.0, 1.0)


def _compensation(park, reg, y, ac, r_pv) -> float:
    """Payment for ordered curtailment, when the rules provide one (independent of the battery)."""
    if reg.curtailment_compensation_frac <= 0:
        return 0.0
    ordered = ac * _order(park, reg, y)
    return float((ordered * np.maximum(r_pv, 0.0)).sum() * y.dt_h * reg.curtailment_compensation_frac)


def simulate_with_battery(
    park: Park,
    reg: RegulatoryConfig,
    y: YearInputs,
    year: int,
    dp: DispatchParams,
    policy: str = "perfect",
    lookahead_days: int = 1,
    keep_traces: bool = False,
    lp_cache: dict | None = None,
    exact: bool = False,
) -> YearResult:
    """Simulate one year with a battery of given power and usable energy.

    The day is the commitment unit (the day-ahead market clears once per day). With
    lookahead_days = 1 the model solves two days and commits the first, so that energy can be
    carried across midnight when that pays.
    """
    if dp.power_mw <= 0 or dp.energy_usable_mwh <= 0:
        return simulate_pv_only(park, reg, y, year)
    kind = effective_support_kind(park, reg, year, with_battery=True, coupling=dp.coupling)
    r_pv, r_dis = remuneration(park.support, reg, y, year, kind)
    pv_avail, clipped = _pv_available(park, dp.coupling, y.pv_dc_mw)
    ac_potential = np.minimum(y.pv_dc_mw * park.eta_inverter, park.pv_ac_kw / 1000.0)
    export_limit = park.grid_export_limit_kw / 1000.0
    static = static_cap_mw(park, reg, y.local_hour, True)
    # a curtailment order limits the whole injection to the allowed share of the PV potential;
    # outside orders the station may use its static limit (also at night, from the battery)
    order = _order(park, reg, y)
    cap = np.where(order > 0, np.minimum(static, ac_potential * (1.0 - order)), static)
    if dp.coupling == "dc":
        cap = np.minimum(cap, park.pv_ac_kw / 1000.0)
    if reg.grid_charging_allowed:
        c_imp = y.price + reg.import_levies_eur_mwh
        imp_cap = np.full(len(y.price), min(export_limit, dp.power_mw))
    else:
        c_imp, imp_cap = None, None

    spd = y.steps_per_day
    n_days = len(y.price) // spd
    H = spd * (1 + lookahead_days)
    key = (H, y.dt_h, dp)
    cache = lp_cache if lp_cache is not None else {}
    if key not in cache:
        cache[key] = DispatchLP(H, y.dt_h, dp, cycle_block_steps=spd)
    lp = cache[key]

    T = n_days * spd
    e = np.zeros(T)
    cp = np.zeros(T)
    cg = np.zeros(T)
    d = np.zeros(T)
    soc = np.zeros(T)
    mv_p = 0.0
    mv_e = 0.0
    milp_days = 0
    cyc_in = 0.0
    s0 = 0.0

    def window(arr, a, fill):
        seg = arr[a : a + H]
        if len(seg) < H:  # last days of the year: pad with a neutral tail
            seg = np.concatenate([seg, np.full(H - len(seg), fill)])
        return seg

    if policy == "rules":
        return _simulate_rules(park, reg, y, year, dp, r_pv, r_dis, pv_avail, cap, clipped, keep_traces)

    for k in range(n_days):
        a = k * spd
        if policy == "perfect":
            rp, rd = window(r_pv, a, 0.0), window(r_dis, a, 0.0)
            ci = None if c_imp is None else window(c_imp, a, 1e6)
        elif policy == "persistence":
            # plan with yesterday's remuneration shape for all days of the window (today's is unknown
            # at bidding time); physical inputs (PV, orders) use the day's own values.
            b = max(0, a - spd)
            rp = np.tile(r_pv[b : b + spd], 1 + lookahead_days)
            rd = np.tile(r_dis[b : b + spd], 1 + lookahead_days)
            ci = None if c_imp is None else np.tile(c_imp[b : b + spd], 1 + lookahead_days)
        else:
            raise ValueError(f"unknown policy {policy}")
        res = lp.solve(
            window(pv_avail, a, 0.0),
            window(cap, a, export_limit),
            rp,
            rd,
            ci,
            None if imp_cap is None else window(imp_cap, a, 0.0),
            soc0=s0,
            soc_terminal=0.0,
            commit_steps=spd,
            exact=exact,
        )
        sl = slice(a, a + spd)
        e[sl], cp[sl], cg[sl], d[sl], soc[sl] = res.e, res.cp, res.cg, res.d, res.soc
        s0 = float(res.soc[-1])
        mv_p += float(res.dual_power.sum())
        mv_e += float(res.dual_energy.sum())
        milp_days += int(res.used_milp)
        cyc_in += res.cycled_within_interval_mwh

    out = _settle(park, reg, y, dp, r_pv, r_dis, e, cp, cg, d, soc, pv_avail, clipped, c_imp, mv_p, mv_e, milp_days, keep_traces, T)
    out.cycled_within_interval_mwh = cyc_in
    return out


def _settle(park, reg, y, dp, r_pv, r_dis, e, cp, cg, d, soc, pv_avail, clipped, c_imp, mv_p, mv_e, milp_days, keep, T):
    """Money and energy accounting at the TRUE remuneration, whatever prices the schedule was built on."""
    dt = y.dt_h
    k = dp.k
    rev_pv = float((e * r_pv[:T]).sum() * dt)
    rev_dis = float((d * k * r_dis[:T]).sum() * dt)
    grid_cost = float((cg * c_imp[:T]).sum() * dt) if c_imp is not None else 0.0
    ac = np.minimum(y.pv_dc_mw[:T] * park.eta_inverter, park.pv_ac_kw / 1000.0)
    comp = _compensation(park, reg, y, np.pad(ac, (0, len(y.price) - T)), r_pv)
    used_pv = dp.a_e * e + cp
    lost = np.clip(pv_avail[:T] - used_pv, 0.0, None)
    clipped_lost = clipped[:T] if dp.coupling == "ac" else np.zeros(T)
    if dp.coupling == "dc":
        # in DC coupling 'lost' is measured on the DC side; report it in AC-equivalent energy
        lost = lost * park.eta_inverter
    discharged = float(d.sum() * dt)
    fec = discharged / dp.energy_usable_mwh if dp.energy_usable_mwh > 0 else 0.0
    return YearResult(
        revenue_eur=rev_pv + rev_dis - grid_cost + comp,
        pv_direct_mwh=float(e.sum() * dt),
        discharged_mwh=discharged,
        charged_pv_mwh=float(cp.sum() * dt),
        charged_grid_mwh=float(cg.sum() * dt),
        curtailed_mwh=float(lost.sum() * dt),
        clipped_mwh=float(clipped_lost.sum() * dt),
        grid_cost_eur=grid_cost,
        fec=fec,
        mean_soc_frac=float(soc.mean() / dp.energy_usable_mwh) if dp.energy_usable_mwh > 0 else 0.0,
        marginal_value_power_eur_per_mw=mv_p,
        marginal_value_energy_eur_per_mwh=mv_e,
        soc=soc if keep else None,
        net_export_mw=(e + k * d - cg) if keep else None,
        d=d if keep else None,
        c=(cp + k * cg) if keep else None,
        milp_days=milp_days,
    )


def _simulate_rules(park, reg, y, year, dp, r_pv, r_dis, pv_avail, cap, clipped, keep):
    """Clock rule baseline, no optimisation.

    Charge from PV that cannot be exported (order, export limit, clipping in DC) and from all PV in
    the 6 cheapest-on-average daytime hours of the season; discharge at full power in the evening
    window (17:00 to 23:00 UTC+2 approx) until empty. Purely greedy and causal.
    """
    dt = y.dt_h
    spd = y.steps_per_day
    T = (len(y.price) // spd) * spd
    k = dp.k
    e = np.zeros(T)
    cp = np.zeros(T)
    d = np.zeros(T)
    soc = np.zeros(T)
    s = 0.0
    E, P = dp.energy_usable_mwh, dp.power_mw
    tod = (np.arange(T) % spd) * dt  # hour of day in UTC
    charge_window = (tod >= 8.0) & (tod < 13.0)  # 10:00 to 15:00 Greek winter time
    discharge_window = (tod >= 15.5) & (tod < 21.0)  # 17:30 to 23:00
    for t in range(T):
        pv = pv_avail[t]
        exportable = min(pv / dp.a_e, cap[t]) if r_pv[t] >= 0 else 0.0
        surplus = pv - exportable * dp.a_e
        want = pv if charge_window[t] else surplus
        room = (E - s) / (dp.eta_charge * dt)
        c = max(0.0, min(want, P, room))
        ee = max(0.0, min((pv - c) / dp.a_e, cap[t])) if r_pv[t] >= 0 else 0.0
        dd = 0.0
        if discharge_window[t] and r_dis[t] > 0 and c == 0.0:
            dd = max(0.0, min(P, s * dp.eta_discharge / dt, (cap[t] - ee) / k))
        s = s + dp.eta_charge * c * dt - dd * dt / dp.eta_discharge
        e[t], cp[t], d[t], soc[t] = ee, c, dd, s
    return _settle(park, reg, y, dp, r_pv, r_dis, e, cp, np.zeros(T), d, soc, pv_avail, clipped, None, 0.0, 0.0, 0, keep, T)
