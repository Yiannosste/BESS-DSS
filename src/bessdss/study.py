"""The study pipeline: from a park description to a personalised battery study.

Steps
    1. Park PV production for every weather year (PVGIS, optionally calibrated to measurements).
    2. Market years for every scenario x representative project year x weather year (cached).
    3. For a candidate design (legal category, power, energy): lifetime simulation with
       degradation feeding back into usable energy, giving yearly revenue with and without battery.
    4. Monte Carlo cash-flow paths (scenario, weather sequence, model error, capture, CAPEX) and
       project-finance metrics (NPV, IRR, DSCR).
    5. Sizing: the battery size is an OUTPUT (see bessdss.sizing).

Value decomposition kept all the way to the report:
    incremental revenue = conversion penalty (the park becomes an 11A/11B station and gets a
    static injection cap, loses support in 11B)  +  gross value of the battery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bessdss import assumptions as A
from bessdss.battery.model import BatterySpec, BlastLfpGr250Ah, LinearDegradation, YearUsage
from bessdss.config import Park, RegulatoryConfig, Study
from bessdss.dispatch.lp import DispatchParams
from bessdss.dispatch.simulate import (
    YearInputs,
    greek_local_hour,
    simulate_pv_only,
    simulate_with_battery,
)
from bessdss.finance.model import FinanceParams, FinanceResult, evaluate
from bessdss.market.library import ScenarioLibrary
from bessdss.pv.pvgis import calibrate_to_measured, fetch_pvgis_hourly, to_quarter_hour


def default_finance(study: Study) -> FinanceParams:
    """Finance parameters from the assumptions register, with the study's overrides on top."""
    p = FinanceParams(
        life_years=study.life_years,
        inflation=A.central("inflation"),
        discount_rate_nominal=A.central("discount_rate_nominal"),
        tax_rate=A.central("tax_rate"),
        depreciation_years=int(A.central("depreciation_years")),
        capex_energy_eur_per_kwh=A.central("capex_energy_eur_per_kwh"),
        capex_power_eur_per_kw=A.central("capex_power_eur_per_kw"),
        capex_fixed_eur=A.central("capex_fixed_eur"),
        opex_fixed_frac_of_capex=A.central("opex_fixed_frac_of_capex"),
        insurance_frac_of_capex=A.central("insurance_frac_of_capex"),
        aggregator_fee_frac_of_incremental_revenue=A.central("aggregator_fee_frac"),
        battery_cost_decline_per_year=A.central("battery_cost_decline_per_year"),
        decommissioning_frac_of_capex=A.central("decommissioning_frac_of_capex"),
        gearing=A.central("gearing"),
        debt_tenor_years=int(A.central("debt_tenor_years")),
        debt_rate=A.central("debt_rate"),
        target_dscr=A.central("target_dscr"),
    )
    for k, v in study.finance_overrides.items():
        if not hasattr(p, k):
            raise KeyError(f"unknown finance override '{k}'")
        setattr(p, k, v)
    return p


def _drop_feb29(s: pd.Series) -> pd.Series:
    return s[~((s.index.month == 2) & (s.index.day == 29))]


def park_pv_per_kwp(park: Park, weather_years: list[int], cache_dir: Path | None = None) -> tuple[dict, dict, dict]:
    """Hourly and 15-minute DC power per kWp for each weather year, plus a provenance report."""
    inv_loss = (1.0 - park.eta_inverter) * 100.0
    dc_loss = max(0.0, park.system_loss_pct - inv_loss)  # PVGIS 'loss' covers the whole chain; the inverter is modelled here
    df = fetch_pvgis_hourly(park.lat, park.lon, park.tilt_deg, park.azimuth_deg, park.tracking, dc_loss, cache_dir=cache_dir)
    series = df["p_kw_per_kwp"]
    report = {"source": "PVGIS v5.3 (SARAH-3, ERA5), modelled", "years_available": [int(series.index.year.min()), int(series.index.year.max())]}
    if park.measured_production_csv:
        meas = pd.read_csv(park.measured_production_csv, parse_dates=["ts_utc"]).set_index("ts_utc")["power_kw"]
        if meas.index.tz is None:
            meas.index = meas.index.tz_localize("UTC")
        per_kwp = meas / park.pv_dc_kwp / park.eta_inverter
        series, cal = calibrate_to_measured(series, per_kwp)
        report["calibration_to_measured"] = cal
        report["source"] += ", calibrated to the owner's measured production"
    ann = (series.groupby(series.index.year).sum()).astype(float)
    report["yield_kwh_per_kwp_by_year"] = {int(k): round(float(v), 1) for k, v in ann.items()}
    report["yield_p50"] = float(ann.median())
    report["yield_p90"] = float(ann.quantile(0.10))
    hourly, quarter = {}, {}
    for y in weather_years:
        full = series[series.index.year == y]
        s = _drop_feb29(full)
        if len(s) < 8760:
            continue
        hourly[y] = s.iloc[:8760].to_numpy(float)
        quarter[y] = _drop_feb29(to_quarter_hour(full)).to_numpy(float)[:35040]
    return hourly, quarter, report


@dataclass
class StudyContext:
    study: Study
    lib: ScenarioLibrary
    fin: FinanceParams
    scen_names: list[str]
    rep_years: list[int]
    years: list[int]  # calendar years of operation
    weather_years: list[int]
    pv_h: dict
    pv_q: dict
    pv_report: dict
    _inputs: dict = field(default_factory=dict)
    _base: dict = field(default_factory=dict)

    @classmethod
    def build(cls, study: Study, lib: ScenarioLibrary | None = None, weather_years: list[int] | None = None) -> StudyContext:
        lib = lib or ScenarioLibrary()
        wys = weather_years or lib.weather_years
        pv_h, pv_q, rep = park_pv_per_kwp(study.park, wys)
        wys = [w for w in wys if w in pv_h]
        N = study.life_years
        years = [study.cod_year + i for i in range(N)]
        offs = sorted({0, 2, 4, 7, 10, N - 1} & set(range(N)))
        rep_years = [study.cod_year + o for o in offs]
        names = study.scenarios or [n for n, sc in lib.scenarios.items() if sc.weight > 0]
        return cls(study, lib, default_finance(study), names, rep_years, years, wys, pv_h, pv_q, rep)

    @property
    def scen_weights(self) -> np.ndarray:
        w = np.array([self.lib.scenarios[n].weight for n in self.scen_names], float)
        return w / w.sum()

    def precompute_market(self) -> None:
        for s in self.scen_names:
            for y in self.rep_years:
                for w in self.weather_years:
                    self.lib.market_year(s, y, w)

    def inputs(self, s: str, y: int, w: int, res: str) -> YearInputs:
        key = (s, y, w, res)
        if key in self._inputs:
            return self._inputs[key]
        park = self.study.park
        my = self.lib.market_year(s, y, w)
        ageing = (1.0 - park.pv_degradation_per_year) ** max(0, y - park.commissioning_year)
        scale = park.pv_dc_kwp / 1000.0 * ageing
        if res == "15min":
            yi = YearInputs(
                dt_h=0.25, pv_dc_mw=self.pv_q[w] * scale, price=my.price_qh,
                curtail_frac=np.repeat(my.pv_curtail_frac_h, 4),
                pv_ref_price_monthly=np.repeat(my.pv_ref_price_monthly, 4),
                local_hour=greek_local_hour(y, 35040, 0.25),
            )
        else:
            yi = YearInputs(
                dt_h=1.0, pv_dc_mw=self.pv_h[w] * scale, price=my.price_h, curtail_frac=my.pv_curtail_frac_h,
                pv_ref_price_monthly=my.pv_ref_price_monthly, local_hour=greek_local_hour(y, 8760, 1.0),
            )
        if len(self._inputs) > 400:
            self._inputs.clear()
        self._inputs[key] = yi
        return yi

    def baseline(self, s: str, y: int, w: int, res: str, reg: RegulatoryConfig):
        """The park as it is today (no battery, its own contract and cap)."""
        key = (s, y, w, res, reg.fit_subject_to_curtailment, reg.negative_price_rule, reg.negative_episode_scope, reg.curtailment_compensation_frac)
        if key not in self._base:
            self._base[key] = simulate_pv_only(self.study.park, reg, self.inputs(s, y, w, res), y)
        return self._base[key]


@dataclass
class DesignResult:
    category: str
    power_mw: float
    energy_mwh: float  # nameplate
    usable_mwh: float  # at beginning of life
    coupling: str
    scen_names: list[str]
    rep_years: list[int]
    weather_years: list[int]
    years: list[int]
    gross: np.ndarray  # (S, R, W) battery gross value versus the converted park without battery, EUR real
    penalty: np.ndarray  # (S, R, W) conversion penalty (<= 0), EUR real
    base_rev: np.ndarray  # (S, R, W) revenue of the park today
    discharged: np.ndarray  # (S, R, W) MWh
    fec: np.ndarray  # (S, R, W) nameplate full cycles
    lost_base: np.ndarray  # (S, R, W) PV energy lost today (orders, negative prices, export limit)
    lost_with: np.ndarray  # (S, R, W) PV energy lost with the battery
    mv_power: np.ndarray  # (S, R) EUR per MW per year (mean over weather years)
    mv_energy: np.ndarray  # (S, R) EUR per usable MWh per year
    soh: np.ndarray  # (S, N+1)
    aux_cost: np.ndarray  # (S, R) EUR real per year
    cycled_in: np.ndarray  # (S, R, W) MWh cycled inside single intervals (rule exploitation diagnostic)
    resolution: str
    policy: str

    def interp_years(self, arr: np.ndarray) -> np.ndarray:
        """(S, R, ...) -> (S, N, ...) linear interpolation between representative years."""
        yrs = np.array(self.years, float)
        rep = np.array(self.rep_years, float)
        out = np.empty((arr.shape[0], len(yrs)) + arr.shape[2:])
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        o = out.reshape(arr.shape[0], len(yrs), -1)
        for i in range(flat.shape[0]):
            for j in range(flat.shape[2]):
                o[i, :, j] = np.interp(yrs, rep, flat[i, :, j])
        return out


def _degradation_model(study: Study):
    b = study.battery
    if b.degradation_model == "linear":
        return LinearDegradation()
    return BlastLfpGr250Ah(pack_life_factor=b.pack_life_factor)


def marginal_cycle_cost(study: Study, fin: FinanceParams) -> float:
    """Baseline degradation cost per MWh discharged: value of the capacity fade caused by one cycle.

    capacity fade per full cycle (mid-life, from the life model) x replacement value of the energy
    block / usable share of life (1 - end-of-life SoH), per MWh discharged in one cycle.
    """
    b = study.battery
    model = _degradation_model(study)
    usage = [YearUsage(fec=350.0, mean_soc=0.5, mean_dod=0.9, cell_temp_c=b.cell_temp_c)] * study.life_years
    fade_per_fec = model.marginal_fade_per_fec(usage, study.life_years // 2)
    value_per_mwh_nameplate = fin.capex_energy_eur_per_kwh * 1000.0
    mwh_per_fec = (b.soc_max - b.soc_min) * np.sqrt(b.round_trip_efficiency_ac)
    return float(fade_per_fec * value_per_mwh_nameplate / (1.0 - b.eol_soh) / mwh_per_fec)


def run_design(
    ctx: StudyContext,
    category: str,
    power_mw: float,
    energy_mwh: float,
    resolution: str = "hourly",
    policy: str = "perfect",
    lookahead_days: int = 0,
    reg: RegulatoryConfig | None = None,
    weather_years: list[int] | None = None,
    cycle_cost_eur_mwh: float | None = None,
    scen_names: list[str] | None = None,
) -> DesignResult:
    """Lifetime simulation of one design in every scenario, representative year and weather year."""
    study = ctx.study
    park, b = study.park, study.battery
    reg = (reg or study.regulatory).model_copy(update={"station_category": category})
    wys = weather_years or ctx.weather_years
    names = scen_names or ctx.scen_names
    S, R, W, N = len(names), len(ctx.rep_years), len(wys), study.life_years
    eta = float(np.sqrt(b.round_trip_efficiency_ac))
    window = b.soc_max - b.soc_min
    c_deg = marginal_cycle_cost(study, ctx.fin) if cycle_cost_eur_mwh is None else cycle_cost_eur_mwh
    model = _degradation_model(study)
    spec = BatterySpec(power_mw, energy_mwh, b.soc_min, b.soc_max, eta, eta, aux_kw_per_mwh=b.aux_kw_per_mwh, coupling=b.coupling)
    zeros = lambda *shape: np.zeros(shape)  # noqa: E731
    gross, penalty, base_rev = zeros(S, R, W), zeros(S, R, W), zeros(S, R, W)
    discharged, fec = zeros(S, R, W), zeros(S, R, W)
    lost_base, lost_with, cycled_in = zeros(S, R, W), zeros(S, R, W), zeros(S, R, W)
    mv_p, mv_e, aux_cost = zeros(S, R), zeros(S, R), zeros(S, R)
    soh_all = np.ones((S, N + 1))
    lp_cache: dict = {}
    for si, s in enumerate(names):
        usage: list[YearUsage] = []
        soh = 1.0
        last = YearUsage(fec=0.0)
        for t, y in enumerate(ctx.years):
            if y in ctx.rep_years:
                ri = ctx.rep_years.index(y)
                dp = DispatchParams(
                    power_mw=power_mw, energy_usable_mwh=energy_mwh * window * soh, eta_charge=eta, eta_discharge=eta,
                    coupling=b.coupling, eta_inverter=park.eta_inverter, degradation_cost_eur_mwh=c_deg,
                    max_cycles_per_day=(b.max_cycles_per_year / 365.0) if b.max_cycles_per_year else None,
                )
                f_acc, soc_acc, price_acc = [], [], []
                for wi, w in enumerate(wys):
                    yi = ctx.inputs(s, y, w, resolution)
                    base = ctx.baseline(s, y, w, resolution, reg)
                    # the converted station with a battery of zero size: isolates the conversion penalty
                    conv = _converted_no_battery(park, reg, yi, y, b.coupling)
                    res = simulate_with_battery(park, reg, yi, y, dp, policy=policy, lookahead_days=lookahead_days, lp_cache=lp_cache)
                    base_rev[si, ri, wi] = base.revenue_eur
                    penalty[si, ri, wi] = conv.revenue_eur - base.revenue_eur
                    gross[si, ri, wi] = res.revenue_eur - conv.revenue_eur
                    discharged[si, ri, wi] = res.discharged_mwh
                    f = res.discharged_mwh / eta / energy_mwh if energy_mwh > 0 else 0.0
                    fec[si, ri, wi] = f
                    lost_base[si, ri, wi] = base.curtailed_mwh
                    lost_with[si, ri, wi] = res.curtailed_mwh
                    cycled_in[si, ri, wi] = res.cycled_within_interval_mwh
                    mv_p[si, ri] += res.marginal_value_power_eur_per_mw / W
                    mv_e[si, ri] += res.marginal_value_energy_eur_per_mwh / W
                    f_acc.append(f)
                    soc_acc.append(res.mean_soc_frac)
                    price_acc.append(float(np.maximum(yi.price, 0).mean()))
                mean_soc = b.soc_min + float(np.mean(soc_acc)) * window * soh
                f_mean = float(np.mean(f_acc))
                last = YearUsage(fec=f_mean, mean_soc=mean_soc, mean_dod=window * min(1.0, f_mean / 365.0 + 0.3), cell_temp_c=b.cell_temp_c)
                aux_cost[si, ri] = spec.aux_energy_mwh_per_year() * (float(np.mean(price_acc)) + A.central("aux_energy_charge_eur_mwh"))
            usage.append(last)
            soh = float(model.simulate(usage)[-1])
            soh_all[si, t + 1] = soh
    return DesignResult(
        category=category, power_mw=power_mw, energy_mwh=energy_mwh, usable_mwh=energy_mwh * window, coupling=b.coupling,
        scen_names=names, rep_years=ctx.rep_years, weather_years=wys, years=ctx.years,
        gross=gross, penalty=penalty, base_rev=base_rev, discharged=discharged, fec=fec, lost_base=lost_base,
        lost_with=lost_with, mv_power=mv_p, mv_energy=mv_e, soh=soh_all, aux_cost=aux_cost, cycled_in=cycled_in,
        resolution=resolution, policy=policy,
    )


def _converted_no_battery(park: Park, reg: RegulatoryConfig, yi: YearInputs, year: int, coupling: str):
    """PV-only revenue of the park AFTER conversion to the storage category (cap and support of that
    category) but with a battery that does nothing. Not a legal design, only an accounting step."""
    from bessdss.dispatch.simulate import (
        YearResult,
        _order,
        _pv_available,
        effective_support_kind,
        remuneration,
        static_cap_mw,
    )

    kind = effective_support_kind(park, reg, year, with_battery=True, coupling=coupling)
    r_pv, _ = remuneration(park.support, reg, yi, year, kind)
    ac, clipped = _pv_available(park, "ac", yi.pv_dc_mw)
    cap = np.minimum(static_cap_mw(park, reg, yi.local_hour, True), ac * (1.0 - _order(park, reg, yi)))
    e = np.where(r_pv >= 0, np.minimum(ac, cap), 0.0)
    return YearResult(
        revenue_eur=float((e * r_pv).sum() * yi.dt_h), pv_direct_mwh=float(e.sum() * yi.dt_h), discharged_mwh=0.0,
        charged_pv_mwh=0.0, charged_grid_mwh=0.0, curtailed_mwh=float((ac - e).sum() * yi.dt_h),
        clipped_mwh=float(clipped.sum() * yi.dt_h), grid_cost_eur=0.0, fec=0.0, mean_soc_frac=0.0,
        marginal_value_power_eur_per_mw=0.0, marginal_value_energy_eur_per_mwh=0.0,
    )


# ----------------------------------------------------------------------------- cash-flow paths
@dataclass
class PathSet:
    """Monte Carlo sample of the uncertain factors (shared by all designs for fair comparison)."""

    scen_idx: np.ndarray  # (n,)
    weather_idx: np.ndarray  # (n, N)
    capture: np.ndarray  # (n,)
    model_err: np.ndarray  # (n,)
    capex_mult: np.ndarray  # (n,)

    @classmethod
    def sample(cls, ctx: StudyContext, n_paths: int = 2000, seed: int = 1, n_weather: int | None = None) -> PathSet:
        rng = np.random.default_rng(seed)
        W = n_weather or len(ctx.weather_years)
        lo, hi = A.low_high("forecast_capture_factor")
        clo, chi = A.low_high("capex_uncertainty_multiplier")
        return cls(
            scen_idx=rng.choice(len(ctx.scen_names), size=n_paths, p=ctx.scen_weights),
            weather_idx=rng.integers(0, W, size=(n_paths, ctx.study.life_years)),
            capture=rng.triangular(lo, A.central("forecast_capture_factor"), hi, size=n_paths),
            model_err=np.exp(rng.normal(0.0, A.central("spread_model_error_sigma"), size=n_paths)),
            capex_mult=rng.triangular(clo, 1.0, chi, size=n_paths),
        )


@dataclass
class DesignEconomics:
    design: DesignResult
    fin: FinanceResult
    revenue_real: np.ndarray  # (n, N) incremental revenue per path, EUR real
    extra_capex: float

    def metrics(self, target_dscr: float | None = None) -> dict:
        f = self.fin
        q = lambda a, p: float(np.nanpercentile(a, p))  # noqa: E731
        return dict(
            category=self.design.category, power_mw=self.design.power_mw, energy_mwh=self.design.energy_mwh,
            usable_mwh=self.design.usable_mwh, duration_h=self.design.usable_mwh / self.design.power_mw,
            capex_eur=float(np.median(f.capex_nominal)),
            npv_mean=float(f.npv.mean()), npv_p50=q(f.npv, 50), npv_p90=q(f.npv, 10), npv_p10=q(f.npv, 90),
            prob_npv_negative=float((f.npv < 0).mean()),
            cvar10_npv=float(np.sort(f.npv)[: max(1, len(f.npv) // 10)].mean()),
            irr_p50=q(f.irr_project, 50), irr_p90=q(f.irr_project, 10),
            irr_equity_p50=q(f.irr_equity, 50),
            payback_p50=q(f.payback_years, 50) if np.isfinite(f.payback_years).any() else float("nan"),
            prob_no_payback=float(np.isnan(f.payback_years).mean()),
            dscr_min_p50=q(f.dscr_min, 50), dscr_min_p90=q(f.dscr_min, 10),
            prob_dscr_below_1=float((f.dscr_min < 1.0).mean()),
            lcos_p50=q(f.lcos_eur_mwh, 50),
            revenue_y1_p50=q(self.revenue_real[:, 0], 50), revenue_y1_p90=q(self.revenue_real[:, 0], 10),
        )


def guarantee_cost(study: Study) -> float:
    """Bank fee for the connection guarantee letter (the letter itself is returned)."""
    kw = study.park.grid_export_limit_kw
    rate = 0.0
    for upper, eur in study.regulatory.guarantee_eur_per_kw_tiers:
        if kw <= upper:
            rate = eur
            break
    return kw * rate * A.central("guarantee_fee_rate")


def economics(ctx: StudyContext, d: DesignResult, paths: PathSet, fin: FinanceParams | None = None) -> DesignEconomics:
    """Turn a simulated design into cash-flow paths and finance metrics."""
    fin = fin or ctx.fin
    gross = d.interp_years(d.gross)  # (S, N, W)
    pen = d.interp_years(d.penalty)
    dis = d.interp_years(d.discharged)
    aux = d.interp_years(d.aux_cost)  # (S, N)
    n, N = paths.weather_idx.shape
    yy = np.arange(N)[None, :]
    wi = paths.weather_idx % gross.shape[2]
    si = paths.scen_idx[:, None]
    g = gross[si, yy, wi]
    p = pen[si, yy, wi]
    rev = p + g * (paths.capture * paths.model_err)[:, None]
    disch = dis[si, yy, wi]
    extra = guarantee_cost(ctx.study)
    res = evaluate(
        fin, d.power_mw, d.energy_mwh, rev, discharged_mwh=disch, extra_opex_real=aux[paths.scen_idx],
        capex_multiplier=paths.capex_mult, extra_capex_eur=extra,
    )
    return DesignEconomics(design=d, fin=res, revenue_real=rev, extra_capex=extra)
