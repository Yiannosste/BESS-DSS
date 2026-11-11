"""Calibration of the long-term market engine on real Greek data.

Everything here is estimated from public history (see registers/data_provenance.yaml):
    * potential (pre-curtailment) national PV output, rebuilt from satellite/reanalysis irradiance,
      because the published "solar generation" is measured AFTER curtailment,
    * weather years (load shape, PV and wind capacity factors),
    * the conditional quantile surface of the normalised price and the library of rank days,
    * the 15-minute texture library,
    * the link between the monthly thermal price level and gas and CO2 prices,
    * the link between surplus depth and PV curtailment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bessdss.data import loaders
from bessdss.market.cqr import fit_surface, monthly_level
from bessdss.market.longterm import MarketCalibration, gas_marginal_cost

LOW_PRICE = 5.0  # EUR/MWh: at or below this an hour is treated as possibly curtailed


def national_ghi(nasa: pd.DataFrame) -> pd.Series:
    cols = [c for c in nasa.columns if c.endswith("ALLSKY_SFC_SW_DWN")]
    return nasa[cols].mean(axis=1).rename("ghi")


def fit_potential_pv(solar_mw: pd.Series, ghi: pd.Series, price: pd.Series, iters: int = 30) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Potential PV output (MW) from irradiance.

    Model: solar = c[year-month] * beta[month-of-year, hour] * GHI, fitted by alternating least
    squares only on hours that are surely not curtailed (price above LOW_PRICE). beta captures
    geometry (tilt, sun path) and c the growth of installed capacity.
    Returns (potential_mw, beta by cell, c by year-month).
    """
    df = pd.DataFrame({"solar": solar_mw, "ghi": ghi, "price": price}).dropna()
    df["ym"] = df.index.year * 100 + df.index.month
    df["cell"] = df.index.month * 100 + df.index.hour
    ok = (df.price > LOW_PRICE) & (df.ghi > 20)
    d = df[ok].copy()
    c = pd.Series(1.0, index=np.sort(df.ym.unique()))
    for _ in range(iters):
        d["x"] = d.ghi * d.ym.map(c)
        beta = (d.solar * d.x).groupby(d.cell).sum() / (d.x**2).groupby(d.cell).sum()
        d["x2"] = d.ghi * d.cell.map(beta)
        c = (d.solar * d.x2).groupby(d.ym).sum() / (d.x2**2).groupby(d.ym).sum()
        c = c.reindex(np.sort(df.ym.unique())).interpolate().bfill().ffill()
        norm = c.mean()
        c = c / norm
    pot = (df.ghi * df.cell.map(beta).fillna(0.0) * df.ym.map(c)).clip(lower=0.0)
    return pot.rename("pv_potential_mw"), beta, c


def _capacity_series(cap: pd.DataFrame, technology: str, scope_contains: str) -> pd.Series:
    d = cap[(cap.technology == technology) & (cap.granularity == "year") & cap.scope.str.contains(scope_contains)]
    s = pd.Series(d.value.to_numpy(float), index=pd.to_datetime(d.date.to_numpy(), utc=True)).sort_index()
    return s


def _epoch_s(idx: pd.DatetimeIndex) -> np.ndarray:
    """Seconds since epoch as float, independent of the index time unit (pandas 3 mixes units)."""
    return idx.as_unit("s").asi8.astype(float)


def _drop_feb29(df: pd.DataFrame) -> pd.DataFrame:
    return df[~((df.index.month == 2) & (df.index.day == 29))]


def build_calibration(
    window_start: str = "2025-01-01",
    weather_years: tuple[int, ...] = tuple(range(2015, 2024)),
    level_fit_start: str = "2023-01-01",
    seed: int = 0,
    processed_dir=None,
) -> tuple[MarketCalibration, dict]:
    """Estimate every piece of the market engine. Returns the calibration and a diagnostics dict."""
    sysd = loaders.load_system(processed_dir)
    sysd = sysd[[c for c in sysd.columns if c.endswith("_mw")]].resample("1h").mean()
    # short gaps in the public series (a few hours per year) are bridged by interpolation
    main = ["load_mw", "pv_mw", "wind_mw"]
    sysd[main] = sysd[main].interpolate(limit=6, limit_direction="both")
    price = loaders.load_dam_prices_hourly(processed_dir)
    price_native = loaders.load_dam_prices(processed_dir)
    fuel = loaders.load_fuel_prices(processed_dir)
    cap = loaders.load_installed_capacity(processed_dir)
    nasa = pd.read_parquet(loaders._path("nasa_power_points.parquet", processed_dir))
    ghi = national_ghi(nasa)
    diag: dict = {}

    # ---- potential PV and hidden curtailment
    pot, beta, c = fit_potential_pv(sysd["pv_mw"], ghi, price)
    df = sysd.join(pot, how="inner").join(price.rename("price"), how="inner").fillna(0.0)
    gap = (df.pv_potential_mw - df.pv_mw)
    low = df.price <= LOW_PRICE
    sunny = ghi.reindex(df.index) > 20
    unc = (~low) & sunny
    diag["potential_pv_r2_uncurtailed"] = float(
        1 - ((df.pv_mw - df.pv_potential_mw)[unc] ** 2).sum() / ((df.pv_mw[unc] - df.pv_mw[unc].mean()) ** 2).sum()
    )
    diag["hidden_pv_curtailment_twh"] = {
        int(y): round(float(gap[low & (df.index.year == y)].clip(lower=0).sum() / 1e6), 3) for y in range(2022, 2027)
    }
    diag["potential_pv_twh"] = {int(y): round(float(df.pv_potential_mw[df.index.year == y].sum() / 1e6), 2) for y in range(2022, 2027)}

    # ---- capacity factors per GW installed
    pv_cap = _capacity_series(cap, "pv", "IRENA")  # MW at year end
    wind_cap = _capacity_series(cap, "wind", "IRENA")
    t = df.index
    def cap_at(series: pd.Series) -> np.ndarray:
        return np.interp(_epoch_s(t), _epoch_s(series.index), series.to_numpy())
    pv_cap_t = cap_at(pv_cap)
    wind_cap_t = cap_at(wind_cap)
    # ratio of fitted scale c to official capacity: constant if the coverage of the solar series is constant
    ym = pd.Series(t.year * 100 + t.month, index=t)
    c_t = ym.map(c).to_numpy()
    k_by_year = pd.Series(c_t / pv_cap_t, index=t).groupby(t.year).median()
    diag["pv_scale_per_mw_by_year"] = {int(y): float(v) for y, v in k_by_year.items()}
    k_ref = float(k_by_year.loc[2021:2025].median())
    cell = pd.Series(t.month * 100 + t.hour, index=t)
    pv_cf = (ghi.reindex(t) * cell.map(beta).fillna(0.0) * k_ref).clip(lower=0.0)  # MW per MW installed
    wind_cf = (df.wind_mw / wind_cap_t).clip(0.0, 1.0)

    weather = {}
    for y in weather_years:
        sel = df.index.year == y
        if sel.sum() < 8700:
            continue
        w = pd.DataFrame(
            {"load_shape": df.load_mw[sel] / df.load_mw[sel].mean(), "pv_cf": pv_cf[sel], "wind_cf": wind_cf[sel]}
        )
        w = _drop_feb29(w).iloc[:8760]
        if len(w) == 8760 and w.notna().all().all():
            weather[int(y)] = w
    diag["weather_years"] = sorted(weather)
    diag["pv_cf_full_load_hours"] = {y: round(float(w.pv_cf.sum()), 0) for y, w in weather.items()}
    diag["wind_cf_mean"] = {y: round(float(w.wind_cf.mean()), 3) for y, w in weather.items()}

    # ---- state variable z and price surface on the calibration window
    sto = (df.hydro_pumping_mw + df.battery_charge_mw - df.battery_discharge_mw - df.hydro_pumped_gen_mw) / 1000.0
    z = ((df.load_mw - df.pv_potential_mw - df.wind_mw) / 1000.0 + sto).rename("z")
    M = monthly_level(df.price, z)
    M_h = pd.Series(M.reindex(df.index.to_period("M")).to_numpy(), index=df.index)
    win = df.index >= pd.Timestamp(window_start, tz="UTC")
    pi = (df.price / M_h)[win]
    zw = z[win]
    surface = fit_surface(zw.to_numpy(), pi.to_numpy())
    rng = np.random.default_rng(seed)
    u = pd.Series(surface.rank(zw.to_numpy(), pi.to_numpy(), rng), index=zw.index)
    days, months, weekend = [], [], []
    for day, v in u.groupby(u.index.floor("D")):
        if len(v) == 24:
            days.append(v.to_numpy())
            months.append(day.month)
            weekend.append(day.dayofweek >= 5)
    diag["rank_days"] = len(days)
    diag["window"] = [str(zw.index.min()), str(zw.index.max())]

    # ---- 15-minute texture (only exists since the 15-minute MTU went live)
    q = price_native[price_native.resolution_min == 15]["price_eur_mwh"]
    tex_days, tex_month = [], []
    if len(q):
        hourly_mean = q.groupby(q.index.floor("h")).transform("mean")
        lvl = pd.Series(M.reindex(q.index.to_period("M")).to_numpy(), index=q.index)
        dev = ((q - hourly_mean) / lvl).dropna()
        for day, v in dev.groupby(dev.index.floor("D")):
            if len(v) == 96:
                tex_days.append(v.to_numpy())
                tex_month.append(day.month)
    diag["texture_days"] = len(tex_days)

    # ---- thermal price level versus fuel prices
    f = fuel.copy()
    f.index = f.index.to_period("M") if not isinstance(f.index, pd.PeriodIndex) else f.index
    mc = pd.Series(gas_marginal_cost(f["ttf_eur_mwh"], f["eua_eur_t"]), index=f.index)
    both = pd.concat([M.rename("M"), mc.rename("mc")], axis=1).dropna()
    both = both[both.index >= pd.Period(level_fit_start, "M")]
    A = np.vstack([np.ones(len(both)), both.mc.to_numpy()]).T
    coef, *_ = np.linalg.lstsq(A, both.M.to_numpy(), rcond=None)
    fitted = A @ coef
    diag["level_fit"] = dict(
        intercept=float(coef[0]), slope=float(coef[1]),
        r2=float(1 - ((both.M - fitted) ** 2).sum() / ((both.M - both.M.mean()) ** 2).sum()),
        mape=float((np.abs(both.M - fitted) / both.M).mean()), n_months=int(len(both)),
    )
    ratio = pd.Series(both.M.to_numpy() / fitted, index=both.index)
    seas = ratio.groupby(ratio.index.month).mean().reindex(range(1, 13)).fillna(1.0).to_numpy()
    seas = 1.0 + 0.5 * (seas / seas.mean() - 1.0)  # shrink half-way to 1: under four years of data
    seas = seas / seas.mean()

    # ---- PV curtailment versus surplus depth, zero-price sunny hours of the window
    zero = (df.price <= 0.5) & win & sunny
    zc = z[zero].to_numpy()
    cg = (gap[zero] / 1000.0).clip(lower=0.0).to_numpy()
    if len(zc) > 50:
        B = np.vstack([np.ones(len(zc)), -zc]).T
        (a, b), *_ = np.linalg.lstsq(B, cg, rcond=None)
    else:
        a, b = 0.0, 0.0
    diag["curtailment_link"] = dict(a_gw=float(a), b=float(b), n=int(len(zc)), corr=float(np.corrcoef(zc, cg)[0, 1]) if len(zc) > 50 else None)

    calib = MarketCalibration(
        surface=surface,
        rank_days=np.array(days), rank_month=np.array(months), rank_weekend=np.array(weekend),
        texture_days=np.array(tex_days) if tex_days else np.zeros((0, 96)), texture_month=np.array(tex_month),
        weather=weather, level_intercept=float(coef[0]), level_slope=float(coef[1]), level_seasonal=seas,
        curt_a_gw=float(max(a, 0.0)), curt_b=float(max(b, 0.0)),
        meta=dict(
            window_start=window_start, built_on=str(pd.Timestamp.now(tz="UTC").date()), k_ref=k_ref,
            pv_capacity_source="IRENA", note="prices in nominal EUR of the calibration window",
        ),
    )
    # history table kept for validation and plots
    hist = pd.DataFrame({"z": z, "price": df.price, "M": M_h, "pv_potential_mw": df.pv_potential_mw, "pv_mw": df.pv_mw})
    diag["_history"] = hist
    return calib, diag
