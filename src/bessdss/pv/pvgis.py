"""PV production from PVGIS (JRC), many historical weather years, per kWp.

PVGIS hourly output is MODELLED from satellite irradiance (SARAH-3) and reanalysis temperature
(ERA5). It is not a measurement of any real park. Its strength is the long record (2005 to 2023)
which gives the year-to-year variability needed for a "bad year" analysis.

The module also
  * resamples hourly power to 15-minute steps (energy-conserving interpolation, labelled modelled),
  * calibrates the modelled series to measured production when the owner provides it.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
PVGIS_FIRST_YEAR = 2005
PVGIS_LAST_YEAR = 2023
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "cache" / "pvgis"


def fetch_pvgis_hourly(
    lat: float,
    lon: float,
    tilt_deg: float = 25.0,
    azimuth_deg: float = 0.0,
    tracking: str = "fixed",
    system_loss_pct: float = 14.0,
    start_year: int = PVGIS_FIRST_YEAR,
    end_year: int = PVGIS_LAST_YEAR,
    cache_dir: Path | None = None,
    timeout_s: int = 180,
) -> pd.DataFrame:
    """Hourly PV output per kWp installed (kW/kWp, before any inverter clipping) plus weather.

    azimuth_deg follows the PVGIS convention: 0 = south, 90 = west, -90 = east.
    tracking: "fixed" or "single_axis" (horizontal north-south axis).
    Returns a DataFrame indexed by UTC hour start with columns p_kw_per_kwp, g_poa_wm2, t2m_c.
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "startyear": start_year,
        "endyear": end_year,
        "pvcalculation": 1,
        "peakpower": 1,
        "loss": system_loss_pct,
        "outputformat": "json",
    }
    if tracking == "single_axis":
        params["trackingtype"] = 1  # single horizontal axis aligned north-south
    else:
        params["angle"] = tilt_deg
        params["aspect"] = azimuth_deg
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    f = cache_dir / f"pvgis_{key}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(PVGIS_URL, params=params, timeout=timeout_s)
            if r.status_code == 200:
                break
            last_err = RuntimeError(f"PVGIS HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as exc:  # network hiccup: retry with backoff
            last_err = exc
        time.sleep(2**attempt)
    else:
        raise RuntimeError(f"PVGIS request failed: {last_err}")
    js = r.json()
    hourly = pd.DataFrame(js["outputs"]["hourly"])
    # PVGIS stamps each hour at the satellite scan minute (e.g. 10:09). We keep the hour start.
    ts = pd.to_datetime(hourly["time"], format="%Y%m%d:%H%M", utc=True).dt.floor("h")
    df = pd.DataFrame(
        {
            "p_kw_per_kwp": hourly["P"].to_numpy() / 1000.0,
            "g_poa_wm2": hourly["G(i)"].to_numpy(),
            "t2m_c": hourly["T2m"].to_numpy(),
        },
        index=pd.DatetimeIndex(ts, name="ts_utc"),
    )
    df.attrs["radiation_db"] = js["inputs"]["meteo_data"]["radiation_db"]
    df.to_parquet(f)
    return df


def to_quarter_hour(hourly: pd.Series) -> pd.Series:
    """Energy-conserving resampling of hourly mean power to 15-minute mean power.

    Linear interpolation between hour centres gives a smooth shape, then each hour's four values
    are rescaled so that their mean equals the hourly value. This keeps the daily energy exact
    but it does NOT add real intra-hour cloud variability (documented limitation).
    """
    s = hourly.astype(float)
    idx15 = pd.date_range(s.index[0], s.index[-1] + pd.Timedelta(minutes=45), freq="15min", tz="UTC")
    centres = s.copy()
    centres.index = s.index + pd.Timedelta(minutes=30)
    target = idx15 + pd.Timedelta(minutes=7.5)
    x = np.interp(
        target.as_unit("s").asi8.astype(float),
        centres.index.as_unit("s").asi8.astype(float),
        centres.to_numpy(),
    )
    x = np.clip(x, 0.0, None).reshape(-1, 4)
    hourly_vals = s.to_numpy()
    m = x.mean(axis=1)
    scale = np.divide(hourly_vals, m, out=np.zeros_like(m), where=m > 1e-12)
    x = x * scale[:, None]
    # hours with production but zero interpolated mean (isolated spikes): fall back to flat
    flat = (m <= 1e-12) & (hourly_vals > 0)
    x[flat] = hourly_vals[flat, None]
    return pd.Series(x.reshape(-1), index=idx15, name=s.name)


def calibrate_to_measured(modelled: pd.Series, measured: pd.Series) -> tuple[pd.Series, dict]:
    """Scale modelled production so that monthly energy matches the owner's measured production.

    Both series are power in the same unit on a UTC index (any resolution). One factor per calendar
    month is fitted on the overlapping period and applied to every modelled year. Returns the
    calibrated series and a small report (factors, overlap, bias before and after).
    """
    mh = modelled.resample("1h").mean()
    yh = measured.resample("1h").mean()
    both = pd.concat([mh.rename("mod"), yh.rename("meas")], axis=1).dropna()
    if len(both) < 24 * 60:
        raise ValueError("Need at least about two months of overlapping measured data to calibrate.")
    g = both.groupby(both.index.month).sum()
    factors = (g["meas"] / g["mod"]).replace([np.inf, -np.inf], np.nan)
    overall = both["meas"].sum() / both["mod"].sum()
    factors = factors.reindex(range(1, 13)).fillna(overall).clip(0.5, 1.5)
    out = modelled * modelled.index.month.map(factors).to_numpy()
    report = {
        "overlap_hours": int(len(both)),
        "overall_ratio_measured_to_modelled": float(overall),
        "monthly_factors": {int(k): float(v) for k, v in factors.items()},
        "hourly_correlation": float(both["mod"].corr(both["meas"])),
    }
    return out, report


def annual_yield_stats(p_kw_per_kwp: pd.Series) -> pd.DataFrame:
    """Specific yield per calendar year (kWh/kWp) with its deviation from the long-term mean."""
    dt_h = (p_kw_per_kwp.index[1] - p_kw_per_kwp.index[0]).total_seconds() / 3600.0
    y = (p_kw_per_kwp * dt_h).groupby(p_kw_per_kwp.index.year).sum()
    return pd.DataFrame({"yield_kwh_per_kwp": y, "deviation_pct": 100 * (y / y.mean() - 1)})
