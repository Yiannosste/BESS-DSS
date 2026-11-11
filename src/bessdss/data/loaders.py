"""Loaders for the processed market datasets (offline, no network access).

Every time-series loader returns a pandas object indexed by a timezone-aware
UTC ``DatetimeIndex`` named ``ts_utc`` that marks the START of each interval.
The files are produced by :mod:`bessdss.data.fetch_market`; provenance, known
gaps and licences are listed in ``registers/data_provenance_market.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def _path(name: str, processed_dir: Path | None) -> Path:
    path = (processed_dir or PROCESSED_DIR) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run 'python scripts/fetch_market_data.py' to download the market data."
        )
    return path


def _load_ts(name: str, processed_dir: Path | None) -> pd.DataFrame:
    """Read a parquet file and return it indexed by tz-aware UTC ``ts_utc``."""
    df = pd.read_parquet(_path(name, processed_dir))
    ts = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.drop(columns="ts_utc")
    df.index = pd.DatetimeIndex(ts, name="ts_utc")
    return df.sort_index()


def load_dam_prices(processed_dir: Path | None = None) -> pd.DataFrame:
    """Greek day-ahead price in native resolution.

    Columns: ``price_eur_mwh``, ``resolution_min`` (60 until 2025-09-30 22:00 UTC,
    15 afterwards), ``source``.  Before 2020-11-01 the series is the System
    Marginal Price of the former mandatory pool, afterwards the DAM clearing price.
    """
    return _load_ts("dam_price_gr.parquet", processed_dir)


def load_dam_prices_hourly(processed_dir: Path | None = None) -> pd.Series:
    """Hourly day-ahead price: the 15-minute part is averaged to hours.

    Convenience view only; the stored file keeps the native resolution.
    """
    price = load_dam_prices(processed_dir)["price_eur_mwh"]
    return price.resample("1h").mean().dropna().rename("price_eur_mwh")


def load_dam_prices_henex(processed_dir: Path | None = None) -> pd.DataFrame:
    """Official HEnEx DAM clearing prices (from 2020-11-01) used for the cross-check."""
    return _load_ts("dam_price_henex_gr.parquet", processed_dir)


def load_system(processed_dir: Path | None = None) -> pd.DataFrame:
    """Hourly Greek system data: load, generation per type, net imports (MW).

    Sign convention: generation and load positive; pumping and battery charging
    negative; net imports positive.
    """
    return _load_ts("system_gr.parquet", processed_dir)


def load_installed_capacity(processed_dir: Path | None = None) -> pd.DataFrame:
    """Installed capacity, long format.

    Columns: ``date`` (datetime, end of year or mid-month sample day),
    ``granularity`` (year or month), ``technology``, ``value``, ``unit``,
    ``scope`` and ``source``.  Sources have DIFFERENT scopes: filter on
    ``source``/``scope`` before comparing numbers.
    """
    df = pd.read_csv(_path("installed_capacity_gr.csv", processed_dir))
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_imbalance(processed_dir: Path | None = None) -> pd.DataFrame:
    """ADMIE 15-minute imbalance settlement prices, mFRR prices and balancing volumes."""
    return _load_ts("imbalance_gr.parquet", processed_dir)


def load_curtailment(processed_dir: Path | None = None) -> pd.DataFrame:
    """Scheduled RES curtailment ("Energy Surplus" of the ADMIE ISP runs), MW.

    ``curtail_sched_mw`` is the latest available ISP run per period; NaN means
    that the ISP files of that time did not report the quantity (not zero).
    Multiply by ``resolution_min / 60`` to obtain MWh.
    """
    return _load_ts("curtailment_gr.parquet", processed_dir)


def load_res_forecast(processed_dir: Path | None = None) -> pd.DataFrame:
    """Hourly day-ahead forecasts (PV, wind, load) next to the actual values."""
    return _load_ts("res_forecast_gr.parquet", processed_dir)


def load_fuel_prices(processed_dir: Path | None = None) -> pd.DataFrame:
    """Monthly TTF gas price (EUR/MWh) and EUA price (EUR/t).

    The index is the first instant of each month in UTC.
    """
    df = pd.read_csv(_path("fuel_prices.csv", processed_dir))
    idx = pd.to_datetime(df["month"], format="%Y-%m").dt.tz_localize("UTC")
    df.index = pd.DatetimeIndex(idx, name="ts_utc")
    return df.drop(columns="month")


def load_eua_auctions(processed_dir: Path | None = None) -> pd.DataFrame:
    """Individual EEX primary EUA auctions (date index in UTC, clearing price EUR/t)."""
    df = pd.read_csv(_path("eua_auctions_daily.csv", processed_dir))
    idx = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    df.index = pd.DatetimeIndex(idx, name="ts_utc")
    return df.drop(columns="date")


def load_neighbour_prices(processed_dir: Path | None = None, wide: bool = True) -> pd.DataFrame:
    """Day-ahead prices of neighbouring bidding zones.

    With ``wide`` (default) one column per zone, native timestamps (zones with
    hourly prices have NaN on the quarter-hour stamps of 15-minute zones).
    """
    df = _load_ts("neighbours_price.parquet", processed_dir)
    if not wide:
        return df
    return df.pivot_table(index=df.index, columns="bzn", values="price_eur_mwh", observed=True).sort_index()
