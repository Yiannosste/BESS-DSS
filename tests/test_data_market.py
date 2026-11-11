"""Offline checks of the processed market datasets.

The tests read only ``data/processed`` (no network).  Each test is skipped when
its file has not been generated yet (``python scripts/fetch_market_data.py``).
Numbers that are asserted for closed periods (row counts, gap counts, counts of
negative prices) are the ones documented in
``registers/data_provenance_market.yaml`` on 2026-11-10.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bessdss.data import loaders

P = loaders.PROCESSED_DIR
GO_LIVE_15MIN = pd.Timestamp("2025-09-30 22:00", tz="UTC")  # 2025-10-01 00:00 CEST
TARGET_MODEL = pd.Timestamp("2020-10-31 23:00", tz="UTC")  # 2020-11-01 00:00 CET


def _need(name: str) -> None:
    if not (P / name).exists():
        pytest.skip(f"{name} not generated (run scripts/fetch_market_data.py)")


def _assert_utc_index(df: pd.DataFrame | pd.Series) -> None:
    idx = df.index
    assert isinstance(idx, pd.DatetimeIndex)
    assert str(idx.tz) == "UTC"
    assert idx.name == "ts_utc"
    assert idx.is_monotonic_increasing
    assert idx.is_unique


def _hours_per_year(df: pd.DataFrame) -> pd.Series:
    return (df["resolution_min"] / 60.0).groupby(df.index.year).sum()


# --------------------------------------------------------------------------- DAM price
@pytest.fixture(scope="module")
def dam() -> pd.DataFrame:
    _need("dam_price_gr.parquet")
    return loaders.load_dam_prices()


def test_dam_schema(dam: pd.DataFrame) -> None:
    _assert_utc_index(dam)
    assert list(dam.columns) == ["price_eur_mwh", "resolution_min", "source"]
    assert dam["price_eur_mwh"].dtype == np.float64
    assert dam["price_eur_mwh"].notna().all()
    assert set(dam["resolution_min"].unique()) == {15, 60}
    assert dam.index.min() == pd.Timestamp("2015-01-01 00:00", tz="UTC")
    assert dam.index.max() >= pd.Timestamp("2026-11-10 22:45", tz="UTC")


def test_dam_resolution_switch(dam: pd.DataFrame) -> None:
    """Hourly until 2025-09-30, 15 minutes from delivery day 2025-10-01 (CET)."""
    assert (dam.loc[dam.index < GO_LIVE_15MIN, "resolution_min"] == 60).all()
    assert (dam.loc[dam.index >= GO_LIVE_15MIN, "resolution_min"] == 15).all()
    assert GO_LIVE_15MIN in dam.index


def test_dam_no_gaps(dam: pd.DataFrame) -> None:
    """Every interval starts exactly where the previous one ends (documented: 0 gaps)."""
    step = dam.index.to_series().diff().dt.total_seconds().div(60.0).iloc[1:]
    expected = dam["resolution_min"].shift(1).iloc[1:].astype(float)
    assert int((step.to_numpy() != expected.to_numpy()).sum()) == 0


def test_dam_complete_closed_years(dam: pd.DataFrame) -> None:
    hours = _hours_per_year(dam)
    for year in range(2015, 2026):
        expected = 8784 if year % 4 == 0 else 8760
        assert hours[year] == expected, year
    rows = dam.groupby(dam.index.year).size()
    assert rows[2024] == 8784
    assert rows[2025] == 6552 + 2208 * 4  # 273 hourly days (UTC) then 92 days of 15 minutes


def test_dam_dst_days(dam: pd.DataFrame) -> None:
    """In market time (CET/CEST) the clock-change days have 23 and 25 hours."""
    local = dam.tz_convert("Europe/Berlin")
    hours = (local["resolution_min"] / 60.0).groupby(local.index.date).sum()
    hours.index = pd.to_datetime(hours.index)
    full = hours.iloc[1:-1]  # first and last day are cut by the UTC file boundaries
    assert set(full.unique()) <= {23.0, 24.0, 25.0}
    for year in range(2015, 2026):
        y = full[full.index.year == year]
        assert (y == 23.0).sum() == 1 and (y == 25.0).sum() == 1, year
    assert full.loc["2025-03-30"] == 23.0 and full.loc["2025-10-26"] == 25.0


def test_dam_plausible_range(dam: pd.DataFrame) -> None:
    price = dam["price_eur_mwh"]
    assert price.min() >= -500.0  # harmonised minimum clearing price of SDAC
    assert price.max() <= 4000.0
    assert price.min() == pytest.approx(-50.0)  # documented observed minimum
    assert price.max() == pytest.approx(942.0)  # documented observed maximum (2022)
    yearly = (price * dam["resolution_min"]).groupby(dam.index.year).sum() / dam["resolution_min"].groupby(dam.index.year).sum()
    assert 40.0 < yearly[2016] < 46.0
    assert 270.0 < yearly[2022] < 290.0


def test_dam_negative_and_zero_counts(dam: pd.DataFrame) -> None:
    """Documented interval counts for closed years (native resolution)."""
    price = dam["price_eur_mwh"]
    neg = (price < 0).groupby(dam.index.year).sum()
    zero = (price == 0).groupby(dam.index.year).sum()
    documented_neg = {2015: 0, 2016: 0, 2017: 0, 2018: 0, 2019: 0, 2020: 2, 2021: 5, 2022: 1, 2023: 0, 2024: 11, 2025: 177}
    documented_zero = {2015: 5, 2016: 10, 2017: 7, 2018: 25, 2019: 13, 2020: 21, 2021: 9, 2022: 4, 2023: 24, 2024: 18, 2025: 357}
    for year, count in documented_neg.items():
        assert int(neg[year]) == count, f"negative {year}"
    for year, count in documented_zero.items():
        assert int(zero[year]) == count, f"zero {year}"


def test_dam_hourly_view() -> None:
    _need("dam_price_gr.parquet")
    hourly = loaders.load_dam_prices_hourly()
    _assert_utc_index(hourly)
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_dam_crosscheck_with_henex(dam: pd.DataFrame) -> None:
    """The ENTSO-E mirror must reproduce the official HEnEx clearing prices."""
    _need("dam_price_henex_gr.parquet")
    henex = loaders.load_dam_prices_henex()
    _assert_utc_index(henex)
    assert henex.index.min() == TARGET_MODEL
    joined = henex[["price_eur_mwh"]].join(dam["price_eur_mwh"].rename("ec"), how="left")
    assert joined["ec"].notna().all()  # documented: no HEnEx interval is missing in the mirror
    diff = (joined["ec"] - joined["price_eur_mwh"]).abs()
    assert len(joined) > 50_000
    assert int((diff > 0.011).sum()) == DOCUMENTED_HENEX_MISMATCHES
    # same go-live instant of the 15-minute MTU in both sources
    assert henex.loc[henex["resolution_min"] == 15].index.min() == GO_LIVE_15MIN


DOCUMENTED_HENEX_MISMATCHES = 0


def test_dam_crosscheck_report() -> None:
    _need("dam_price_crosscheck.json")
    report = json.loads((P / "dam_price_crosscheck.json").read_text(encoding="utf-8"))
    assert report["mismatch_intervals"] == DOCUMENTED_HENEX_MISMATCHES
    assert report["missing_in_energy_charts"] == 0
    assert report["delivery_days_compared"] > 1900


# --------------------------------------------------------------------------- system
@pytest.fixture(scope="module")
def system() -> pd.DataFrame:
    _need("system_gr.parquet")
    return loaders.load_system()


def test_system_schema(system: pd.DataFrame) -> None:
    _assert_utc_index(system)
    required = {
        "load_mw", "pv_mw", "wind_mw", "gas_mw", "lignite_mw", "hydro_reservoir_mw", "hydro_pumped_gen_mw",
        "hydro_pumping_mw", "other_mw", "net_import_sched_mw", "net_import_phys_total_mw", "resolution_min", "source",
    }
    assert required <= set(system.columns)
    assert (system["resolution_min"] == 60).all()
    assert system.index.min() == pd.Timestamp("2015-01-01 00:00", tz="UTC")


def test_system_hourly_index_gaps(system: pd.DataFrame) -> None:
    """Documented: the hourly index itself has no missing hour up to the last sample."""
    step = system.index.to_series().diff().dropna()
    assert int((step != pd.Timedelta(hours=1)).sum()) == 0


def test_system_documented_missing_values(system: pd.DataFrame) -> None:
    closed = system.loc[:"2025-12-31 23:00"]
    n = len(closed)
    assert n == 96432  # 11 complete years
    # documented NaN counts 2015..2025 (ENTSO-E publication gaps, left as NaN)
    documented = {"load_mw": 35, "pv_mw": 73, "wind_mw": 73, "gas_mw": 60, "lignite_mw": 352}
    for col, count in documented.items():
        assert int(closed[col].isna().sum()) == count, col
    # pumping consumption and batteries are only reported from 2026 on
    assert closed["hydro_pumping_mw"].notna().sum() == 0
    assert system["hydro_pumping_mw"].notna().sum() > 1000


def test_system_plausible_ranges(system: pd.DataFrame) -> None:
    load = system["load_mw"].dropna()
    # documented outlier: 4 hours on 2022-10-31 with an implausibly low published load
    low = load[load < 2500.0]
    assert len(low) == 4 and set(low.index.strftime("%Y-%m-%d")) == {"2022-10-31"}
    assert load.max() < 12500.0
    for col in ("pv_mw", "wind_mw", "gas_mw", "lignite_mw", "hydro_reservoir_mw", "hydro_pumped_gen_mw"):
        assert system[col].min() >= 0.0, col
    assert system["pv_mw"].max() < 12000.0
    assert system["hydro_pumping_mw"].max() <= 0.0  # consumption is stored negative
    assert system["net_import_phys_total_mw"].abs().max() < 4000.0
    # PV must be (almost) zero at night: 23:00 UTC is 01:00 or 02:00 in Greece
    night = system.loc[system.index.hour == 23, "pv_mw"].dropna()
    assert night.quantile(0.999) <= 5.0


def test_system_pv_growth(system: pd.DataFrame) -> None:
    annual_twh = system["pv_mw"].groupby(system.index.year).sum() / 1e6
    assert 3.0 < annual_twh[2016] < 4.5
    assert 10.0 < annual_twh[2024] < 13.0


# --------------------------------------------------------------------------- imbalance
@pytest.fixture(scope="module")
def imbalance() -> pd.DataFrame:
    _need("imbalance_gr.parquet")
    return loaders.load_imbalance()


def test_imbalance_schema_and_completeness(imbalance: pd.DataFrame) -> None:
    _assert_utc_index(imbalance)
    for col in ("imbalance_price_eur_mwh", "mfrr_price_up_eur_mwh", "mfrr_price_down_eur_mwh",
                "act_bal_energy_up_mwh", "act_bal_energy_down_mwh", "system_deviation_mwh", "curt_flag"):
        assert col in imbalance.columns, col
    assert (imbalance["resolution_min"] == 15).all()
    assert imbalance.index.min() == TARGET_MODEL
    assert imbalance.index.max() >= pd.Timestamp("2026-11-01 22:45", tz="UTC")
    # documented: no missing quarter hour between first and last sample
    step = imbalance.index.to_series().diff().dropna()
    assert int((step != pd.Timedelta(minutes=15)).sum()) == 0
    assert imbalance["imbalance_price_eur_mwh"].notna().all()


def test_imbalance_plausible_ranges(imbalance: pd.DataFrame) -> None:
    price = imbalance["imbalance_price_eur_mwh"]
    assert price.min() >= -1000.0 and price.max() <= 4240.0  # documented extremes (2020/2021 launch period)
    recent = price.loc["2022-01-01":]
    assert recent.min() >= -100.0 and recent.max() <= 2500.0
    assert (imbalance["act_bal_energy_up_mwh"].dropna() >= 0).all()
    assert (imbalance["act_bal_energy_down_mwh"].dropna() >= 0).all()
    flag = imbalance["curt_flag"].dropna()
    assert set(flag.unique()) <= {0.0, 1.0}
    assert flag.index.min() == pd.Timestamp("2025-07-17 22:00", tz="UTC")  # documented first week with the flag


def test_imbalance_documented_negative_counts(imbalance: pd.DataFrame) -> None:
    price = imbalance["imbalance_price_eur_mwh"]
    neg = (price < 0).groupby(imbalance.index.year).sum()
    assert {int(y): int(v) for y, v in neg.loc[2021:2025].items()} == {2021: 1812, 2022: 0, 2023: 0, 2024: 0, 2025: 2727}


# --------------------------------------------------------------------------- curtailment and forecasts
def test_curtailment() -> None:
    _need("curtailment_gr.parquet")
    df = loaders.load_curtailment()
    _assert_utc_index(df)
    for col in ("curtail_sched_mw", "surplus_isp1_mw", "surplus_isp2_mw", "surplus_isp3_mw", "res_fc_isp1_mw", "load_fc_isp1_mw"):
        assert col in df.columns, col
    assert set(df["resolution_min"].unique()) <= {15, 30}
    cur = df["curtail_sched_mw"].dropna()
    assert len(cur) > 10_000
    assert cur.min() >= 0.0
    assert cur.max() < 12000.0  # cannot exceed the RES fleet
    assert (df["res_fc_isp1_mw"].dropna() >= 0).all()
    load = df["load_fc_isp1_mw"].dropna()
    assert load.between(2000.0, 12500.0).all()
    # scheduled curtailment is a daytime (PV) phenomenon in Greece: mostly between 06 and 16 UTC
    energy = cur * df.loc[cur.index, "resolution_min"] / 60.0
    day_share = energy[(energy.index.hour >= 6) & (energy.index.hour < 16)].sum() / energy.sum()
    assert day_share > 0.8


def test_res_forecast() -> None:
    _need("res_forecast_gr.parquet")
    df = loaders.load_res_forecast()
    _assert_utc_index(df)
    for col in ("pv_da_fc_mw", "wind_da_fc_mw", "load_da_fc_mw", "pv_actual_mw", "wind_actual_mw", "load_actual_mw"):
        assert col in df.columns, col
    assert (df.index.to_series().diff().dropna() % pd.Timedelta(hours=1) == pd.Timedelta(0)).all()
    both = df[["pv_da_fc_mw", "pv_actual_mw"]].dropna()
    assert len(both) > 40_000
    assert both.corr().iloc[0, 1] > 0.9  # a day-ahead PV forecast must track the actual closely
    load = df[["load_da_fc_mw", "load_actual_mw"]].dropna()
    assert load.corr().iloc[0, 1] > 0.9
    assert (df["pv_da_fc_mw"].dropna() >= 0).all()


# --------------------------------------------------------------------------- capacity, fuel, neighbours
def test_installed_capacity() -> None:
    _need("installed_capacity_gr.csv")
    df = loaders.load_installed_capacity()
    assert list(df.columns) == ["date", "granularity", "technology", "value", "unit", "scope", "source"]
    assert set(df["granularity"]) <= {"year", "month"}
    assert set(df["unit"]) <= {"MW", "MWh"}
    assert (df["value"] >= 0).all()
    assert not df.duplicated(["date", "technology", "source"]).any()
    irena_pv = df[(df["technology"] == "pv") & df["source"].str.startswith("IRENA")].set_index("date")["value"].sort_index()
    assert irena_pv.index.min().year <= 2015 and irena_pv.index.max().year >= 2024
    assert irena_pv.loc["2015":].is_monotonic_increasing
    assert 2400.0 < irena_pv.loc["2015-12-31"] < 2800.0
    ec = df[df["source"].str.startswith("energy-charts")]
    assert {"pv", "wind", "gas", "lignite", "hydro_reservoir", "hydro_pumped"} <= set(ec["technology"])


def test_fuel_prices() -> None:
    _need("fuel_prices.csv")
    df = loaders.load_fuel_prices()
    _assert_utc_index(df)
    assert df.index.min() == pd.Timestamp("2015-01-01", tz="UTC")
    months = df.index.to_period("M")
    assert (np.diff(months.astype("int64")) == 1).all()  # contiguous months
    ttf = df["ttf_eur_mwh"].dropna()
    assert ttf.index.max() >= pd.Timestamp("2026-09-01", tz="UTC")
    assert ttf.between(3.0, 260.0).all()
    assert ttf.loc["2022-08-01"] > 200.0  # the 2022 gas crisis peak month
    recomputed = df["ttf_usd_mmbtu"] / df["usd_per_eur"] / 0.29307107
    assert np.allclose(recomputed.dropna(), df["ttf_eur_mwh"].dropna(), atol=2e-3)
    eua = df["eua_eur_t"]
    # documented gap: no open EEX auction file for 2015 and 2016
    assert eua.loc[:"2016-12-01"].isna().all() and int(eua.loc[:"2016-12-01"].isna().sum()) == 24
    assert eua.loc["2017-01-01":"2026-10-01"].notna().all()
    assert eua.dropna().between(3.0, 110.0).all()


def test_eua_auctions() -> None:
    _need("eua_auctions_daily.csv")
    df = loaders.load_eua_auctions()
    assert str(df.index.tz) == "UTC" and df.index.is_monotonic_increasing
    assert df["eua_auction_price_eur_t"].between(3.0, 110.0).all()
    assert len(df) > 1800


def test_neighbour_prices() -> None:
    _need("neighbours_price.parquet")
    long = loaders.load_neighbour_prices(wide=False)
    assert {"bzn", "price_eur_mwh", "resolution_min"} <= set(long.columns)
    assert str(long.index.tz) == "UTC"
    zones = set(long["bzn"].astype(str).unique())
    assert {"BG", "IT-South"} <= zones
    assert not long.reset_index().duplicated(["ts_utc", "bzn"]).any()
    assert long["price_eur_mwh"].between(-500.0, 4000.0).all()
    wide = loaders.load_neighbour_prices()
    _assert_utc_index(wide)
