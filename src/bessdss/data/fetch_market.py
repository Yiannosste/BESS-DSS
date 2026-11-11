"""Fetchers for REAL public Greek power-market data.

Every function in this module downloads published data from a public source,
caches the raw answer on disk under ``data/raw/`` and writes a tidy processed
file under ``data/processed/``.  Nothing here is modelled or synthetic: when a
quantity is not openly available the function leaves it out and the gap is
documented in ``registers/data_provenance_market.yaml``.

Conventions
-----------
* All timestamps are UTC, timezone-aware and mark the START of the interval.
* Native resolution is kept (no resampling) unless the docstring says otherwise.
* Fetchers are idempotent: closed periods are read from the on-disk cache, only
  the open (current) period is refreshed.
* Requests are rate limited per host and retried with exponential backoff.
  TLS verification is never disabled and the client identifies itself honestly.

Sources
-------
* energy-charts.info API (Fraunhofer ISE), a mirror of the ENTSO-E Transparency
  Platform: day-ahead prices, generation per type, load, cross-border flows,
  day-ahead forecasts, installed power.
* HEnEx (enexgroup.gr): official Day-Ahead Market results, used as cross-check.
* ADMIE / IPTO file API (admie.gr): imbalance settlement prices, ISP results
  (scheduled RES curtailment, RES and load forecasts), MV RES injections.
* IRENA IRENASTAT (pxweb.irena.org): installed capacity statistics.
* World Bank Pink Sheet, ECB reference rates, EEX primary auction reports:
  gas price (TTF), USD/EUR and EU ETS allowance price.

Licences
--------
Day-ahead PRICES are not open data: energy-charts serves them "for private and
internal use only", HEnEx forbids redistribution without written consent and
ENTSO-E does not list item 12.1.D among the data free for re-use.  The price
files written by this module are therefore meant for local analysis and must
not be redistributed in a public repository; share this code instead.  Details
per dataset: ``registers/data_provenance_market.yaml``.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import gzip
import html
import io
import json
import logging
import re
import shutil
import threading
import time
import warnings
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

USER_AGENT = "bessdss-data-fetcher/0.1 (academic research; python-requests)"
EC_API = "https://api.energy-charts.info"
ADMIE_LIST_API = "https://www.admie.gr/getOperationMarketFilewRange"
HENEX_BASE = "https://www.enexgroup.gr"
HENEX_ARCHIVE_PAGE = HENEX_BASE + "/en/web/guest/dam-idm-archive"
HENEX_DAM_PAGE = HENEX_BASE + "/en/markets-publications-el-day-ahead-market"
IRENA_TABLE = (
    "https://pxweb.irena.org/api/v1/en/IRENASTAT/Power%20Capacity%20and%20Generation/"
    "Country_ELECSTAT_2026_H2_PX.px"
)
IRENA_FOLDER = "https://pxweb.irena.org/api/v1/en/IRENASTAT/Power%20Capacity%20and%20Generation"
WB_LANDING = "https://www.worldbank.org/en/research/commodity-markets"
WB_FALLBACK = (
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/"
    "related/CMO-Historical-Data-Monthly.xlsx"
)
ECB_FX = "https://data-api.ecb.europa.eu/service/data/EXR/M.USD.EUR.SP00.A"
EEX_AUCTION = (
    "https://public.eex-group.com/eex/eua-auction-report/"
    "emission-spot-primary-market-auction-report-{year}-data.{ext}"
)

# The Greek market (HEnEx delivery day, ADMIE dispatch day) runs on CET/CEST, the
# clock of the coupled European day-ahead market, NOT on Greek local time (EET).
MARKET_TZ = "Europe/Berlin"
MMBTU_TO_MWH = 0.29307107  # 1 mmbtu = 0.29307107 MWh (exact conversion of 1e6 BTU_IT)
FIRST_YEAR = 2015
TARGET_MODEL_START = dt.date(2020, 11, 1)  # Greek DAM/IDM/balancing market go-live


# --------------------------------------------------------------------------- http
class _Http:
    """Small polite HTTP client: per-host rate limit, retries, honest User-Agent."""

    def __init__(self, min_interval: float = 0.35, retries: int = 5, timeout: float = 120.0) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval = min_interval
        self.retries = retries
        self.timeout = timeout
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _wait(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            ready = self._last.get(host, 0.0) + self.min_interval
            self._last[host] = max(now, ready)
        if ready > now:
            time.sleep(ready - now)

    def get(self, url: str, params: dict[str, Any] | None = None, ok_404: bool = False) -> requests.Response | None:
        """GET with retries. Returns ``None`` for a 404 when ``ok_404`` is set."""
        host = urlparse(url).netloc
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            self._wait(host)
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # network hiccup: back off and retry
                last_exc = exc
                time.sleep(min(60.0, 2.0**attempt))
                continue
            if resp.status_code == 404 and ok_404:
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                last_exc = requests.HTTPError(f"{resp.status_code} for {url}")
                retry_after = resp.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 2.0 ** (attempt + 1)
                time.sleep(min(120.0, delay))
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"giving up on {url}: {last_exc}")


_HTTP = _Http()


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _is_fresh(path: Path, max_age_hours: float | None) -> bool:
    """True if ``path`` exists and is younger than ``max_age_hours`` (None = forever)."""
    if not path.exists():
        return False
    if max_age_hours is None:
        return True
    return (time.time() - path.stat().st_mtime) < max_age_hours * 3600.0


def _cached_json(
    url: str,
    params: dict[str, Any] | None,
    cache: Path,
    max_age_hours: float | None = None,
) -> Any:
    """Fetch JSON with a gzip on-disk cache."""
    if _is_fresh(cache, max_age_hours):
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    resp = _HTTP.get(url, params=params)
    assert resp is not None
    data = resp.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    tmp.replace(cache)
    return data


def _cached_bytes(url: str, cache: Path, max_age_hours: float | None = None, ok_404: bool = False) -> bytes | None:
    """Fetch a binary file with an on-disk cache. Returns ``None`` for 404 if allowed."""
    if _is_fresh(cache, max_age_hours):
        return cache.read_bytes()
    resp = _HTTP.get(url, ok_404=ok_404)
    if resp is None:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    tmp.write_bytes(resp.content)
    tmp.replace(cache)
    return resp.content


def _to_utc_index(unix_seconds: Iterable[int]) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(list(unix_seconds), dtype="int64"), unit="s", utc=True).as_unit("ns")


def _write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# --------------------------------------------------------------------------- energy-charts
def _ec_year(endpoint: str, params: dict[str, str], year: int, refresh_hours: float = 12.0) -> dict[str, Any]:
    """One UTC calendar year of an energy-charts endpoint (cached as json.gz).

    Past years are immutable in the cache; the current year is refreshed when
    the cached copy is older than ``refresh_hours``.
    """
    key = "_".join(f"{v}" for _, v in sorted(params.items()))
    cache = RAW_DIR / "energy_charts" / f"{_slug(endpoint)}_{_slug(key)}_{year}.json.gz"
    current = year >= _utc_now().year
    query = dict(params, start=f"{year}-01-01T00:00Z", end=f"{year}-12-31T23:59Z")
    return _cached_json(f"{EC_API}/{endpoint}", query, cache, refresh_hours if current else None)


def _years(start_year: int, end_year: int | None) -> list[int]:
    last = end_year if end_year is not None else _utc_now().year
    return list(range(start_year, last + 1))


def _ec_price_frame(bzn: str, start_year: int, end_year: int | None) -> pd.DataFrame:
    """Day-ahead price of one bidding zone in native resolution."""
    parts: list[pd.DataFrame] = []
    licence = ""
    years = _years(start_year, end_year)
    for i, year in enumerate(reversed(years)):  # newest first: an unknown zone fails on the first call
        try:
            data = _ec_year("price", {"bzn": bzn}, year)
        except (requests.HTTPError, RuntimeError) as exc:
            log.info("energy-charts price %s %s not available: %s", bzn, year, exc)
            if i == 0:
                break  # the zone is not served at all: do not spend 12 more requests on it
            continue
        if not data.get("unix_seconds"):
            continue
        licence = data.get("license_info", licence)
        parts.append(pd.DataFrame({"ts_utc": _to_utc_index(data["unix_seconds"]), "price_eur_mwh": data["price"]}))
    if not parts:
        return pd.DataFrame(columns=["ts_utc", "price_eur_mwh", "resolution_min"])
    df = pd.concat(parts, ignore_index=True)
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    df = df.dropna(subset=["price_eur_mwh"]).drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)
    df["resolution_min"] = _native_resolution(df["ts_utc"])
    df.attrs["license_info"] = licence
    return df


def _native_resolution(ts: pd.Series) -> pd.Series:
    """Minutes to the next sample; gaps fall back to the neighbouring step."""
    step = ts.diff().shift(-1).dt.total_seconds().div(60.0)
    step = step.where(step.isin([15.0, 30.0, 60.0]))  # steps longer than 1 h are gaps, not resolution
    step = step.ffill().bfill()
    return step.astype("int16")


# --------------------------------------------------------------------------- 1. DAM prices
def fetch_dam_prices(
    start_year: int = FIRST_YEAR,
    end_year: int | None = None,
    crosscheck: bool = True,
    keep_raw_henex: bool = False,
) -> pd.DataFrame:
    """Greek day-ahead price, native resolution, written to ``dam_price_gr.parquet``.

    Source is the energy-charts mirror of ENTSO-E Transparency Platform item
    12.1.D for bidding zone GR.  Until 2020-10-31 this is the System Marginal
    Price of the mandatory pool, from 2020-11-01 the HEnEx DAM clearing price.
    With ``crosscheck`` the complete official HEnEx series is downloaded as well
    (see :func:`fetch_henex_dam`) and compared interval by interval; the result
    is stored in ``dam_price_crosscheck.json``.

    Licence: the API marks GR prices "for private and internal use only"; do
    not redistribute the resulting file (see the module docstring).
    """
    df = _ec_price_frame("GR", start_year, end_year)
    df["source"] = "energy-charts.info (ENTSO-E TP 12.1.D, BZN GR)"
    _write_parquet(df, PROCESSED_DIR / "dam_price_gr.parquet")
    log.info("dam_price_gr: %d rows %s .. %s", len(df), df.ts_utc.min(), df.ts_utc.max())
    if crosscheck:
        henex = fetch_henex_dam(keep_raw=keep_raw_henex)
        report = crosscheck_dam(df, henex)
        (PROCESSED_DIR / "dam_price_crosscheck.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return df


def _henex_links(page_html: str) -> dict[str, str]:
    """Map document title -> absolute download URL from a HEnEx Liferay page."""
    out: dict[str, str] = {}
    pattern = re.compile(r'href="(/[^"]*get_file\?uuid=[^"]+)"[^>]*>(.*?)</a>', re.S)
    for href, body in pattern.findall(page_html):
        title = re.sub(r"<[^>]+>", "", body).replace("&nbsp;", " ").strip()
        if title:
            out[title] = HENEX_BASE + html.unescape(href)
    return out


def _parse_henex_summary(content: bytes) -> tuple[dt.date, list[float]] | None:
    """Extract (delivery day, clearing prices per MTU) from a DAM ResultsSummary file.

    Layout: a header row with the delivery date in column A and the MTU numbers,
    then a block ``Market Clearing Price`` whose first ``Greece Mainland`` row is
    the clearing price (15-minute MCP after the 15-minute MTU go-live).
    """
    import openpyxl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    try:
        ws = wb.worksheets[0]
        ws.reset_dimensions()  # some workbooks declare a wrong sheet dimension (A1:A1)
        day: dt.date | None = None
        in_price_block = False
        for row in ws.iter_rows(min_row=1, max_row=14, values_only=True):
            first = row[0] if row else None
            if isinstance(first, dt.datetime) and day is None:
                day = first.date()
                continue
            label = str(first).strip() if first is not None else ""
            if label.startswith("Market Clearing Price"):
                in_price_block = True
                continue
            if in_price_block and label.startswith("Greece Mainland"):
                # last numeric cell is the daily average ("TOTAL" column): drop it
                cells = list(row[1:])
                while cells and cells[-1] is None:
                    cells.pop()
                values = [float(v) for v in cells[:-1] if isinstance(v, (int, float))]
                if day is None or not values:
                    return None
                return day, values
        return None
    finally:
        wb.close()


def _henex_rows(day: dt.date, values: list[float], src: str) -> pd.DataFrame | None:
    """Turn one delivery day into UTC rows. The delivery day starts at 00:00 CET/CEST."""
    n = len(values)
    if n in (23, 24, 25):
        step = 60
    elif n in (92, 96, 100):
        step = 15
    else:
        log.warning("HEnEx %s: unexpected number of MTUs %d", src, n)
        return None
    start = pd.Timestamp(day).tz_localize(MARKET_TZ).tz_convert("UTC")
    ts = start + pd.to_timedelta(np.arange(n) * step, unit="min")
    return pd.DataFrame(
        {
            "ts_utc": pd.DatetimeIndex(ts).as_unit("ns"),
            "price_eur_mwh": values,
            "resolution_min": np.int16(step),
            "delivery_day": pd.Timestamp(day),
            "src_file": src,
        }
    )


def fetch_henex_dam(sample_step_current_year: int = 3, keep_raw: bool = False) -> pd.DataFrame:
    """Official HEnEx DAM clearing prices, written to ``dam_price_henex_gr.parquet``.

    Closed years come from the yearly ``ResultsSummary`` zip archives on the
    HEnEx archive page (complete since 2020-11-01).  The current year is only
    published as daily files behind a paginated list, so one list page out of
    ``sample_step_current_year`` is read (7 delivery days per page): a sample.
    The zips are large (15 to 90 MB each); after parsing only a small per-year
    parquet cache is kept unless ``keep_raw`` is set.
    """
    cache_dir = RAW_DIR / "henex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    archive = _HTTP.get(HENEX_ARCHIVE_PAGE)
    assert archive is not None
    links = _henex_links(archive.text)
    yearly = {int(t[:4]): url for t, url in links.items() if re.match(r"^\d{4}_EL-.*_ResultsSummary", t)}
    for year, url in sorted(yearly.items()):
        parsed = cache_dir / f"henex_dam_{year}.parquet"
        if parsed.exists():
            frames.append(pd.read_parquet(parsed))
            continue
        log.info("HEnEx: downloading yearly archive %s", year)
        zpath = cache_dir / f"{year}_ResultsSummary.zip"
        content = _cached_bytes(url, zpath)
        assert content is not None
        rows = _parse_henex_zip(content)
        rows.to_parquet(parsed, index=False)
        frames.append(rows)
        if not keep_raw:
            zpath.unlink(missing_ok=True)

    frames.append(_henex_current_year_sample(cache_dir, sample_step_current_year))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    df = df.sort_values(["ts_utc", "src_file"]).drop_duplicates("ts_utc", keep="last").reset_index(drop=True)
    df["source"] = "HEnEx EL-DAM ResultsSummary (enexgroup.gr)"
    _write_parquet(df, PROCESSED_DIR / "dam_price_henex_gr.parquet")
    log.info("dam_price_henex_gr: %d rows %s .. %s", len(df), df.ts_utc.min(), df.ts_utc.max())
    return df


def _parse_henex_zip(content: bytes) -> pd.DataFrame:
    """Parse every DAM ResultsSummary workbook of a yearly archive (latest version per day)."""
    zf = zipfile.ZipFile(io.BytesIO(content))
    pat = re.compile(r"(\d{8})_EL-DAM_ResultsSummary_EN_v(\d+)\.xlsx$")
    best: dict[str, tuple[int, str]] = {}
    for name in zf.namelist():
        m = pat.search(name)
        if m and (m.group(1) not in best or int(m.group(2)) > best[m.group(1)][0]):
            best[m.group(1)] = (int(m.group(2)), name)
    frames: list[pd.DataFrame] = []
    for _, (_, name) in sorted(best.items()):
        short = name.rsplit("/", 1)[-1]
        try:
            parsed = _parse_henex_summary(zf.read(name))
        except Exception as exc:  # corrupt member: report and go on
            log.warning("HEnEx %s unreadable: %s", short, exc)
            continue
        if parsed is None:
            log.warning("HEnEx %s: no clearing price row found", short)
            continue
        rows = _henex_rows(parsed[0], parsed[1], short)
        if rows is not None:
            frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _henex_current_year_sample(cache_dir: Path, step: int) -> pd.DataFrame:
    """Sample of daily ResultsSummary files of the running year (paginated list)."""
    first = _HTTP.get(HENEX_DAM_PAGE)
    assert first is not None
    text = first.text
    anchor = text.find("_EL-DAM_ResultsSummary_EN_")
    if anchor < 0:
        log.warning("HEnEx: ResultsSummary list not found on the DAM page")
        return pd.DataFrame()
    cur = text.find("_cur=", anchor)
    href_start = text.rfind('href="', 0, cur) + 6
    page_url = html.unescape(text[href_start : text.find('"', href_start)])
    page_url = re.sub(r"_cur=\d+", "_cur={cur}", page_url)

    frames: list[pd.DataFrame] = []
    page, empty = 1, 0
    while empty < 1 and page < 200:
        resp = _HTTP.get(page_url.format(cur=page))
        assert resp is not None
        links = {t: u for t, u in _henex_links(resp.text).items() if "_EL-DAM_ResultsSummary_EN_" in t}
        if not links:
            empty += 1
        for title, url in sorted(links.items()):
            raw = _cached_bytes(url, cache_dir / "daily" / f"{title}.xlsx")
            if raw is None:
                continue
            parsed = _parse_henex_summary(raw)
            if parsed is None:
                continue
            rows = _henex_rows(parsed[0], parsed[1], f"{title}.xlsx")
            if rows is not None:
                frames.append(rows)
        page += step
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def crosscheck_dam(ec: pd.DataFrame, henex: pd.DataFrame, tol: float = 0.011) -> dict[str, Any]:
    """Compare the energy-charts series with HEnEx official results.

    Both series are joined on the UTC timestamp.  An interval counts as a
    mismatch when the absolute price difference exceeds ``tol`` EUR/MWh.
    """
    j = henex.merge(ec[["ts_utc", "price_eur_mwh", "resolution_min"]], on="ts_utc", how="left", suffixes=("_henex", "_ec"))
    j["diff"] = j["price_eur_mwh_ec"] - j["price_eur_mwh_henex"]
    j["year"] = j["ts_utc"].dt.year
    missing = j["price_eur_mwh_ec"].isna()
    bad = (~missing) & (j["diff"].abs() > tol)
    per_year = {}
    for year, g in j.groupby("year"):
        gm = g["price_eur_mwh_ec"].isna()
        gb = (~gm) & (g["diff"].abs() > tol)
        per_year[int(year)] = {
            "henex_intervals": int(len(g)),
            "days_compared": int(g["delivery_day"].nunique()),
            "missing_in_energy_charts": int(gm.sum()),
            "mismatch_intervals": int(gb.sum()),
            "max_abs_diff_eur_mwh": float(g.loc[~gm, "diff"].abs().max()) if (~gm).any() else None,
        }
    worst = j.loc[bad].assign(absdiff=lambda x: x["diff"].abs()).nlargest(15, "absdiff")
    report = {
        "generated_utc": _utc_now().isoformat(),
        "tolerance_eur_mwh": tol,
        "henex_period": [str(henex.ts_utc.min()), str(henex.ts_utc.max())],
        "intervals_compared": int((~missing).sum()),
        "delivery_days_compared": int(j.loc[~missing, "delivery_day"].nunique()),
        "missing_in_energy_charts": int(missing.sum()),
        "mismatch_intervals": int(bad.sum()),
        "mismatch_days": sorted({str(d.date()) for d in j.loc[bad, "delivery_day"]}),
        "missing_days": sorted({str(d.date()) for d in j.loc[missing, "delivery_day"]}),
        "per_year": per_year,
        "worst_mismatches": [
            {
                "ts_utc": str(r.ts_utc),
                "henex": float(r.price_eur_mwh_henex),
                "energy_charts": float(r.price_eur_mwh_ec),
            }
            for r in worst.itertuples()
        ],
        "first_15min_ts_energy_charts": str(ec.loc[ec.resolution_min == 15, "ts_utc"].min()),
        "first_15min_ts_henex": str(henex.loc[henex.resolution_min == 15, "ts_utc"].min()),
    }
    log.info(
        "DAM cross-check: %d intervals, %d mismatches, %d missing",
        report["intervals_compared"], report["mismatch_intervals"], report["missing_in_energy_charts"],
    )
    return report


# --------------------------------------------------------------------------- 2. system
_EC_GEN_MAP = {
    "Load": "load_mw",
    "Solar": "pv_mw",
    "Wind onshore": "wind_mw",
    "Wind offshore": "wind_offshore_mw",
    "Hydro Run-of-River": "hydro_ror_mw",
    "Hydro water reservoir": "hydro_reservoir_mw",
    "Hydro pumped storage": "hydro_pumped_gen_mw",
    "Hydro pumped storage consumption": "hydro_pumping_mw",
    "Fossil gas": "gas_mw",
    "Fossil brown coal / lignite": "lignite_mw",
    "Fossil oil": "oil_mw",
    "Battery": "battery_discharge_mw",
    "Battery Consumption": "battery_charge_mw",
    "Cross border electricity trading": "net_import_sched_mw",
}
# Series computed by energy-charts from the others: not stored (can be re-derived).
_EC_DERIVED = {"Residual load", "Renewable share of load", "Renewable share of generation"}
_BORDER_MAP = {
    "Albania": "al",
    "Bulgaria": "bg",
    "Italy": "it",
    "North Macedonia": "mk",
    "Turkey": "tr",
    "sum": "total",
}


def _ec_series_frame(endpoint: str, params: dict[str, str], list_key: str, years: list[int]) -> pd.DataFrame:
    """Wide frame (UTC index) of all named series of an energy-charts endpoint."""
    parts: list[pd.DataFrame] = []
    for year in years:
        data = _ec_year(endpoint, params, year)
        ts = data.get("unix_seconds") or []
        if not ts:
            continue
        cols = {s["name"]: pd.to_numeric(pd.Series(s["data"]), errors="coerce").to_numpy() for s in data[list_key]}
        parts.append(pd.DataFrame(cols, index=_to_utc_index(ts)))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_system(start_year: int = FIRST_YEAR, end_year: int | None = None) -> pd.DataFrame:
    """Greek system data (generation per type, load, cross-border), ``system_gr.parquet``.

    energy-charts ``public_power`` (ENTSO-E actual generation per production
    type 16.1.B/C and actual total load 6.1.A) plus ``cbpf`` (physical flows
    12.1.G).  Sign convention: generation and load positive, consumption of
    pumps and batteries negative, net imports positive.  ``other_mw`` is the sum
    of every generation series that has no dedicated column.
    """
    years = _years(start_year, end_year)
    gen = _ec_series_frame("public_power", {"country": "gr"}, "production_types", years)
    gen = gen.drop(columns=[c for c in gen.columns if c in _EC_DERIVED])
    unknown = [c for c in gen.columns if c not in _EC_GEN_MAP]
    out = gen.rename(columns=_EC_GEN_MAP)
    if unknown:
        log.info("system_gr: series folded into other_mw: %s", unknown)
        out["other_mw"] = out[unknown].sum(axis=1, min_count=1)
        out = out.drop(columns=unknown)
    else:
        out["other_mw"] = np.nan

    flows = _ec_series_frame("cbpf", {"country": "gr"}, "countries", years)
    flows = flows.rename(columns={c: f"net_import_phys_{_BORDER_MAP.get(c, _slug(c))}_mw" for c in flows.columns})
    flows = flows * 1000.0  # GW -> MW
    out = out.join(flows, how="outer")

    out.index.name = "ts_utc"
    out = out.reset_index()
    out["resolution_min"] = _native_resolution(out["ts_utc"])
    out["source"] = "energy-charts.info (ENTSO-E TP 16.1.B/C, 6.1.A, 12.1.F/G)"
    first = ["ts_utc", "resolution_min", "load_mw", "pv_mw", "wind_mw"]
    out = out[first + [c for c in out.columns if c not in first]]
    _write_parquet(out, PROCESSED_DIR / "system_gr.parquet")
    log.info("system_gr: %d rows %s .. %s", len(out), out.ts_utc.min(), out.ts_utc.max())
    return out


def fetch_neighbour_prices(
    zones: tuple[str, ...] = ("BG", "IT-South", "IT-Calabria", "RO", "MK", "AL", "TR"),
    start_year: int = FIRST_YEAR,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Day-ahead prices of neighbouring bidding zones (long format), ``neighbours_price.parquet``.

    Zones that energy-charts does not serve are skipped and reported in the log.
    """
    frames: list[pd.DataFrame] = []
    for zone in zones:
        df = _ec_price_frame(zone, start_year, end_year)
        if df.empty:
            log.warning("neighbours: no data for bidding zone %s", zone)
            continue
        df.insert(1, "bzn", zone)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["bzn"] = out["bzn"].astype("category")
    out["price_eur_mwh"] = out["price_eur_mwh"].astype("float32")
    out["source"] = "energy-charts.info (ENTSO-E TP 12.1.D)"
    out["source"] = out["source"].astype("category")
    _write_parquet(out, PROCESSED_DIR / "neighbours_price.parquet")
    log.info("neighbours_price: %d rows, zones %s", len(out), sorted(out.bzn.unique()))
    return out


# --------------------------------------------------------------------------- ADMIE machinery
def admie_list(category: str, start: dt.date, end: dt.date, max_age_hours: float | None = 12.0) -> list[dict[str, str]]:
    """File listing of one ADMIE category for a date range (cached)."""
    cache = RAW_DIR / "admie" / "_listings" / f"{category}_{start:%Y%m%d}_{end:%Y%m%d}.json.gz"
    params = {"dateStart": f"{start:%Y-%m-%d}", "dateEnd": f"{end:%Y-%m-%d}", "FileCategory": category}
    data = _cached_json(ADMIE_LIST_API, params, cache, max_age_hours)
    return list(data) if isinstance(data, list) else []


def _admie_listing_chunked(category: str, start: dt.date, end: dt.date) -> list[dict[str, str]]:
    """Listing split by calendar year so that closed years stay cached forever."""
    out: list[dict[str, str]] = []
    today = _utc_now().date()
    for year in range(start.year, end.year + 1):
        a = max(start, dt.date(year, 1, 1))
        b = min(end, dt.date(year, 12, 31))
        closed = b < today - dt.timedelta(days=120)  # ADMIE republishes corrected versions for months
        out.extend(admie_list(category, a, b, None if closed else 12.0))
    seen: set[str] = set()
    unique = []
    for item in out:
        if item["file_path"] not in seen:
            seen.add(item["file_path"])
            unique.append(item)
    return unique


def _admie_latest(listing: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the most recently published file for every covered period."""
    best: dict[tuple[str, str], tuple[pd.Timestamp, str, dict[str, str]]] = {}
    for item in listing:
        key = (item["file_fromdate"], item["file_todate"])
        published = pd.to_datetime(item.get("file_published", ""), format="%d.%m.%Y %H:%M", errors="coerce")
        if pd.isna(published):
            published = pd.Timestamp("1970-01-01")
        rank = (published, item["file_path"], item)
        if key not in best or rank[:2] > best[key][:2]:
            best[key] = rank
    return [v[2] for _, v in sorted(best.items(), key=lambda kv: pd.to_datetime(kv[0][0], format="%d.%m.%Y"))]


def _admie_collect(
    category: str,
    start: dt.date,
    end: dt.date,
    parser: Callable[[bytes, str], pd.DataFrame | None],
    keep_raw: bool = False,
    workers: int = 4,
) -> pd.DataFrame:
    """Download and parse every (latest version) file of a category.

    The parsed rows are stored in ``data/raw/admie/_parsed/<category>.parquet``
    keyed by file name, so a second call only downloads files that are new or
    were republished.  Raw workbooks are deleted after parsing unless
    ``keep_raw`` is set (the complete raw tree is several hundred MB).
    """
    parsed_path = RAW_DIR / "admie" / "_parsed" / f"{category}.parquet"
    failed_path = RAW_DIR / "admie" / "_parsed" / f"{category}_failed.json"
    cache = pd.read_parquet(parsed_path) if parsed_path.exists() else pd.DataFrame()
    failed: dict[str, str] = json.loads(failed_path.read_text()) if failed_path.exists() else {}

    wanted = _admie_latest(_admie_listing_chunked(category, start, end))
    names = {item["file_path"].rsplit("/", 1)[-1]: item for item in wanted}
    done = set(cache["src_file"].unique()) if len(cache) else set()
    todo = [n for n in names if n not in done and n not in failed]
    log.info("ADMIE %s: %d files wanted, %d cached, %d to download", category, len(names), len(done & set(names)), len(todo))

    raw_dir = RAW_DIR / "admie" / category

    def work(name: str) -> tuple[str, pd.DataFrame | None, str]:
        url = names[name]["file_path"]
        try:
            content = _cached_bytes(url, raw_dir / name, ok_404=True)
            if content is None:
                return name, None, "404"
            frame = parser(content, name)
            if not keep_raw:
                (raw_dir / name).unlink(missing_ok=True)
            return name, frame, "" if frame is not None and len(frame) else "no rows parsed"
        except Exception as exc:  # one broken workbook must not stop the run
            return name, None, f"{type(exc).__name__}: {exc}"

    def checkpoint(frames: list[pd.DataFrame]) -> None:
        nonlocal cache
        if frames:
            cache = pd.concat([cache, *frames], ignore_index=True)
            parsed_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = parsed_path.with_suffix(".tmp")
            cache.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(parsed_path)
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text(json.dumps(failed, indent=1, ensure_ascii=False), encoding="utf-8")

    chunk = 400  # the parsed cache is saved after every chunk so an interrupted run can resume
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for lo in range(0, len(todo), chunk):
            new_frames: list[pd.DataFrame] = []
            for name, frame, err in pool.map(work, todo[lo : lo + chunk]):
                if err:
                    failed[name] = err
                    log.warning("ADMIE %s: %s -> %s", category, name, err)
                else:
                    assert frame is not None
                    frame = frame.copy()
                    frame["src_file"] = name
                    new_frames.append(frame)
            checkpoint(new_frames)
            log.info("ADMIE %s: %d/%d", category, min(lo + chunk, len(todo)), len(todo))
    if not keep_raw and raw_dir.exists() and not any(raw_dir.iterdir()):
        raw_dir.rmdir()
    if not len(cache):
        return cache
    # only the latest version of each period contributes to the result
    return cache[cache["src_file"].isin(names)].reset_index(drop=True)


def _market_sequence(first_local: Any, n: int, step_min: int) -> pd.DatetimeIndex:
    """UTC stamps of ``n`` consecutive periods starting at a naive CET/CEST wall time.

    ADMIE and HEnEx workbooks label periods with naive market time.  Counting
    periods from the (unambiguous) first label is robust on clock-change days:
    a 23 hour day simply has fewer columns and a 25 hour day more.
    """
    start = pd.Timestamp(first_local).tz_localize(MARKET_TZ).tz_convert("UTC")
    return pd.DatetimeIndex(start + pd.to_timedelta(np.arange(n) * step_min, unit="min")).as_unit("ns")


def _read_excel_any(content: bytes, name: str, **kwargs: Any) -> pd.DataFrame:
    # trust the content, not the extension: ADMIE serves many xlsx files named *.xls
    if content[:2] == b"PK":
        engine = "openpyxl"
    elif content[:4] == b"\xd0\xcf\x11\xe0":
        engine = "xlrd"
    else:
        engine = "xlrd" if name.lower().endswith(".xls") else "openpyxl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_excel(io.BytesIO(content), engine=engine, **kwargs)


# --------------------------------------------------------------------------- 4. imbalance
_IMB_COLUMNS = [
    (r"total activated balancing energy up", "act_bal_energy_up_mwh"),
    (r"total activated balancing energy down", "act_bal_energy_down_mwh"),
    (r"balancing energy idev", "bal_energy_idev_mwh"),
    (r"balancing energy udev", "bal_energy_udev_mwh"),
    (r"isp energy", "isp_energy_mwh"),
    (r"system deviation", "system_deviation_mwh"),
    (r"total activated balancing energy \(mwh\)|total activated balancing energy$", "act_bal_energy_net_mwh"),
    (r"curt_flag", "curt_flag"),
    (r"imbalance price", "imbalance_price_eur_mwh"),
    (r"mfrr price up", "mfrr_price_up_eur_mwh"),
    (r"mfrr price down", "mfrr_price_down_eur_mwh"),
    (r"uplift account 1", "uplift_1_eur_mwh"),
    (r"uplift account 2", "uplift_2_eur_mwh"),
    (r"uplift account 3", "uplift_3_eur_mwh"),
]


def _parse_imbabe(content: bytes, name: str) -> pd.DataFrame | None:
    """Parse a weekly IMBABE workbook (15-minute settlement prices and volumes)."""
    raw = _read_excel_any(content, name, sheet_name=0, header=None)
    header = [re.sub(r"\s+", " ", str(c).replace("\xa0", " ")).strip().lower() for c in raw.iloc[0].tolist()]
    if "startdate" not in header:
        return None
    body = raw.iloc[1:].reset_index(drop=True)
    start = pd.to_datetime(body.iloc[:, header.index("startdate")], errors="coerce", dayfirst=True)
    body = body.loc[start.notna()].reset_index(drop=True)
    start = start.dropna().reset_index(drop=True)
    if start.empty:
        return None
    out = pd.DataFrame({"ts_utc": _market_sequence(start.iloc[0], len(start), 15)})
    # sanity: the sequence must end where the labels end (naive CET/CEST wall time)
    last_wall = out["ts_utc"].iloc[-1].tz_convert(MARKET_TZ).tz_localize(None)
    if last_wall != start.iloc[-1]:
        raise ValueError(f"{name}: period labels are not contiguous ({last_wall} vs {start.iloc[-1]})")
    used: set[int] = set()
    for pattern, col in _IMB_COLUMNS:
        for idx, label in enumerate(header):
            if idx in used or not re.search(pattern, label):
                continue
            out[col] = pd.to_numeric(body.iloc[:, idx], errors="coerce").to_numpy()
            used.add(idx)
            break
    return out


def fetch_imbalance(
    start: dt.date = TARGET_MODEL_START,
    end: dt.date | None = None,
    keep_raw: bool = False,
) -> pd.DataFrame:
    """ADMIE imbalance settlement prices at 15 minutes, ``imbalance_gr.parquet``.

    File category ``IMBABE`` ("Activated Balancing Energy and Settlement
    Prices", one workbook per week, republished when settlement is revised; the
    latest version wins).  Columns: imbalance price, mFRR up/down prices,
    activated balancing energy up/down, system deviation (from 2025 files),
    uplift account charges and the ``curt_flag`` column that newer files carry.
    Period labels in the files are CET/CEST wall time and are converted to UTC.
    """
    end = end or _utc_now().date()
    df = _admie_collect("IMBABE", start, end, _parse_imbabe, keep_raw=keep_raw)
    df = df.sort_values(["ts_utc", "src_file"]).drop_duplicates("ts_utc", keep="last").reset_index(drop=True)
    df["resolution_min"] = np.int16(15)
    df["source"] = "ADMIE file API, category IMBABE"
    first = ["ts_utc", "resolution_min", "imbalance_price_eur_mwh", "mfrr_price_up_eur_mwh", "mfrr_price_down_eur_mwh"]
    df = df[first + [c for c in df.columns if c not in first and c != "src_file"] + ["src_file"]]
    df["src_file"] = df["src_file"].astype("category")
    df["source"] = df["source"].astype("category")
    # Keep the file small (about 6 MB instead of 15): prices stay as published, volumes are
    # rounded to 0.01 MWh and uplift charges to 0.001 EUR/MWh, everything stored as float32.
    volumes = [c for c in df.columns if c.endswith("_mwh") and "eur" not in c]
    uplifts = [c for c in df.columns if c.startswith("uplift_")]
    df[volumes] = df[volumes].round(2)
    df[uplifts] = df[uplifts].round(3)
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype("float32")
    path = PROCESSED_DIR / "imbalance_gr.parquet"
    df.to_parquet(path, engine="pyarrow", compression="zstd", compression_level=19, index=False)
    log.info("imbalance_gr: %d rows %s .. %s", len(df), df.ts_utc.min(), df.ts_utc.max())
    return df


# --------------------------------------------------------------------------- 5. curtailment / forecasts
_ISP_ROWS = {
    "greece load": "load_fc_mw",
    "greece losses": "losses_fc_mw",
    "mandatory hydro injections": "mandatory_hydro_mw",
    "renewables": "res_fc_mw",
    "thermal units": "thermal_sched_mw",
    "hydro units": "hydro_sched_mw",
    "total dispatchable load": "dispatchable_load_mw",
    "energy surplus": "energy_surplus_mw",
    "net cbs schedules": "net_cbs_sched_mw",
}


def _parse_isp_results(content: bytes, name: str) -> pd.DataFrame | None:
    """Parse the system overview block of an ISP results workbook.

    The first sheet starts with a header row (label ``Non Dispatchable Load``)
    that carries the period start times, followed by one row per quantity.
    Resolution is 30 minutes in older files and 15 minutes in newer ones.
    ``Energy Surplus`` (present in newer files only) is the generation the
    schedule could not place, i.e. the RES curtailment ordered by the ISP.
    """
    import openpyxl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    try:
        ws = wb.worksheets[0]
        ws.reset_dimensions()  # some workbooks declare a wrong sheet dimension (A1:A1)
        times: list[dt.datetime] = []
        cols: list[int] = []
        data: dict[str, list[float]] = {}
        for row in ws.iter_rows(min_row=1, max_row=40, values_only=True):
            label = str(row[0]).strip().lower() if row and row[0] is not None else ""
            if not times:
                stamps = [(i, v) for i, v in enumerate(row) if isinstance(v, dt.datetime)]
                if len(stamps) >= 20:
                    cols = [i for i, _ in stamps]
                    times = [v for _, v in stamps]
                continue
            if label.startswith("energy marginal prices"):
                break
            if label in _ISP_ROWS and _ISP_ROWS[label] not in data:
                vals = []
                for i in cols:
                    v = row[i] if i < len(row) else None
                    vals.append(float(v) if isinstance(v, (int, float)) else np.nan)
                data[_ISP_ROWS[label]] = vals
        if not times or not data:
            return None
        step = int(round((times[1] - times[0]).total_seconds() / 60.0))
        out = pd.DataFrame({"ts_utc": _market_sequence(times[0], len(times), step)})
        out["resolution_min"] = np.int16(step)
        out["dispatch_day"] = pd.Timestamp(times[0].date())
        for col, vals in data.items():
            out[col] = np.asarray(vals, dtype="float32")
        return out
    finally:
        wb.close()


def fetch_isp(run: str, start: dt.date, end: dt.date | None = None, keep_raw: bool = False) -> pd.DataFrame:
    """Parsed ISP results (``run`` is ``ISP1``, ``ISP2`` or ``ISP3``)."""
    end = end or _utc_now().date() + dt.timedelta(days=1)
    df = _admie_collect(f"{run}ISPResults", start, end, _parse_isp_results, keep_raw=keep_raw)
    if not len(df):
        return df
    return df.sort_values(["ts_utc", "src_file"]).drop_duplicates("ts_utc", keep="last").reset_index(drop=True)


def fetch_curtailment(
    start: dt.date = TARGET_MODEL_START,
    end: dt.date | None = None,
    keep_raw: bool = False,
) -> pd.DataFrame:
    """Scheduled RES curtailment from the ADMIE ISP, ``curtailment_gr.parquet``.

    What it is: the ``Energy Surplus`` row of the Integrated Scheduling Process
    results, i.e. the RES generation that the TSO schedule could not absorb and
    ordered to be curtailed (redispatch type, ex ante, system level).  It is NOT
    metered curtailed energy, it does not contain real-time curtailment orders
    issued after the last ISP, curtailment by the distribution operator, or
    voluntary market-based curtailment of plants that did not sell.

    Three runs are stored side by side (ISP1 day ahead, ISP2 evening before,
    ISP3 intraday for the second half of the day) plus ``curtail_sched_mw``,
    the value of the latest available run for each period (stored positive).
    """
    runs = {"ISP1": fetch_isp("ISP1", start, end, keep_raw)}
    isp1_raw = runs["ISP1"]
    has_surplus = isp1_raw.dropna(subset=["energy_surplus_mw"]) if "energy_surplus_mw" in isp1_raw.columns else isp1_raw.iloc[:0]
    if len(has_surplus):
        # older workbooks have no "Energy Surplus" row, so the later runs add nothing before that day
        refine_start = max(start, has_surplus["dispatch_day"].min().date())
        for run in ("ISP2", "ISP3"):
            runs[run] = fetch_isp(run, refine_start, end, keep_raw)
    else:
        runs["ISP2"] = runs["ISP3"] = pd.DataFrame()
    idx = sorted(set().union(*[set(df["ts_utc"]) for df in runs.values() if len(df)]))
    out = pd.DataFrame(index=pd.DatetimeIndex(idx, name="ts_utc"))
    for run, df in runs.items():
        key = run.lower()
        if not len(df) or "energy_surplus_mw" not in df.columns:
            out[f"surplus_{key}_mw"] = np.nan
            continue
        s = df.set_index("ts_utc")["energy_surplus_mw"]
        out[f"surplus_{key}_mw"] = -s.reindex(out.index)  # files store the surplus negative
    isp1 = runs["ISP1"].set_index("ts_utc")
    out["resolution_min"] = isp1["resolution_min"].reindex(out.index)
    for run in ("ISP2", "ISP3"):
        if len(runs[run]):
            out["resolution_min"] = out["resolution_min"].fillna(
                runs[run].set_index("ts_utc")["resolution_min"].reindex(out.index)
            )
    out["curtail_sched_mw"] = out["surplus_isp3_mw"].fillna(out["surplus_isp2_mw"]).fillna(out["surplus_isp1_mw"])
    out["res_fc_isp1_mw"] = isp1["res_fc_mw"].reindex(out.index)
    out["load_fc_isp1_mw"] = isp1["load_fc_mw"].reindex(out.index)
    out["net_cbs_sched_isp1_mw"] = isp1["net_cbs_sched_mw"].reindex(out.index)
    # rows before the first file that carries an "Energy Surplus" row stay NaN: unknown, not zero
    out = out.dropna(subset=["resolution_min"]).reset_index()
    out["resolution_min"] = out["resolution_min"].astype("int16")
    out["source"] = "ADMIE file API, categories ISP1ISPResults/ISP2ISPResults/ISP3ISPResults"
    first = ["ts_utc", "resolution_min", "curtail_sched_mw"]
    out = out[first + [c for c in out.columns if c not in first]]
    _write_parquet(out, PROCESSED_DIR / "curtailment_gr.parquet")
    log.info("curtailment_gr: %d rows %s .. %s", len(out), out.ts_utc.min(), out.ts_utc.max())
    return out


def fetch_res_forecast(start_year: int = FIRST_YEAR, end_year: int | None = None) -> pd.DataFrame:
    """Day-ahead forecasts versus actuals, hourly, ``res_forecast_gr.parquet``.

    * ``*_da_fc_mw``: ENTSO-E day-ahead forecasts (wind and solar 14.1.D, total
      load 6.1.B) as mirrored by energy-charts ``public_power_forecast``.
    * ``*_actual_mw``: ENTSO-E actuals from ``system_gr.parquet``.
    * ``res_fc_isp1_mw`` / ``load_fc_isp1_mw``: hourly mean of the ADMIE ISP1
      forecast (all RES including small hydro and CHP; taken from
      ``curtailment_gr.parquet`` when that file exists).
    """
    years = _years(start_year, end_year)
    out = pd.DataFrame()
    for ptype, col in (("solar", "pv_da_fc_mw"), ("wind_onshore", "wind_da_fc_mw"), ("load", "load_da_fc_mw")):
        parts = []
        for year in years:
            data = _ec_year("public_power_forecast", {"country": "gr", "production_type": ptype, "forecast_type": "day-ahead"}, year)
            ts, vals = data.get("unix_seconds") or [], data.get("forecast_values") or []
            if ts and len(vals) == len(ts):
                parts.append(pd.Series(pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(), index=_to_utc_index(ts)))
        if parts:
            s = pd.concat(parts)
            out = out.join(s[~s.index.duplicated(keep="last")].rename(col), how="outer")
        else:
            out[col] = np.nan

    system_path = PROCESSED_DIR / "system_gr.parquet"
    if system_path.exists():
        sysdf = pd.read_parquet(system_path).set_index("ts_utc")
        actual = sysdf[["pv_mw", "wind_mw", "load_mw"]].rename(
            columns={"pv_mw": "pv_actual_mw", "wind_mw": "wind_actual_mw", "load_mw": "load_actual_mw"}
        )
        out = out.join(actual, how="outer")
    curt_path = PROCESSED_DIR / "curtailment_gr.parquet"
    if curt_path.exists():
        isp = pd.read_parquet(curt_path).set_index("ts_utc")[["res_fc_isp1_mw", "load_fc_isp1_mw"]]
        hourly = isp.resample("1h").mean()
        out = out.join(hourly, how="left")
    out = out.dropna(how="all").sort_index()
    out.index.name = "ts_utc"
    out = out.reset_index()
    out["resolution_min"] = np.int16(60)
    out["source"] = "energy-charts.info (ENTSO-E 14.1.D, 6.1.B, 16.1.B/C, 6.1.A); ADMIE ISP1"
    _write_parquet(out, PROCESSED_DIR / "res_forecast_gr.parquet")
    log.info("res_forecast_gr: %d rows %s .. %s", len(out), out.ts_utc.min(), out.ts_utc.max())
    return out


# --------------------------------------------------------------------------- 3. installed capacity
_EC_CAP_MAP = {
    "Solar AC": "pv",
    "Solar DC": "pv_dc",
    "Wind onshore": "wind",
    "Wind offshore": "wind_offshore",
    "Hydro Run-of-River": "hydro_ror",
    "Hydro water reservoir": "hydro_reservoir",
    "Hydro pumped storage": "hydro_pumped",
    "Fossil gas": "gas",
    "Fossil brown coal / lignite": "lignite",
    "Fossil oil": "oil",
    "Biomass": "biomass",
    "Geothermal": "geothermal",
    "Other renewables": "other_renewables",
    "Battery Storage (Power)": "battery_power",
    "Battery Storage (Capacity)": "battery_energy",
}
_IRENA_TECH = {
    "Solar photovoltaic": "pv",
    "Onshore wind energy": "wind",
    "Renewable hydropower": "hydro_renewable",
    "Mixed hydropower": "hydro_mixed",
    "Mechanical storage (Pumped hydro)": "hydro_pumped",
    "Natural gas": "gas",
    "Coal": "lignite",
    "Oil": "oil",
    "Solid biofuels": "biomass_solid",
    "Gas biofuels": "biogas",
    "Total Renewable": "total_renewable",
    "Total Non-Renewable": "total_non_renewable",
}
_RESMV_TECH = {"β/α": "wind", "μυησ": "small_hydro", "σηθυα": "chp", "φ/β": "pv", "βιο": "biogas_biomass"}
_GREEK_FOLD = str.maketrans({"ς": "σ", "b": "β", "a": "α", "ά": "α", "ύ": "υ", "ή": "η", "ί": "ι", "ό": "ο", "έ": "ε"})


def _capacity_energy_charts() -> pd.DataFrame:
    cache = RAW_DIR / "energy_charts" / "installed_power_gr_yearly.json.gz"
    data = _cached_json(f"{EC_API}/installed_power", {"country": "gr", "time_step": "yearly", "installation_decommission": "false"}, cache, 24.0)
    rows = []
    for series in data["production_types"]:
        name = series["name"]
        is_energy = "capacity" in name.lower() and "battery" in name.lower()
        for year, value in zip(data["time"], series["data"], strict=False):
            if value is None:
                continue
            rows.append(
                {
                    "date": f"{year}-12-31",
                    "granularity": "year",
                    "technology": _EC_CAP_MAP.get(name, _slug(name)),
                    "value": round(float(value) * 1000.0, 3),
                    "unit": "MWh" if is_energy else "MW",
                    "scope": "Greece, total (as reported to ENTSO-E / energy-charts)",
                    "source": "energy-charts.info installed_power?country=gr",
                }
            )
    return pd.DataFrame(rows)


def _capacity_irena() -> pd.DataFrame:
    """IRENASTAT installed capacity for Greece (on-grid + off-grid, MW at year end)."""
    table_url = IRENA_TABLE
    try:
        meta_resp = _HTTP.get(table_url)
    except Exception:
        # the table id carries the release (e.g. 2026_H2): pick the current country table
        folder = _HTTP.get(IRENA_FOLDER)
        assert folder is not None
        ids = [t["id"] for t in folder.json() if t["id"].startswith("Country_ELECSTAT")]
        table_url = f"{IRENA_FOLDER}/{ids[0]}"
        meta_resp = _HTTP.get(table_url)
    assert meta_resp is not None
    meta = {v["code"]: v for v in meta_resp.json()["variables"]}
    tech = meta["Technology"]
    wanted = [code for code, text in zip(tech["values"], tech["valueTexts"], strict=True) if text in _IRENA_TECH]
    query = {
        "query": [
            {"code": "Country/area", "selection": {"filter": "item", "values": ["GRC"]}},
            {"code": "Technology", "selection": {"filter": "item", "values": wanted}},
            {"code": "Data Type", "selection": {"filter": "item", "values": ["1"]}},
            {"code": "Grid connection", "selection": {"filter": "item", "values": ["0"]}},
        ],
        "response": {"format": "json"},
    }
    cache = RAW_DIR / "irena" / "irenastat_capacity_grc.json.gz"
    if _is_fresh(cache, 24.0 * 30):
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        _HTTP._wait(urlparse(table_url).netloc)
        resp = _HTTP.session.post(table_url, json=query, timeout=120)
        resp.raise_for_status()
        data = json.loads(resp.content.decode("utf-8-sig"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wt", encoding="utf-8") as fh:
            json.dump(data, fh)
    codes = [c["code"] for c in data["columns"]]
    tech_text = dict(zip(tech["values"], tech["valueTexts"], strict=True))
    year_text = dict(zip(meta["Year"]["values"], meta["Year"]["valueTexts"], strict=True))
    rows = []
    for rec in data["data"]:
        key = dict(zip(codes, rec["key"], strict=False))
        try:
            value = float(rec["values"][0])
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "date": f"{year_text[key['Year']]}-12-31",
                "granularity": "year",
                "technology": _IRENA_TECH[tech_text[key["Technology"]]],
                "value": round(value, 3),
                "unit": "MW",
                "scope": "Greece, total (IRENA statistics, on-grid and off-grid)",
                "source": f"IRENA IRENASTAT {table_url.rsplit('/', 1)[-1]}",
            }
        )
    return pd.DataFrame(rows)


def _parse_resmv(content: bytes, name: str) -> pd.DataFrame | None:
    """Parse a daily RESMV workbook: hourly MV RES injections and installed power.

    Columns come in pairs per technology: energy (kWh) and installed power (kW).
    Returns one row per hour (hour index 1..24/25 of the Greek local day).
    """
    raw = _read_excel_any(content, name, sheet_name=0, header=None)
    header_row = None
    for i in range(min(6, len(raw))):
        if any("ΩΡΑ" in str(c).upper() for c in raw.iloc[i].tolist()):
            header_row = i
            break
    if header_row is None:
        return None
    header = [str(c).strip().lower() for c in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1 :]
    day = pd.to_datetime(body.iloc[:, 0].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")
    keep = day.notna()
    out = pd.DataFrame({"day": day[keep].to_numpy(), "hour": pd.to_numeric(body.iloc[:, 1][keep], errors="coerce").to_numpy()})
    for idx, label in enumerate(header):
        folded = label[:6].translate(_GREEK_FOLD)  # e.g. Latin "B/A" and Greek "Β/Α" are both used
        for key, tech in _RESMV_TECH.items():
            if folded.startswith(key):
                values = pd.to_numeric(body.iloc[:, idx][keep], errors="coerce").to_numpy(dtype="float64")
                if "ισχ" in label:  # "ΕΓΚ ΙΣΧΥΣ KW": installed power
                    # Files up to mid 2025 give installed power in MW although the header says kW
                    # (e.g. 1787 for PV); newer files really are in kW (e.g. 4842381).
                    in_kw = np.nanmedian(values) > 20000.0 if np.isfinite(values).any() else False
                    out[f"{tech}_inst_mw"] = values / 1000.0 if in_kw else values
                else:  # "ΕΝΕΡΓΕΙΑ KWh": hourly energy
                    out[f"{tech}_mwh"] = values / 1000.0
    return out if len(out.columns) > 2 else None


def fetch_resmv(sample_days: Iterable[dt.date], keep_raw: bool = False) -> pd.DataFrame:
    """Daily RESMV files (MV-connected RES, non certified metering) for selected days."""
    wanted = sorted(set(sample_days))
    if not wanted:
        return pd.DataFrame()
    parsed_path = RAW_DIR / "admie" / "_parsed" / "RESMV.parquet"
    cache = pd.read_parquet(parsed_path) if parsed_path.exists() else pd.DataFrame()
    have = set(pd.to_datetime(cache["day"]).dt.date) if len(cache) else set()
    todo = [d for d in wanted if d not in have]
    frames = [cache] if len(cache) else []
    if todo:
        listing = _admie_listing_chunked("RESMV", min(todo), max(todo))
        by_day: dict[dt.date, dict[str, str]] = {}
        for item in _admie_latest(listing):
            day = dt.datetime.strptime(item["file_fromdate"], "%d.%m.%Y").date()
            by_day[day] = item
        for day in todo:
            item = by_day.get(day)
            if item is None:
                continue
            name = item["file_path"].rsplit("/", 1)[-1]
            path = RAW_DIR / "admie" / "RESMV" / name
            content = _cached_bytes(item["file_path"], path, ok_404=True)
            if content is None:
                continue
            try:
                frame = _parse_resmv(content, name)
            except Exception as exc:
                log.warning("RESMV %s: %s", name, exc)
                frame = None
            if frame is not None and len(frame):
                frame["src_file"] = name
                frames.append(frame)
            if not keep_raw:
                path.unlink(missing_ok=True)
        if frames:
            cache = pd.concat(frames, ignore_index=True).drop_duplicates(["day", "hour"], keep="last")
            parsed_path.parent.mkdir(parents=True, exist_ok=True)
            cache.to_parquet(parsed_path, index=False)
    if not len(cache):
        return cache
    return cache[pd.to_datetime(cache["day"]).dt.date.isin(wanted)].reset_index(drop=True)


def _capacity_admie_mv(first: dt.date = dt.date(2015, 11, 1)) -> pd.DataFrame:
    """Monthly installed power of MV-connected RES, read from the RESMV file of each month's 15th."""
    today = _utc_now().date()
    days = [d.date() for d in pd.date_range(first, today - dt.timedelta(days=3), freq="MS") + pd.Timedelta(days=14)]
    days = [d for d in days if d < today - dt.timedelta(days=2)]
    df = fetch_resmv(days)
    rows = []
    if not len(df):
        return pd.DataFrame(rows)
    inst_cols = [c for c in df.columns if c.endswith("_inst_mw")]
    daily = df.groupby("day")[inst_cols].max()  # installed power is repeated for every hour
    for day, rec in daily.iterrows():
        for col in inst_cols:
            if pd.notna(rec[col]) and rec[col] > 0:
                rows.append(
                    {
                        "date": f"{pd.Timestamp(day):%Y-%m-%d}",
                        "granularity": "month",
                        "technology": col.removesuffix("_inst_mw"),
                        "value": round(float(rec[col]), 3),
                        "unit": "MW",
                        "scope": "Greece mainland, MV distribution-connected only (ADMIE RESMV, non certified)",
                        "source": "ADMIE file API, category RESMV",
                    }
                )
    return pd.DataFrame(rows)


def fetch_installed_capacity() -> pd.DataFrame:
    """Installed capacity by technology, long format CSV ``installed_capacity_gr.csv``.

    Three real sources with different scope are stacked (column ``scope``):
    energy-charts yearly, IRENA yearly and ADMIE RESMV monthly (MV-connected RES
    only).  Battery storage has no open consistent series; rows appear only if
    a source reports them.
    """
    parts = []
    for fn in (_capacity_energy_charts, _capacity_irena, _capacity_admie_mv):
        try:
            parts.append(fn())
        except Exception as exc:  # one source down must not hide the others
            log.warning("installed capacity: %s failed: %s", fn.__name__, exc)
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out = out.sort_values(["source", "technology", "date"]).reset_index(drop=True)
    out.to_csv(PROCESSED_DIR / "installed_capacity_gr.csv", index=False)
    log.info("installed_capacity_gr: %d rows", len(out))
    return out


def _parse_scada_realization(content: bytes, name: str) -> dict[str, float] | None:
    """Daily energy (MWh) of HV-connected PV and of all HV RES from a SystemRealizationSCADA file.

    In the block of RES stations the photovoltaic stations carry the suffix
    ``(PV)``; ``TOTAL RES`` is the sum of all telemetered stations connected to
    the transmission system.  The last numeric cell of each row is the day sum.
    """
    raw = _read_excel_any(content, name, sheet_name="System_Production", header=None)
    labels = raw.iloc[:, 1].astype(str).str.strip()

    def day_sum(row: pd.Series) -> float:
        vals = pd.to_numeric(row.iloc[2:], errors="coerce").dropna()
        return float(vals.iloc[-1]) if len(vals) else float("nan")

    pv_rows = raw[labels.str.endswith("(PV)")]
    total = raw[labels == "TOTAL RES"]
    if total.empty:
        return None
    return {
        "hv_pv_mwh": float(sum(day_sum(r) for _, r in pv_rows.iterrows())) if len(pv_rows) else 0.0,
        "hv_res_total_mwh": day_sum(total.iloc[0]),
        "hv_pv_stations": float(len(pv_rows)),
    }


def check_pv_definition(days: Iterable[dt.date]) -> pd.DataFrame:
    """Compare ENTSO-E "Solar" for Greece with ADMIE metering for sample days.

    For each day: daily energy of ENTSO-E solar (``system_gr.parquet``), of the
    PV stations connected to the transmission system (SystemRealizationSCADA)
    and of the MV distribution-connected PV (RESMV).  LV rooftop PV is not
    metered in any open ADMIE file.  Result: ``pv_definition_check.csv``.
    """
    days = sorted(set(days))
    mv = fetch_resmv(days)
    mv_day = mv.groupby(pd.to_datetime(mv["day"]).dt.date)[["pv_mwh", "pv_inst_mw"]].agg({"pv_mwh": "sum", "pv_inst_mw": "max"})
    listing = {
        dt.datetime.strptime(i["file_fromdate"], "%d.%m.%Y").date(): i
        for i in _admie_latest(_admie_listing_chunked("SystemRealizationSCADA", min(days), max(days)))
        if i["file_path"].lower().endswith((".xls", ".xlsx"))
    }
    system = pd.read_parquet(PROCESSED_DIR / "system_gr.parquet").set_index("ts_utc")["pv_mw"]
    local = system.tz_convert("Europe/Athens")
    rows = []
    for day in days:
        rec: dict[str, Any] = {"day": f"{day:%Y-%m-%d}"}
        item = listing.get(day)
        if item is not None:
            name = item["file_path"].rsplit("/", 1)[-1]
            path = RAW_DIR / "admie" / "SystemRealizationSCADA" / name
            content = _cached_bytes(item["file_path"], path, ok_404=True)
            if content is not None:
                try:
                    rec.update(_parse_scada_realization(content, name) or {})
                except Exception as exc:
                    log.warning("SCADA %s: %s", name, exc)
                path.unlink(missing_ok=True)
        if day in mv_day.index:
            rec["mv_pv_mwh"] = float(mv_day.loc[day, "pv_mwh"])
            rec["mv_pv_inst_mw"] = float(mv_day.loc[day, "pv_inst_mw"])
        sel = local[local.index.date == day]
        rec["entsoe_solar_mwh"] = float(sel.sum()) if sel.notna().sum() >= 23 else float("nan")
        rows.append(rec)
    out = pd.DataFrame(rows)
    if {"hv_pv_mwh", "mv_pv_mwh"} <= set(out.columns):
        out["admie_hv_plus_mv_pv_mwh"] = out["hv_pv_mwh"] + out["mv_pv_mwh"]
        out["entsoe_over_admie"] = out["entsoe_solar_mwh"] / out["admie_hv_plus_mv_pv_mwh"]
    out.round(3).to_csv(PROCESSED_DIR / "pv_definition_check.csv", index=False)
    return out


# --------------------------------------------------------------------------- 6. fuel prices
def _world_bank_monthly() -> pd.DataFrame:
    """World Bank Pink Sheet monthly: 'Natural gas, Europe' in USD/mmbtu (TTF since 2015-04)."""
    url = WB_FALLBACK
    try:
        landing = _HTTP.get(WB_LANDING)
        assert landing is not None
        found = re.findall(r"https://thedocs\.worldbank\.org/[^\"']*CMO-Historical-Data-Monthly\.xlsx", landing.text)
        if found:
            url = found[0]
    except Exception as exc:
        log.warning("World Bank landing page not readable (%s); using fallback URL", exc)
    content = _cached_bytes(url, RAW_DIR / "worldbank" / "CMO-Historical-Data-Monthly.xlsx", 24.0)
    assert content is not None
    raw = _read_excel_any(content, "cmo.xlsx", sheet_name="Monthly Prices", header=None)
    header_row = next(i for i in range(12) if "Natural gas, Europe" in [str(c).strip() for c in raw.iloc[i].tolist()])
    col = [str(c).strip() for c in raw.iloc[header_row].tolist()].index("Natural gas, Europe")
    body = raw.iloc[header_row + 2 :]
    month = pd.to_datetime(body.iloc[:, 0].astype(str).str.replace("M", "-", regex=False), format="%Y-%m", errors="coerce")
    df = pd.DataFrame({"month": month.to_numpy(), "ttf_usd_mmbtu": pd.to_numeric(body.iloc[:, col], errors="coerce").to_numpy()})
    df = df.dropna()
    df.attrs["url"] = url
    return df


def _ecb_usd_per_eur() -> pd.DataFrame:
    """ECB monthly average reference rate, USD per 1 EUR."""
    cache = RAW_DIR / "ecb" / "exr_m_usd_eur.csv"
    if not _is_fresh(cache, 24.0):
        resp = _HTTP.get(ECB_FX, params={"format": "csvdata", "startPeriod": "2014-01"})
        assert resp is not None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(resp.content)
    raw = pd.read_csv(cache)
    return pd.DataFrame(
        {"month": pd.to_datetime(raw["TIME_PERIOD"], format="%Y-%m"), "usd_per_eur": raw["OBS_VALUE"].astype(float)}
    )


def _eex_eua_auctions(first_year: int = 2015) -> pd.DataFrame:
    """EEX primary market auction results (one row per auction), all available years."""
    frames = []
    this_year = _utc_now().year
    for year in range(first_year, this_year + 1):
        content, used_ext = None, ""
        for ext in ("xlsx", "xls"):
            cache = RAW_DIR / "eex" / f"eua_auctions_{year}.{ext}"
            content = _cached_bytes(EEX_AUCTION.format(year=year, ext=ext), cache, 24.0 if year == this_year else None, ok_404=True)
            if content is not None:
                used_ext = ext
                break
        if content is None:
            log.info("EEX auction report %s: not published at the known URL", year)
            continue
        raw = _read_excel_any(content, f"x.{used_ext}", sheet_name=0, header=None)
        header_row = next((i for i in range(12) if "Auction Name" in [str(c).strip() for c in raw.iloc[i].tolist()]), None)
        if header_row is None:
            log.warning("EEX auction report %s: header not found", year)
            continue
        header = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.iloc[header_row].tolist()]
        body = raw.iloc[header_row + 1 :]

        def col(prefix: str, header: list[str] = header, body: pd.DataFrame = body) -> pd.Series | None:
            idx = next((i for i, h in enumerate(header) if h.lower().startswith(prefix)), None)
            return None if idx is None else body.iloc[:, idx]

        date, name, price, status = col("date"), col("auction name"), col("auction price"), col("status")
        if date is None or name is None or price is None:
            log.warning("EEX auction report %s: expected columns not found", year)
            continue
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(date, errors="coerce"),
                "auction_name": name.astype(str),
                # reports before 2018 have no status column: a clearing price means success
                "status": status.astype(str).str.lower() if status is not None else "successful",
                "eua_auction_price_eur_t": pd.to_numeric(price, errors="coerce"),
            }
        ).dropna(subset=["date", "eua_auction_price_eur_t"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["status"].str.contains("success", na=False)]
    # keep general allowances only (EUAA aviation auctions are a different product)
    out = out[~out["auction_name"].str.contains("EUAA|aviation", case=False, na=False)]
    return out.sort_values("date").reset_index(drop=True)


def fetch_fuel_prices(start: str = "2015-01") -> pd.DataFrame:
    """Monthly TTF gas price and EUA price, ``fuel_prices.csv``.

    * ``ttf_eur_mwh`` = World Bank "Natural gas, Europe" (USD/mmbtu, TTF) divided
      by the ECB monthly average USD-per-EUR rate and by 0.29307107 MWh/mmbtu.
    * ``eua_eur_t`` = unweighted monthly mean of the clearing prices of all
      successful EEX primary auctions of general allowances in that month.
    The individual auctions are saved as ``eua_auctions_daily.csv``.
    """
    gas = _world_bank_monthly()
    fx = _ecb_usd_per_eur()
    out = gas.merge(fx, on="month", how="outer").sort_values("month")
    out["ttf_eur_mwh"] = out["ttf_usd_mmbtu"] / out["usd_per_eur"] / MMBTU_TO_MWH
    try:
        auctions = _eex_eua_auctions()
        auctions.assign(date=auctions["date"].dt.strftime("%Y-%m-%d")).to_csv(
            PROCESSED_DIR / "eua_auctions_daily.csv", index=False
        )
        monthly = auctions.groupby(auctions["date"].dt.to_period("M"))["eua_auction_price_eur_t"].agg(["mean", "count"])
        monthly.index = monthly.index.to_timestamp()
        out = out.merge(
            monthly.rename(columns={"mean": "eua_eur_t", "count": "eua_n_auctions"}),
            left_on="month", right_index=True, how="left",
        )
    except Exception as exc:
        log.warning("EEX EUA auctions failed: %s", exc)
        out["eua_eur_t"] = np.nan
        out["eua_n_auctions"] = np.nan
    out = out[out["month"] >= pd.Timestamp(start)].reset_index(drop=True)
    out["month"] = out["month"].dt.strftime("%Y-%m")
    out["ttf_source"] = "World Bank Pink Sheet 'Natural gas, Europe' (TTF); FX: ECB EXR.M.USD.EUR.SP00.A"
    out["eua_source"] = "EEX EUA primary market auction reports (monthly mean of auction clearing prices)"
    out = out.round({"ttf_usd_mmbtu": 4, "usd_per_eur": 5, "ttf_eur_mwh": 3, "eua_eur_t": 3})
    out.to_csv(PROCESSED_DIR / "fuel_prices.csv", index=False)
    log.info("fuel_prices: %d months %s .. %s", len(out), out.month.min(), out.month.max())
    return out


# --------------------------------------------------------------------------- housekeeping
def raw_tree_size_mb(sub: str = "") -> float:
    """Size of ``data/raw/<sub>`` in MB."""
    base = RAW_DIR / sub
    if not base.exists():
        return 0.0
    return sum(p.stat().st_size for p in base.rglob("*") if p.is_file()) / 1e6


def prune_raw(sub: str, keep: tuple[str, ...] = ("_parsed", "_listings")) -> None:
    """Delete raw downloads below ``data/raw/<sub>`` except the compact caches in ``keep``."""
    base = RAW_DIR / sub
    if not base.exists():
        return
    for child in base.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def fetch_all(crosscheck: bool = True, neighbours: bool = True) -> None:
    """Regenerate every processed market dataset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fetch_dam_prices(crosscheck=crosscheck)
    fetch_system()
    fetch_imbalance()
    fetch_curtailment()
    fetch_res_forecast()
    fetch_installed_capacity()
    fetch_fuel_prices()
    if neighbours:
        fetch_neighbour_prices()
