"""Hourly surface irradiance (GHI) and 2 m temperature from NASA POWER for the representative points.

Why: PVGIS ends in 2023, but zero prices and curtailment took off in 2024 to 2026. To rebuild the
POTENTIAL (pre-curtailment) national PV output for those years we need irradiance up to today.
NASA POWER (MERRA-2 / CERES, about 0.5 degree grid) is public and has no key. Data are modelled
(reanalysis and satellite), not measured on the ground.
"""
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, "scripts")
from fetch_pvgis_points import POINTS  # noqa: E402

URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def fetch(lat, lon, year, end=None):
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": f"{year}0101",
        "end": end or f"{year}1231",
        "format": "JSON",
        "time-standard": "UTC",
    }
    for attempt in range(5):
        try:
            r = requests.get(URL, params=params, timeout=180)
            if r.status_code == 200:
                par = r.json()["properties"]["parameter"]
                df = pd.DataFrame(par)
                df.index = pd.to_datetime(df.index, format="%Y%m%d%H", utc=True)
                return df
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"NASA POWER failed for {lat},{lon},{year}")


if __name__ == "__main__":
    frames = {}
    for name, (lat, lon) in POINTS.items():
        parts = []
        for year in range(2015, 2027):
            end = "20261109" if year == 2026 else None
            parts.append(fetch(lat, lon, year, end))
            time.sleep(0.5)
        df = pd.concat(parts)
        df = df.where(df > -900)  # -999 marks not yet available
        frames[name] = df
        print(name, len(df), df["ALLSKY_SFC_SW_DWN"].last_valid_index(), flush=True)
    out = pd.concat(frames, axis=1)
    out.columns = [f"{a}__{b}" for a, b in out.columns]
    out.index.name = "ts_utc"
    out.to_parquet("data/processed/nasa_power_points.parquet")
    print("saved", out.shape)
