"""Download PVGIS multi-year hourly series for the national representative points and example parks.

The representative points approximate where Greek PV capacity sits. They are used only to build a
weather-coherent national PV profile for the market model (same weather year as the park).
"""
import sys

import pandas as pd

from bessdss.pv.pvgis import fetch_pvgis_hourly

# name: (lat, lon). Regions with most ground-mounted PV in Greece.
POINTS = {
    "thessaly_larissa": (39.64, 22.42),
    "central_macedonia_kilkis": (40.99, 22.87),
    "west_macedonia_kozani": (40.30, 21.79),
    "central_greece_viotia": (38.44, 23.10),
    "fthiotida_lamia": (38.90, 22.43),
    "peloponnese_tripoli": (37.51, 22.37),
    "west_greece_agrinio": (38.62, 21.41),
    "thrace_xanthi": (41.13, 24.89),
}

if __name__ == "__main__":
    out = {}
    for name, (lat, lon) in POINTS.items():
        df = fetch_pvgis_hourly(lat, lon, tilt_deg=25, azimuth_deg=0)
        out[name] = df["p_kw_per_kwp"]
        print(name, len(df), round(df["p_kw_per_kwp"].sum() / 19, 1), "kWh/kWp/yr", flush=True)
    nat = pd.DataFrame(out)
    nat.to_parquet("data/processed/pvgis_national_points.parquet")
    print("saved", nat.shape, file=sys.stderr)
