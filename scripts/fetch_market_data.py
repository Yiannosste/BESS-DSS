#!/usr/bin/env python
"""Regenerate every processed Greek market dataset from its public source.

Usage
-----
    python scripts/fetch_market_data.py                 # everything
    python scripts/fetch_market_data.py dam system      # selected datasets
    python scripts/fetch_market_data.py --no-crosscheck # skip the HEnEx download (about 350 MB)
    python scripts/fetch_market_data.py --keep-raw      # keep raw ADMIE/HEnEx workbooks (about 1.5 GB)

The run is idempotent: closed periods are served from ``data/raw`` caches and
only new or republished files are downloaded.  A first complete run downloads
roughly 1.5 GB (mostly ADMIE ISP workbooks and HEnEx yearly archives) and takes
45 to 60 minutes because requests are rate limited on purpose.  Raw workbooks
are deleted after parsing unless ``--keep-raw`` is given; compact parsed caches
stay in ``data/raw/admie/_parsed`` and ``data/raw/henex``.

Extra packages on top of the project dependencies: ``openpyxl`` (xlsx) and
``xlrd`` (legacy xls files of ADMIE and EEX).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bessdss.data import fetch_market as fm  # noqa: E402

DATASETS = ("dam", "system", "imbalance", "curtailment", "forecast", "capacity", "fuel", "neighbours")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", metavar="DATASET", help=f"datasets to refresh (default: all): {', '.join(DATASETS)}")
    parser.add_argument("--no-crosscheck", action="store_true", help="do not download HEnEx results for the DAM cross-check")
    parser.add_argument("--keep-raw", action="store_true", help="keep raw ADMIE and HEnEx workbooks on disk")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}")
    selected = args.datasets or list(DATASETS)
    fm.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    steps: dict[str, Callable[[], object]] = {
        "dam": lambda: fm.fetch_dam_prices(crosscheck=not args.no_crosscheck, keep_raw_henex=args.keep_raw),
        "system": fm.fetch_system,
        "imbalance": lambda: fm.fetch_imbalance(keep_raw=args.keep_raw),
        "curtailment": lambda: fm.fetch_curtailment(keep_raw=args.keep_raw),
        "forecast": fm.fetch_res_forecast,  # after system and curtailment: it joins both
        "capacity": fm.fetch_installed_capacity,
        "fuel": fm.fetch_fuel_prices,
        "neighbours": fm.fetch_neighbour_prices,
    }
    failed: list[str] = []
    for name in DATASETS:
        if name not in selected:
            continue
        t0 = time.time()
        try:
            steps[name]()
            logging.info("%s done in %.0f s", name, time.time() - t0)
        except Exception:  # keep going: one source being down must not block the others
            logging.exception("%s FAILED", name)
            failed.append(name)

    logging.info("raw cache size: %.1f MB", fm.raw_tree_size_mb())
    if failed:
        logging.error("failed datasets: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
