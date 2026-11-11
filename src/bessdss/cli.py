"""Command line interface.

    bessdss study examples/parks/park_a_fip_1mw.yaml --out results/park_a
    bessdss calibrate          # re-estimate the market engine from data/processed
    bessdss market             # print the scenario summary table
    bessdss compliance park.yaml --category 11A --power 0.5 --energy 1.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bessdss", description="Decision support for adding a battery to an existing Greek PV park")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("study", help="run the full study for one park")
    s.add_argument("park_yaml")
    s.add_argument("--out", default=None)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--paths", type=int, default=2000)
    sub.add_parser("calibrate", help="re-estimate the market engine from the processed data")
    m = sub.add_parser("market", help="print scenario statistics")
    m.add_argument("--years", default="2027,2030,2035,2040")
    c = sub.add_parser("compliance", help="regulatory check of one design")
    c.add_argument("park_yaml")
    c.add_argument("--category", default="11A")
    c.add_argument("--power", type=float, required=True)
    c.add_argument("--energy", type=float, required=True)
    r = sub.add_parser("report", help="rebuild report.html from an existing results.json")
    r.add_argument("results_json")
    a = ap.parse_args(argv)

    if a.cmd == "study":
        from bessdss.config import Study
        from bessdss.pipeline import run_study

        study = Study.from_yaml(a.park_yaml)
        out = a.out or f"results/{Path(a.park_yaml).stem}"
        run_study(study, out, n_workers=a.workers, n_paths=a.paths)
    elif a.cmd == "calibrate":
        import warnings

        from bessdss.market.calibrate import build_calibration
        from bessdss.market.library import DEFAULT_CAL_DIR

        warnings.simplefilter("ignore")
        cal, diag = build_calibration(weather_years=tuple(range(2015, 2026)))
        diag.pop("_history", None)
        cal.save(DEFAULT_CAL_DIR)
        print(json.dumps(diag, indent=1, default=str, ensure_ascii=False))
    elif a.cmd == "market":
        from bessdss.market.library import ScenarioLibrary

        lib = ScenarioLibrary()
        rows = lib.summary(years=tuple(int(x) for x in a.years.split(",")), weather_years=[2016, 2019, 2022])
        cols = list(rows[0])
        print("  ".join(f"{c:>14}" for c in cols))
        for row in rows:
            print("  ".join(f"{str(row[c]):>14}" for c in cols))
    elif a.cmd == "compliance":
        from bessdss.config import Study
        from bessdss.regulatory.compliance import check

        for ch in check(Study.from_yaml(a.park_yaml), a.category, a.power, a.energy):
            print(f"[{ch['status']:4}] {ch['id']} {ch['title']}: {ch['text']}")
            for rule in ch["rules"]:
                print(f"         {rule['id']} {rule['citation'][:110]} (verified {rule['verified_on']}, {rule['confidence']})")
    elif a.cmd == "report":
        from bessdss.report.build import rebuild

        print(rebuild(a.results_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
