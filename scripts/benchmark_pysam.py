"""Benchmark of our dispatch engine against NREL SAM (PySAM) on identical inputs.

Same PV profile (PVGIS, Larissa 2023, 1.1 MWp behind a 1 MW inverter), same battery
(0.5 MW / 1.222 MWh nameplate, AC-coupled, PV charging only), same REAL hourly Greek day-ahead
prices (Nov 2025 to Oct 2026), no injection caps (SAM cannot represent the Greek ones).

SAM: CustomGeneration + Battery, merchant-plant price signal, automated look-ahead dispatch.
Ours: LP with perfect foresight of the day and the clock-rule baseline.
Revenue is computed outside both tools with the same formula: sum(grid export x price).

Run:  python scripts/benchmark_pysam.py     (needs: pip install NREL-PySAM)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    p = pd.read_parquet(ROOT / "data/processed/dam_price_gr.parquet").set_index("ts_utc")["price_eur_mwh"]
    p = p[(p.index >= "2025-11-01") & (p.index < "2026-11-01")].resample("1h").mean().to_numpy()[:8760]
    pv = pd.read_parquet(ROOT / "data/processed/pvgis_national_points.parquet")["thessaly_larissa"]
    pv = pv[pv.index.year == 2023].to_numpy()[:8760] * 1100.0  # kW DC for 1.1 MWp
    # shift PV so that it lines up with a Nov-to-Oct price year (same day-of-year, months rotated)
    pv = np.roll(pv, -int((pd.Timestamp("2023-11-01") - pd.Timestamp("2023-01-01")).days) * 24)
    ac = np.minimum(pv * 0.98, 1000.0)
    return p, ac


def run_sam(price, ac_kw, power_kw=500.0, energy_kwh=1222.2, custom_dispatch_kw=None):
    import PySAM.Battery as B
    import PySAM.BatteryTools as BT
    import PySAM.CustomGeneration as CG

    cfg = "CustomGenerationBatteryMerchantPlant"
    g = CG.default(cfg)
    b = B.from_existing(g, cfg)
    g.Plant.spec_mode = 1
    g.Plant.user_capacity_factor = 0
    g.Plant.energy_output_array = [float(x) for x in ac_kw]
    g.Plant.system_capacity = 1000.0
    g.Lifetime.system_use_lifetime_output = 0
    g.Lifetime.analysis_period = 1
    g.execute(0)
    BT.battery_model_sizing(b, power_kw, energy_kwh, 500.0)
    b.BatterySystem.batt_ac_or_dc = 1
    # SAM adds cell (voltage) losses of about 8 % per round trip on top of the converters: 97.9 % each way
    # gives an overall AC round-trip efficiency close to the 88 % used in our model
    b.BatterySystem.batt_ac_dc_efficiency = 97.9
    b.BatterySystem.batt_dc_ac_efficiency = 97.9
    b.BatterySystem.batt_replacement_option = 0
    b.BatteryCell.batt_minimum_SOC = 5.0
    b.BatteryCell.batt_maximum_SOC = 95.0
    d = b.BatteryDispatch
    d.batt_dispatch_choice = 0  # automated economic dispatch, perfect look-ahead
    if custom_dispatch_kw is not None:
        d.batt_dispatch_choice = 2  # follow a given battery power series (kW, + discharge, - charge)
        d.batt_custom_dispatch = [float(x) for x in custom_dispatch_kw]
    d.batt_dispatch_auto_can_charge = 1
    d.batt_dispatch_auto_can_gridcharge = 0
    d.batt_look_ahead_hours = 24
    d.batt_dispatch_update_frequency_hours = 24
    d.batt_cycle_cost_choice = 1
    d.batt_cycle_cost = [0.014]  # $/kWh cycled, the same 14 EUR/MWh our engine uses
    b.PriceSignal.forecast_price_signal_model = 1
    b.PriceSignal.mp_enable_energy_market_revenue = 1
    b.PriceSignal.mp_energy_market_revenue = [[1.0, float(x)] for x in price]
    b.Lifetime.system_use_lifetime_output = 0
    b.Lifetime.analysis_period = 1
    b.execute(0)
    o = b.Outputs
    grid = np.array(o.grid_power)[:8760]  # kW, positive = export
    dis = np.array(o.batt_to_grid)[:8760]
    extra = {}
    for k in ("batt_annual_charge_energy", "batt_annual_discharge_energy", "batt_annual_energy_loss", "batt_annual_energy_system_loss"):
        try:
            extra[k] = float(np.array(getattr(o, k)).ravel()[1 if np.size(getattr(o, k)) > 1 else 0]) / 1000.0
        except Exception:
            pass
    return dict(
        **extra,
        revenue=float((grid * price).sum() / 1000.0), export_mwh=float(np.clip(grid, 0, None).sum() / 1000.0),
        batt_to_grid_mwh=float(dis.sum() / 1000.0), rte=float(o.average_battery_roundtrip_efficiency),
        cycles=float(np.array(o.batt_cycles)[-1]) if len(o.batt_cycles) else float("nan"),
    )


def run_ours(price, ac_kw, power_mw=0.5, energy_mwh=1.2222):
    from bessdss.config import Park, RegulatoryConfig, SupportScheme
    from bessdss.dispatch.lp import DispatchParams
    from bessdss.dispatch.simulate import YearInputs, simulate_pv_only, simulate_with_battery

    park = Park(name="b", lat=39.64, lon=22.42, pv_dc_kwp=1100, pv_ac_kw=1000, grid_export_limit_kw=1000, eta_inverter=1.0,
                support=SupportScheme(kind="merchant"))
    # no Greek caps for a like-for-like comparison: 11A window emptied
    reg = RegulatoryConfig(cap_11a_frac_of_installed=10.0)
    y = YearInputs(1.0, ac_kw / 1000.0, price, np.zeros(8760), steps_per_day=24)
    eta = np.sqrt(0.88)
    dp = DispatchParams(power_mw, energy_mwh * 0.9, eta, eta, degradation_cost_eur_mwh=14.0)
    base = simulate_pv_only(park, reg, y, 2026)
    out = {"pv_only_revenue": base.revenue_eur}
    for pol in ("perfect", "persistence", "rules"):
        r = simulate_with_battery(park, reg, y, 2026, dp, policy=pol, lookahead_days=0, keep_traces=(pol == "perfect"))
        out[pol] = dict(revenue=r.revenue_eur, gain=r.revenue_eur - base.revenue_eur, discharged_mwh=r.discharged_mwh, cycles=r.fec)
        if pol == "perfect":
            out["_schedule_kw"] = (r.d - r.c) * 1000.0
            out[pol]["charged_mwh"] = float(r.c.sum())
    return out


if __name__ == "__main__":
    price, ac = inputs()
    ours = run_ours(price, ac)
    sched = ours.pop("_schedule_kw")
    try:
        idle = run_sam(price, ac, custom_dispatch_kw=np.zeros(8760))  # SAM's own baseline (it applies its own system losses)
        sam = run_sam(price, ac)
        sam["gain"] = sam["revenue"] - idle["revenue"]
        # physics cross-check: SAM's detailed battery model following OUR schedule
        phys = run_sam(price, ac, custom_dispatch_kw=sched)
        phys["gain"] = phys["revenue"] - idle["revenue"]
        sam["baseline_revenue"] = phys["baseline_revenue"] = idle["revenue"]
    except Exception as exc:  # PySAM missing or API mismatch: report, do not hide
        sam, phys = {"error": repr(exc)}, {"error": repr(exc)}
    res = {"inputs": {"pv_mwh": float(ac.sum() / 1000), "mean_price": float(price.mean())}, "ours": ours,
           "pysam_own_dispatch": sam, "pysam_physics_with_our_schedule": phys}
    print(json.dumps(res, indent=1))
    out = ROOT / "docs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_pysam.json").write_text(json.dumps(res, indent=1))
    sys.exit(0)
