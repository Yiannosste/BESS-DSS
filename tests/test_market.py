"""Market engine (REQ-13, REQ-14). Uses the committed calibration; skipped when it is absent."""

import numpy as np
import pytest

from bessdss.market.cqr import fit_surface
from bessdss.market.fleet import FleetDispatch, StorageUnit
from bessdss.market.library import DEFAULT_CAL_DIR
from bessdss.market.longterm import MarketCalibration, MarketSimulator, YearDrivers, annualised_cost

pytestmark = pytest.mark.skipif(not (DEFAULT_CAL_DIR / "market_calibration.npz").exists(), reason="no market calibration")


@pytest.fixture(scope="module")
def sim():
    return MarketSimulator(MarketCalibration.load(DEFAULT_CAL_DIR))


def _drivers(**kw):
    base = dict(year=2030, pv_gw=17.0, wind_gw=7.5, load_twh=55.0, flat_load_gw=0.3, bess_gw=2.0, bess_duration_h=2.0,
                pumped_gw=1.4, pumped_duration_h=6.0, gas_eur_mwh=30.0, co2_eur_t=95.0, saturation_shift_gw=1.0)
    base.update(kw)
    return YearDrivers(**base)


def test_surface_is_monotone():
    rng = np.random.default_rng(0)
    z = rng.uniform(-3, 8, 4000)
    pi = np.clip(0.2 * z + rng.normal(0, 0.2, 4000), 0, None)
    s = fit_surface(z, pi)
    assert np.all(np.diff(s.q, axis=0) >= -1e-12)  # more residual load never lowers the price
    assert np.all(np.diff(s.q, axis=1) >= -1e-12)
    u = s.rank(z, pi, rng)
    err = np.abs(s.price(z, u) - pi)  # rank and price are consistent inverses (tails beyond the 1 % grid are clipped)
    assert np.median(err) < 0.01 and np.quantile(err, 0.97) < 0.1


def test_fleet_planner_flattens_net_load():
    b = np.linspace(-2, 8, 41)
    seg = np.clip(0.5 * (b[1:] + b[:-1]) * 20, 0, None)
    rl = 3 + 2 * np.sin(np.linspace(0, 2 * np.pi, 24) - 2)
    fd = FleetDispatch(24, 1.0, b, 1)
    r = fd.solve(rl, seg, [StorageUnit("b", 1.0, 2.0)])
    net = r["net_load_gw"]
    assert net.max() < rl.max() and net.min() > rl.min()
    assert np.all(r["charge_gw"] <= 1 + 1e-9) and np.all(r["soc_gwh"] <= 2 + 1e-9)


def test_more_storage_lowers_spread(sim):
    d = _drivers()
    lo = sim.simulate_year(d, 2019, seed=0, bess_gw=1.0)
    hi = sim.simulate_year(d, 2019, seed=0, bess_gw=8.0)
    assert hi.stats()["tb4"] < lo.stats()["tb4"] - 10
    assert hi.stats()["zero_hours"] <= lo.stats()["zero_hours"]
    m_lo = sim.marginal_battery_margin(lo, 2.0)
    m_hi = sim.marginal_battery_margin(hi, 2.0)
    assert m_hi < 0.6 * m_lo  # the rebound: entry destroys the margin it chases


def test_more_pv_lowers_capture_price(sim):
    a = sim.simulate_year(_drivers(pv_gw=14.0), 2019, seed=0).stats()
    b = sim.simulate_year(_drivers(pv_gw=22.0), 2019, seed=0).stats()
    assert b["pv_capture_price"] < a["pv_capture_price"]
    assert b["zero_hours"] > a["zero_hours"]


def test_gas_price_moves_the_level(sim):
    a = sim.simulate_year(_drivers(gas_eur_mwh=20.0), 2019, seed=0).stats()
    b = sim.simulate_year(_drivers(gas_eur_mwh=50.0), 2019, seed=0).stats()
    assert b["mean_price"] > a["mean_price"] * 1.3


def test_in_sample_statistics():
    """Resampled ranks on the calibration window reproduce mean price, daily spread and zero-price hours."""
    import warnings

    from bessdss.market.calibrate import build_calibration

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal, diag = build_calibration()
    except FileNotFoundError:
        pytest.skip("processed market data not available")
    h = diag["_history"].dropna()
    h = h[h.index >= "2025-01-01"]
    h = h.iloc[: len(h) // 24 * 24]
    rng = np.random.default_rng(1)
    sim = MarketSimulator(cal)
    days = len(h) // 24
    months = h.index.month.to_numpy().reshape(days, 24)[:, 12]
    we = h.index.dayofweek.to_numpy().reshape(days, 24)[:, 12] >= 5
    u = sim.sample_ranks(months, we, rng).reshape(-1)
    p = h.M.to_numpy() * cal.surface.price(h.z.to_numpy(), u)
    act = h.price.to_numpy()

    def tb4(x):
        s = np.sort(x.reshape(days, 24), axis=1)
        return (s[:, -4:].mean(1) - s[:, :4].mean(1)).mean()

    assert p.mean() == pytest.approx(act.mean(), rel=0.05)
    assert tb4(p) == pytest.approx(tb4(act), rel=0.10)
    assert (p <= 0.5).sum() == pytest.approx((act <= 0.5).sum(), rel=0.15)


def test_annualised_cost():
    assert annualised_cost(100.0, 0.02, 0.08, 15) == pytest.approx(100 * (0.08 * 1.08**15 / (1.08**15 - 1) + 0.02))
