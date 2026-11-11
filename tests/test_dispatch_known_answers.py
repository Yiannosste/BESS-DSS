"""Known-answer tests for the dispatch LP (REQ-06, REQ-07).

Every test is a toy case where the right answer can be computed by hand.
All inputs here are synthetic by construction and are not market data.
"""

import numpy as np
import pytest

from bessdss.dispatch.lp import DispatchLP, DispatchParams


def _solve(p, pv, cap, r_pv, r_dis, c_imp=None, imp=None, dt=1.0, **kw):  # noqa: PLR0913
    lp = DispatchLP(len(pv), dt, p)
    return lp.solve(np.array(pv, float), np.array(cap, float), np.array(r_pv, float), np.array(r_dis, float),
                    None if c_imp is None else np.array(c_imp, float),
                    None if imp is None else np.array(imp, float), **kw)


def test_flat_price_no_arbitrage():
    """Flat prices and round-trip losses: the battery must stay idle."""
    p = DispatchParams(power_mw=1, energy_usable_mwh=2, eta_charge=0.9, eta_discharge=0.9)
    T = 24
    pv = [0] * 6 + [1] * 12 + [0] * 6
    res = _solve(p, pv, [1] * T, [50] * T, [50] * T)
    assert res.d.sum() == pytest.approx(0, abs=1e-9)
    assert res.objective == pytest.approx(50 * 12)


def test_two_price_arbitrage_analytic():
    """PV 1 MW for 2 h at 10 EUR, evening 2 h at 100 EUR, battery 1 MW / 2 MWh.

    Charging 2 MWh of PV stores 2*eta_c, discharge delivers 2*eta_c*eta_d MWh.
    Gain versus selling PV directly = 2*(eta_c*eta_d*100 - 10).
    """
    ec, ed = 0.95, 0.9
    p = DispatchParams(power_mw=1, energy_usable_mwh=10, eta_charge=ec, eta_discharge=ed)
    pv = [1, 1, 0, 0]
    price = [10, 10, 100, 100]
    res = _solve(p, pv, [1] * 4, price, price)
    assert res.objective == pytest.approx(2 * ec * ed * 100)
    assert res.e.sum() == pytest.approx(0, abs=1e-9)


def test_revenue_upper_bound():
    """Revenue gain can never exceed spread x usable energy x cycles x efficiency (Notion risk list)."""
    rng = np.random.default_rng(1)
    p = DispatchParams(power_mw=1, energy_usable_mwh=2, eta_charge=0.93, eta_discharge=0.93, max_cycles_per_day=1.0)
    T = 96
    price = 60 + 40 * np.sin(np.linspace(0, 2 * np.pi, T)) + rng.normal(0, 5, T)
    pv = np.clip(np.sin(np.linspace(-0.5, np.pi + 0.5, T)), 0, None) * 3
    res = _solve(p, pv, np.full(T, 3.0), price, price, dt=0.25)
    base = (np.maximum(price, 0) * pv * 0.25).sum()
    gain = res.objective - base
    bound = (price.max() - price.min()) * p.energy_usable_mwh * 1.0
    assert 0 <= gain <= bound + 1e-6
    assert (res.d * 0.25).sum() <= 1.0 * p.energy_usable_mwh + 1e-6


def test_shared_export_limit():
    """PV and battery share one export limit: PV export plus discharge never exceeds it."""
    p = DispatchParams(power_mw=2, energy_usable_mwh=4)
    pv = [2, 2, 1.5, 1.5]
    price = [0, 0, 100, 100]
    cap = [1, 1, 1, 1]
    res = _solve(p, pv, cap, price, price)
    assert np.all(res.e + res.d <= 1 + 1e-9)
    # with 1.5 MW PV at 100 EUR and a 1 MW cap, exporting PV directly is best; battery energy is useless
    assert res.objective == pytest.approx(200)


def test_curtailment_rescue():
    """Export cap of zero at noon (curtailment order): the battery saves what fits and sells it later."""
    ec = ed = 0.9
    p = DispatchParams(power_mw=1, energy_usable_mwh=1.8, eta_charge=ec, eta_discharge=ed)
    pv = [1, 1, 0, 0]
    cap = [0, 0, 1, 1]
    price = [50, 50, 50, 50]
    res = _solve(p, pv, cap, price, price)
    assert res.soc.max() == pytest.approx(1.8)
    assert res.objective == pytest.approx(1.8 * ed * 50)


def test_no_grid_charging_by_default():
    p = DispatchParams(power_mw=1, energy_usable_mwh=2)
    price = [0, 0, 100, 100]
    res = _solve(p, [0, 0, 0, 0], [1] * 4, price, price)
    assert res.objective == pytest.approx(0)
    assert res.cg.sum() == 0


def test_grid_charging_when_allowed():
    ec = ed = 0.9
    p = DispatchParams(power_mw=1, energy_usable_mwh=1.8, eta_charge=ec, eta_discharge=ed)
    price = [10, 10, 100, 100]
    res = _solve(p, [0, 0, 0, 0], [1] * 4, price, price, c_imp=price, imp=[1] * 4)
    assert res.objective == pytest.approx(-2 * 10 + 1.8 * ed * 100)


def test_no_simultaneous_charge_discharge_at_negative_import_price():
    """Negative import cost invites 'burning' energy in an LP. The model must not do it."""
    p = DispatchParams(power_mw=1, energy_usable_mwh=1, eta_charge=0.9, eta_discharge=0.9)
    price = [-50, -50, -50, -50, 5]
    res = _solve(p, [0] * 5, [1] * 5, price, price, c_imp=price, imp=[1] * 5, exact=True)
    charge = res.cp + res.cg
    assert np.all(np.minimum(charge, res.d) < 1e-6)
    # best physical plan is sequential: charge 1 MWh (+50), discharge 0.81 MWh (-40.5),
    # charge 1 MWh (+50), top up 0.111 MWh (+5.56), sell 0.9 MWh at 5 (+4.5) = 69.56.
    # (An LP without the binary would 'burn' energy in the same interval and report more.)
    assert res.used_milp
    assert res.objective == pytest.approx(50 - 40.5 + 50 + 50 / 9 + 4.5, rel=1e-6)
    # default mode: time-sharing inside the interval, never more than rated power in total,
    # and the energy cycled this way is reported, not hidden
    ts = _solve(p, [0] * 5, [1] * 5, price, price, c_imp=price, imp=[1] * 5)
    assert np.all(ts.cp + ts.cg + ts.d <= 1 + 1e-9)
    assert ts.cycled_within_interval_mwh > 0
    assert ts.objective >= res.objective - 1e-6


def test_dc_coupling_recovers_clipping():
    """DC/AC ratio 1.5: energy above the inverter limit is lost when AC-coupled, stored when DC-coupled."""
    eta_inv = 1.0
    pv_dc = np.array([1.5, 1.5, 0, 0])
    price = np.array([50, 50, 50, 50.0])
    p_inv = 1.0
    ac = DispatchParams(power_mw=1, energy_usable_mwh=4, eta_charge=1, eta_discharge=1, coupling="ac", eta_inverter=eta_inv)
    dc = DispatchParams(power_mw=1, energy_usable_mwh=4, eta_charge=1, eta_discharge=1, coupling="dc", eta_inverter=eta_inv)
    r_ac = _solve(ac, np.minimum(pv_dc, p_inv), [p_inv] * 4, price, price)
    r_dc = _solve(dc, pv_dc, [p_inv] * 4, price, price)
    assert r_ac.objective == pytest.approx(100)
    assert r_dc.objective == pytest.approx(150)


def test_duals_match_finite_difference():
    """LP duals are the marginal value of power and energy (used as sizing gradients)."""
    rng = np.random.default_rng(3)
    T = 96
    price = 60 + 50 * np.sin(np.linspace(0, 2 * np.pi, T) - 1.5) + rng.normal(0, 8, T)
    pv = np.clip(np.sin(np.linspace(-0.3, np.pi + 0.3, T)), 0, None) * 2

    def val(P, E):
        p = DispatchParams(power_mw=P, energy_usable_mwh=E)
        return _solve(p, pv, np.full(T, 2.0), price, price, dt=0.25)

    r = val(1.0, 2.0)
    h = 1e-4
    dP = (val(1.0 + h, 2.0).objective - val(1.0 - h, 2.0).objective) / (2 * h)
    dE = (val(1.0, 2.0 + h).objective - val(1.0, 2.0 - h).objective) / (2 * h)
    assert r.dual_power.sum() == pytest.approx(dP, rel=1e-3, abs=1e-3)
    assert r.dual_energy.sum() == pytest.approx(dE, rel=1e-3, abs=1e-3)
