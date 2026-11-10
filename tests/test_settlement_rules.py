"""Tests of the support-scheme settlement and of the yearly simulator (REQ-04, REQ-05, REQ-07).

Inputs are small synthetic series built so that the expected money can be computed by hand.
"""

import numpy as np
import pytest

from bessdss.config import Park, RegulatoryConfig, SupportScheme
from bessdss.dispatch.lp import DispatchParams
from bessdss.dispatch.simulate import YearInputs, remuneration, simulate_pv_only, simulate_with_battery


def _park(kind="merchant", price=None, end_year=None, **kw):
    base = dict(
        name="t", lat=39.0, lon=22.0, pv_dc_kwp=1000, pv_ac_kw=1000, grid_export_limit_kw=1000,
        eta_inverter=1.0, support=SupportScheme(kind=kind, price_eur_mwh=price, end_year=end_year),
    )
    base.update(kw)
    return Park(**base)


def _day(price, pv, curt=None, ref=None):
    price = np.array(price, float)
    return YearInputs(
        dt_h=1.0, pv_dc_mw=np.array(pv, float), price=price,
        curtail_frac=np.zeros(len(price)) if curt is None else np.array(curt, float),
        pv_ref_price_monthly=None if ref is None else np.full(len(price), float(ref)),
        steps_per_day=len(price),
    )


def test_merchant_gets_market_price():
    y = _day([10, -5, 80], [1, 1, 0])
    r_pv, r_dis = remuneration(SupportScheme(kind="merchant"), RegulatoryConfig(), y, 2027)
    assert np.allclose(r_pv, [10, -5, 80]) and np.allclose(r_dis, r_pv)


def test_fit_is_flat_and_expires():
    y = _day([10, -5, 80], [1, 1, 0])
    s = SupportScheme(kind="fit", price_eur_mwh=200, end_year=2030)
    r_pv, r_dis = remuneration(s, RegulatoryConfig(), y, 2027)
    assert np.allclose(r_pv, 200) and np.allclose(r_dis, 200)
    r_pv, _ = remuneration(s, RegulatoryConfig(), y, 2031)
    assert np.allclose(r_pv, [10, -5, 80])  # after the contract the park is merchant


def test_fip_sliding_premium_and_negative_price_rule():
    """Reference price 65, monthly PV reference market price 40: premium 25 on top of the market,
    except in intervals with price <= 0 where no aid is paid."""
    y = _day([30, 0, -10, 100], [1, 1, 1, 0], ref=40)
    s = SupportScheme(kind="fip", price_eur_mwh=65)
    r_pv, _ = remuneration(s, RegulatoryConfig(), y, 2027)
    assert np.allclose(r_pv, [55, 0, -10, 125])
    r_pv2, _ = remuneration(s, RegulatoryConfig(negative_price_rule="support_always"), y, 2027)
    assert np.allclose(r_pv2, [55, 25, 15, 125])


def test_fip_two_way_premium_can_be_negative():
    y = _day([120], [1], ref=90)
    r_pv, _ = remuneration(SupportScheme(kind="fip", price_eur_mwh=65), RegulatoryConfig(), y, 2027)
    assert r_pv[0] == pytest.approx(95)


def test_pv_only_curtails_at_negative_price_and_on_order():
    park = _park("merchant")
    y = _day([50, -10, 50, 50], [1, 1, 1, 0], curt=[0, 0, 0.4, 0])
    res = simulate_pv_only(park, RegulatoryConfig(), y, 2027)
    # hour 0: 1 MWh at 50; hour 1: nothing (negative); hour 2: 0.6 MWh at 50
    assert res.revenue_eur == pytest.approx(50 + 0.6 * 50)
    assert res.curtailed_mwh == pytest.approx(1 + 0.4)


def test_fit_battery_only_earns_from_rescued_energy():
    """FiT park, no curtailment: a battery cannot add revenue (it only loses energy)."""
    park = _park("fit", 200)
    reg = RegulatoryConfig()
    y = _day([10, 10, 150, 150], [1, 1, 0, 0])
    dp = DispatchParams(1, 2, 0.95, 0.95)
    base = simulate_pv_only(park, reg, y, 2027)
    with_b = simulate_with_battery(park, reg, y, 2027, dp, lookahead_days=0)
    assert with_b.revenue_eur == pytest.approx(base.revenue_eur)
    # with a curtailment order of 50 % at noon, rescued energy is paid at the tariff after losses
    y2 = _day([10, 10, 150, 150], [1, 1, 0, 0], curt=[0.5, 0.5, 0, 0])
    base2 = simulate_pv_only(park, reg, y2, 2027)
    with_b2 = simulate_with_battery(park, reg, y2, 2027, dp, lookahead_days=0)
    assert base2.revenue_eur == pytest.approx(200)
    assert with_b2.revenue_eur - base2.revenue_eur == pytest.approx(1.0 * 0.95 * 0.95 * 200)


def test_fit_battery_with_market_price_rule_changes_value():
    """If stored energy is paid the market price instead of the tariff, rescue is worth less."""
    park = _park("fit", 200)
    y = _day([10, 10, 150, 150], [1, 1, 0, 0], curt=[0.5, 0.5, 0, 0])
    dp = DispatchParams(1, 2, 0.95, 0.95)
    reg = RegulatoryConfig(stored_energy_support="market_price")
    base = simulate_pv_only(park, reg, y, 2027)
    res = simulate_with_battery(park, reg, y, 2027, dp, lookahead_days=0)
    assert res.revenue_eur - base.revenue_eur == pytest.approx(1.0 * 0.95 * 0.95 * 150)


def test_curtailment_blocks_discharge_too():
    """During an order the cap applies to the whole injection, not only to the PV."""
    park = _park("merchant")
    y = _day([100, 100, 100, 1], [1, 1, 1, 0], curt=[1, 1, 1, 0])
    dp = DispatchParams(1, 5, 1.0, 1.0)
    res = simulate_with_battery(park, RegulatoryConfig(), y, 2027, dp, lookahead_days=0, keep_traces=True)
    assert np.all(res.net_export_mw[:3] <= 1e-9)
    assert res.revenue_eur == pytest.approx(1.0)  # 1 MW for the last hour at 1 EUR (power limit)


def test_policies_are_ordered():
    """Perfect foresight >= persistence >= 0 gain; the rule baseline never beats perfect foresight."""
    rng = np.random.default_rng(5)
    days, spd = 20, 24
    tod = np.tile(np.arange(spd), days)
    price = 70 - 50 * np.exp(-((tod - 12) / 3.0) ** 2) + 60 * np.exp(-((tod - 19) / 1.5) ** 2) + rng.normal(0, 12, days * spd)
    pv = np.clip(np.sin((tod - 6) / 12 * np.pi), 0, None)
    y = YearInputs(1.0, pv, price, np.zeros(days * spd))
    park = _park("merchant")
    reg = RegulatoryConfig()
    dp = DispatchParams(0.5, 1.0, 0.93, 0.93)
    base = simulate_pv_only(park, reg, y, 2027).revenue_eur
    g = {pol: simulate_with_battery(park, reg, y, 2027, dp, policy=pol).revenue_eur - base for pol in ["perfect", "persistence", "rules"]}
    assert g["perfect"] >= g["persistence"] - 1e-6
    assert g["perfect"] >= g["rules"] - 1e-6
    assert g["perfect"] > 0
