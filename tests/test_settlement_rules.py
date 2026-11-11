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
    assert np.allclose(r_pv, 200)
    assert np.allclose(r_dis, [10, -5, 80])  # battery output is paid at market price (REG-A-006)
    _, r_dis2 = remuneration(s, RegulatoryConfig(stored_energy_support="support_price"), y, 2027)
    assert np.allclose(r_dis2, 200)
    r_pv, _ = remuneration(s, RegulatoryConfig(), y, 2031)
    assert np.allclose(r_pv, [10, -5, 80])  # after the contract the park is merchant


def test_fip_sliding_premium_and_negative_price_rule():
    """Reference price 65, monthly PV reference market price 40: premium 25 on top of the market.
    Rule in force (REG-B-014): no aid when the hourly price is negative for MORE than two hours in a row.
    A zero price or a two-hour negative spell keeps the aid."""
    s = SupportScheme(kind="fip", price_eur_mwh=65)
    y = _day([30, 0, -10, -10, 100, -5, -5, -5, 100], [1] * 9, ref=40)
    r_pv, _ = remuneration(s, RegulatoryConfig(), y, 2027)
    assert np.allclose(r_pv, [55, 25, 15, 15, 125, -5, -5, -5, 125])
    reg3 = RegulatoryConfig(negative_episode_scope="from_third_hour")
    assert np.allclose(remuneration(s, reg3, y, 2027)[0], [55, 25, 15, 15, 125, 20, 20, -5, 125])
    strict = RegulatoryConfig(negative_price_rule="nonpositive_all")
    assert np.allclose(remuneration(s, strict, y, 2027)[0], [55, 0, -10, -10, 125, -5, -5, -5, 125])
    none = RegulatoryConfig(negative_price_rule="none")
    assert np.allclose(remuneration(s, none, y, 2027)[0], [55, 25, 15, 15, 125, 20, 20, 20, 125])


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
    # rule in force (REG-A-006): rescued energy leaves the battery at the MARKET price (150), not the tariff
    assert with_b2.revenue_eur - base2.revenue_eur == pytest.approx(1.0 * 0.95 * 0.95 * 150)


def test_fit_battery_alternative_reading_support_price():
    """Alternative reading: if stored energy kept the tariff, rescue would be worth the tariff."""
    park = _park("fit", 200)
    y = _day([10, 10, 150, 150], [1, 1, 0, 0], curt=[0.5, 0.5, 0, 0])
    dp = DispatchParams(1, 2, 0.95, 0.95)
    reg = RegulatoryConfig(stored_energy_support="support_price")
    base = simulate_pv_only(park, reg, y, 2027)
    res = simulate_with_battery(park, reg, y, 2027, dp, lookahead_days=0)
    assert res.revenue_eur - base.revenue_eur == pytest.approx(1.0 * 0.95 * 0.95 * 200)


def test_11a_static_cap_and_no_grid_charging():
    """11A: injection limited to 60 % of installed PV power between 07:00 and 19:00 local time."""
    from bessdss.dispatch.simulate import static_cap_mw

    park = _park("merchant")
    reg = RegulatoryConfig()
    hours = np.arange(24.0)
    cap = static_cap_mw(park, reg, hours, with_battery=True)
    assert np.allclose(cap[7:19], 0.6) and np.allclose(cap[:7], 1.0) and np.allclose(cap[19:], 1.0)
    assert np.allclose(static_cap_mw(park, reg, hours, with_battery=False), 1.0)
    assert np.allclose(static_cap_mw(park, RegulatoryConfig(cap_11a_frac_of_installed=0.4), hours, True)[12], 0.4)
    assert reg.grid_charging_allowed is False
    # the cap costs energy: with a full-power noon and a tiny battery, PV above 60 % is lost
    y = YearInputs(1.0, np.array([0, 1, 1, 0.0]), np.array([50, 50, 50, 50.0]), np.zeros(4), local_hour=np.array([11, 12, 13, 20.0]), steps_per_day=4)
    base = simulate_pv_only(park, reg, y, 2027)
    res = simulate_with_battery(park, reg, y, 2027, DispatchParams(0.1, 0.1, 1.0, 1.0), lookahead_days=0)
    assert base.revenue_eur == pytest.approx(100)
    assert res.revenue_eur == pytest.approx((0.6 + 0.6 + 0.1) * 50)


def test_11b_profile_and_loss_of_support():
    """11B: zero injection 11:00 to 15:00, grid charging allowed, operating aid lost."""
    from bessdss.dispatch.simulate import effective_support_kind, static_cap_mw

    park = _park("fip", 65)
    reg = RegulatoryConfig(station_category="11B")
    cap = static_cap_mw(park, reg, np.arange(24.0), with_battery=True)
    assert np.allclose(cap[11:15], 0.0) and cap[9] == pytest.approx(0.7) and cap[10] == pytest.approx(0.35) and cap[20] == 1.0
    assert reg.grid_charging_allowed is True
    assert effective_support_kind(park, reg, 2027, with_battery=True) == "merchant"
    assert effective_support_kind(park, reg, 2027, with_battery=False) == "fip"


def test_eligibility_cutoff_for_supported_parks():
    """A supported park commissioned before 04.07.2019 does not keep its aid as 11A (default reading)."""
    from bessdss.dispatch.simulate import effective_support_kind

    old = _park("fit", 250, commissioning_year=2012)
    new = _park("fip", 65, commissioning_year=2021)
    reg = RegulatoryConfig()
    assert effective_support_kind(old, reg, 2027, True) == "merchant"
    assert effective_support_kind(new, reg, 2027, True) == "fip"
    alt = RegulatoryConfig(pre2019_supported_park_eligible_11a=True)
    assert effective_support_kind(old, alt, 2027, True) == "fit"
    assert effective_support_kind(new, reg, 2027, True, coupling="dc") == "merchant"


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
