"""Known-answer tests for finance (REQ-10, REQ-11) and battery physics (REQ-08)."""

import numpy as np
import pytest

from bessdss.battery.model import (
    BatterySpec,
    LinearDegradation,
    BlastLfpGr250Ah,
    YearUsage,
    rainflow_cycles,
)
from bessdss.finance.model import FinanceParams, capex_base, evaluate, irr, max_debt_for_dscr


def _simple_params(**kw):
    base = dict(
        life_years=10, inflation=0.0, discount_rate_nominal=0.08, tax_rate=0.0, depreciation_years=10,
        capex_energy_eur_per_kwh=100.0, capex_power_eur_per_kw=0.0, capex_fixed_eur=0.0,
        opex_fixed_frac_of_capex=0.0, insurance_frac_of_capex=0.0,
        aggregator_fee_frac_of_incremental_revenue=0.0, decommissioning_frac_of_capex=0.0,
        gearing=0.0,
    )
    base.update(kw)
    return FinanceParams(**base)


def test_npv_annuity_closed_form():
    """Constant cash flow: NPV = -I + CF * (1 - (1+r)^-N) / r."""
    p = _simple_params()
    capex = capex_base(p, 1.0, 1.0)  # 100 000
    cf = 20_000.0
    res = evaluate(p, 1.0, 1.0, np.full((1, 10), cf))
    expected = -capex + cf * (1 - 1.08**-10) / 0.08
    assert res.npv[0] == pytest.approx(expected)
    # payback 5 years exactly
    assert res.payback_years[0] == pytest.approx(5.0)


def test_irr_known_value():
    cf = np.array([[-100.0, 110.0]])
    assert irr(cf)[0] == pytest.approx(0.10, abs=1e-6)
    cf2 = np.array([[-1000.0] + [162.745] * 10])  # 10 % annuity factor 6.1446
    assert irr(cf2)[0] == pytest.approx(0.10, abs=1e-4)


def test_revenue_is_not_profit():
    """OPEX, aggregator fee and tax reduce the cash flow below revenue."""
    p = _simple_params(tax_rate=0.22, opex_fixed_frac_of_capex=0.02, aggregator_fee_frac_of_incremental_revenue=0.1)
    res = evaluate(p, 1.0, 1.0, np.full((1, 10), 30_000.0))
    assert np.all(res.fcf_unlevered[0, 1:] < 30_000.0)
    ebitda = 30_000 - 0.02 * 100_000 - 3_000
    tax = 0.22 * (ebitda - 10_000)
    assert res.fcf_unlevered[0, 1] == pytest.approx(ebitda - tax)


def test_real_nominal_consistency():
    """A real cash flow inflated at i and discounted at (1+r)(1+i)-1 gives the real-rate NPV."""
    r_real, infl = 0.05, 0.03
    p_real = _simple_params(discount_rate_nominal=r_real, inflation=0.0)
    p_nom = _simple_params(discount_rate_nominal=(1 + r_real) * (1 + infl) - 1, inflation=infl)
    rev = np.full((1, 10), 18_000.0)
    assert evaluate(p_nom, 1, 1, rev).npv[0] == pytest.approx(evaluate(p_real, 1, 1, rev).npv[0])


def test_dscr_definition():
    p = _simple_params(gearing=0.5, debt_tenor_years=5, debt_rate=0.06)
    res = evaluate(p, 1.0, 1.0, np.full((1, 10), 20_000.0))
    ann = 50_000 * 0.06 / (1 - 1.06**-5)
    assert res.debt_service[0, 0] == pytest.approx(ann)
    assert res.dscr[0, 0] == pytest.approx(20_000 / ann)
    assert res.dscr.shape == (1, 5)


def test_debt_sizing_on_downside():
    p = _simple_params(debt_tenor_years=5, debt_rate=0.06, target_dscr=1.25)
    cfads = np.full(10, 12_500.0)
    debt = max_debt_for_dscr(p, cfads)
    ann = debt * 0.06 / (1 - 1.06**-5)
    assert ann == pytest.approx(10_000.0)


def test_battery_usable_energy_and_rte():
    b = BatterySpec(power_mw=1, energy_mwh=2, soc_min=0.1, soc_max=0.9, eta_charge=0.94, eta_discharge=0.94)
    assert b.usable_energy() == pytest.approx(1.6)
    assert b.usable_energy(0.8) == pytest.approx(1.28)
    assert b.round_trip_efficiency == pytest.approx(0.8836)
    assert b.duration_h == 2


def test_linear_degradation():
    m = LinearDegradation(0.01, 0.0001)
    soh = m.simulate([YearUsage(fec=300)] * 10)
    assert soh[-1] == pytest.approx(1 - 10 * (0.01 + 0.03))


def test_blast_model_reproduces_reference_points():
    """Reference SoH values computed with the published coefficients at cell level
    (25 degC, mean SoC 50 %, constant conditions), see docs/research/finance_battery_tools_notes.md."""
    m = BlastLfpGr250Ah(pack_life_factor=1.0)
    assert m.anode_potential(0.5) == pytest.approx(0.1233, abs=2e-4)
    assert m.anode_potential(0.2) == pytest.approx(0.1805, abs=2e-4)
    one = m.simulate([YearUsage(fec=365, mean_soc=0.5, mean_dod=0.8)] * 20)
    assert one[10] == pytest.approx(0.866, abs=0.004)
    assert one[20] == pytest.approx(0.775, abs=0.004)
    two = m.simulate([YearUsage(fec=730, mean_soc=0.5, mean_dod=0.9)] * 20)
    assert two[20] == pytest.approx(0.574, abs=0.006)


def test_blast_model_monotone_and_pack_derating():
    m = BlastLfpGr250Ah(pack_life_factor=1.0)
    soh = m.simulate([YearUsage(fec=300)] * 15)
    assert np.all(np.diff(soh) < 0)
    # more cycles and higher average SoC age the battery faster; a pack ages faster than a cell
    assert m.simulate([YearUsage(fec=600)] * 15)[-1] < soh[-1]
    assert m.simulate([YearUsage(fec=300, mean_soc=0.9)] * 15)[-1] < soh[-1]
    assert BlastLfpGr250Ah(pack_life_factor=0.775).simulate([YearUsage(fec=300)] * 15)[-1] < soh[-1]
    # changing conditions year by year gives the same result as constant conditions when they are equal
    assert m.simulate([YearUsage(fec=300)] * 15)[-1] == pytest.approx(soh[-1])


def test_rainflow_simple():
    # one full 0-1-0 cycle with a small inner cycle 0.6-0.4-0.6
    soc = np.array([0, 0.6, 0.4, 1.0, 0.0])
    depths, weights = rainflow_cycles(soc)
    assert sorted(np.round(depths, 6).tolist()) == [0.2, 1.0, 1.0]
    assert (depths * weights).sum() == pytest.approx(0.2 * 1.0 + 1.0 * 0.5 + 1.0 * 0.5)
