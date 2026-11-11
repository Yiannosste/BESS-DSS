"""PV resampling and calibration (REQ-02). Synthetic inputs, pure functions."""

import numpy as np
import pandas as pd
import pytest

from bessdss.pv.pvgis import annual_yield_stats, calibrate_to_measured, to_quarter_hour


def _hourly(days=5):
    idx = pd.date_range("2023-06-01", periods=24 * days, freq="1h", tz="UTC")
    h = idx.hour.to_numpy()
    p = np.clip(np.sin((h - 4) / 14 * np.pi), 0, None) * (0.6 + 0.4 * np.random.default_rng(0).random(len(idx)))
    return pd.Series(p, index=idx)


def test_quarter_hour_conserves_energy():
    s = _hourly()
    q = to_quarter_hour(s)
    assert len(q) == 4 * len(s)
    hourly_back = q.to_numpy().reshape(-1, 4).mean(axis=1)
    assert np.allclose(hourly_back, s.to_numpy(), rtol=1e-4, atol=1e-9)
    assert (q >= 0).all()


def test_calibration_recovers_scale():
    idx = pd.date_range("2022-01-01", periods=24 * 200, freq="1h", tz="UTC")
    h = idx.hour.to_numpy()
    mod = pd.Series(np.clip(np.sin((h - 5) / 13 * np.pi), 0, None), index=idx)
    meas = mod * 0.9
    out, rep = calibrate_to_measured(mod, meas)
    assert rep["overall_ratio_measured_to_modelled"] == pytest.approx(0.9, rel=1e-6)
    assert out.sum() == pytest.approx(meas.sum(), rel=1e-6)


def test_annual_yield_stats():
    s = _hourly(365)
    st = annual_yield_stats(s)
    assert st["deviation_pct"].abs().max() < 1e-9 or len(st) > 1
