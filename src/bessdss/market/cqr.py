"""Conditional-quantile price model with rank bootstrap ("CQR-rank").

    normalised price  pi_t = price_t / M_month          (M = monthly thermal price level)
    state             z_t  = potential net residual load in GW
                             (load - potential PV - wind + storage net charging)
    model             pi_t = Q(u_t | z_t)

Q(u | z) is the conditional quantile surface of the normalised price given z, estimated from
history without assuming a functional form, then forced to be non-decreasing in z (more
residual load never lowers the price) and in u. The rank u_t in (0, 1) says where the hour sits
inside its conditional distribution. Ranks are strongly persistent and have an hour-of-day
pattern (ramps, neighbours, hydro), so whole historical days of ranks are resampled together.

Why this form
    * A single mean curve regresses to the mean: it reproduces average prices but only about
      80 % of the real daily spread, and never a zero price. A battery lives on the spread.
    * Zero and negative prices appear here as the part of the surface where Q = 0, with the right
      frequency at each level of z, instead of being bolted on.
    * Structural change enters through z: more PV lowers z at noon, storage raises it at noon
      and lowers it in the evening, demand growth raises it everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) regression, pool-adjacent-violators."""
    y = y.astype(float).copy()
    w = w.astype(float).copy()
    n = len(y)
    vals, wts, cnt = [], [], []
    for i in range(n):
        vals.append(y[i])
        wts.append(w[i])
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            ww = wts[-2] + wts[-1]
            c = cnt[-2] + cnt[-1]
            vals.pop(); wts.pop(); cnt.pop()
            vals[-1], wts[-1], cnt[-1] = v, ww, c
    return np.repeat(vals, cnt)


@dataclass
class QuantileSurface:
    z_grid: np.ndarray  # (nz,)
    u_grid: np.ndarray  # (nu,)
    q: np.ndarray  # (nz, nu) normalised price, non-decreasing along both axes

    def price(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of Q(u | z). Inputs are clipped to the grid."""
        z = np.clip(np.asarray(z, float), self.z_grid[0], self.z_grid[-1])
        u = np.clip(np.asarray(u, float), self.u_grid[0], self.u_grid[-1])
        iz = np.clip(np.searchsorted(self.z_grid, z) - 1, 0, len(self.z_grid) - 2)
        iu = np.clip(np.searchsorted(self.u_grid, u) - 1, 0, len(self.u_grid) - 2)
        tz = (z - self.z_grid[iz]) / (self.z_grid[iz + 1] - self.z_grid[iz])
        tu = (u - self.u_grid[iu]) / (self.u_grid[iu + 1] - self.u_grid[iu])
        q = self.q
        return (
            q[iz, iu] * (1 - tz) * (1 - tu)
            + q[iz + 1, iu] * tz * (1 - tu)
            + q[iz, iu + 1] * (1 - tz) * tu
            + q[iz + 1, iu + 1] * tz * tu
        )

    def rank(self, z: np.ndarray, pi: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Rank u = F(pi | z). Where the surface is flat (zero-price mass) the rank is drawn
        uniformly inside the flat interval, which is the standard randomised PIT."""
        z = np.clip(np.asarray(z, float), self.z_grid[0], self.z_grid[-1])
        pi = np.asarray(pi, float)
        iz = np.clip(np.searchsorted(self.z_grid, z) - 1, 0, len(self.z_grid) - 2)
        tz = (z - self.z_grid[iz]) / (self.z_grid[iz + 1] - self.z_grid[iz])
        rows = self.q[iz] * (1 - tz)[:, None] + self.q[iz + 1] * tz[:, None]  # (n, nu)
        lo = (rows < pi[:, None] - 1e-9).sum(axis=1)
        hi = (rows <= pi[:, None] + 1e-9).sum(axis=1)
        ug = np.concatenate([[0.0], self.u_grid, [1.0]])
        u_lo, u_hi = ug[lo], ug[np.minimum(hi + 1, len(ug) - 1)]
        r = rng.random(len(pi)) if rng is not None else 0.5
        return np.clip(u_lo + (u_hi - u_lo) * r, 1e-4, 1 - 1e-4)

    def mean_curve(self) -> np.ndarray:
        """E[pi | z] on z_grid (used by the storage fleet planner)."""
        return np.trapezoid(self.q, self.u_grid, axis=1) / (self.u_grid[-1] - self.u_grid[0])


def fit_surface(
    z: np.ndarray,
    pi: np.ndarray,
    z_grid: np.ndarray | None = None,
    u_grid: np.ndarray | None = None,
    bandwidth_gw: float = 0.35,
    floor_zero_below: float = 0.02,
) -> QuantileSurface:
    """Kernel-weighted empirical quantiles on a z grid, then monotone rearrangement."""
    z = np.asarray(z, float)
    pi = np.asarray(pi, float)
    if z_grid is None:
        z_grid = np.arange(np.floor(z.min()) - 1.0, np.ceil(z.max()) + 0.01, 0.25)
    if u_grid is None:
        u_grid = np.linspace(0.01, 0.99, 99)
    order = np.argsort(pi)
    zs, ps = z[order], pi[order]
    q = np.zeros((len(z_grid), len(u_grid)))
    neff = np.zeros(len(z_grid))
    for i, zc in enumerate(z_grid):
        w = np.exp(-0.5 * ((zs - zc) / bandwidth_gw) ** 2)
        sw = w.sum()
        neff[i] = sw
        if sw < 1e-9:
            q[i] = np.nan
            continue
        cw = np.cumsum(w) / sw
        q[i] = np.interp(u_grid, cw, ps)
    # fill unsupported cells from the nearest supported neighbour, then enforce monotonicity
    good = neff > 3.0
    idx = np.where(good)[0]
    for i in range(len(z_grid)):
        if not good[i]:
            q[i] = q[idx[np.argmin(np.abs(idx - i))]]
    w = np.maximum(neff, 1e-3)
    for j in range(len(u_grid)):
        q[:, j] = _pava(q[:, j], w)
    q = np.sort(q, axis=1)
    q[np.abs(q) < floor_zero_below] = np.where(q[np.abs(q) < floor_zero_below] < 0, q[np.abs(q) < floor_zero_below], 0.0)
    return QuantileSurface(z_grid=np.asarray(z_grid, float), u_grid=np.asarray(u_grid, float), q=q)


def monthly_level(price: pd.Series, z: pd.Series, z_ref: tuple[float, float] = (3.0, 5.0)) -> pd.Series:
    """Monthly thermal price level M: the median price in hours where residual load is in the
    range typically set by gas-fired plants. Index: month period start (UTC)."""
    m = (z > z_ref[0]) & (z < z_ref[1])
    key = price.index.to_period("M")
    lvl = price[m].groupby(key[m]).median()
    allm = price.groupby(key).median()
    return lvl.reindex(allm.index).fillna(allm)
