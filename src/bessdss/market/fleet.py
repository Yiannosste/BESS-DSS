"""System-level storage dispatch inside the price model (competitive equilibrium as a planner LP).

Idea
    The price model maps net residual load x (GW) to a price f(x) that never decreases with x.
    Many small price-taking storage operators, each doing arbitrage, end up in the same state
    as a single planner who minimises total system cost  sum_t F(x_t),  F' = f,  where
    x_t = residual load + charging - discharging.  F is convex because f is monotone, so the
    equilibrium is one convex programme. With a piecewise-linear F it is an LP.

    This is what produces the storage feedback ("more storage, lower spread") endogenously:
    the fleet charges where f is low and discharges where f is high until the price difference
    no longer covers round-trip losses or a capacity limit binds.

Per horizon of T steps, for storage units j = 1..J:
    variables   c_jt, d_jt (GW), s_jt (GWh), curt_t (GW), y_tk (GW) for curve segments k = 1..K
    balance     sum_k y_tk - sum_j c_jt + sum_j d_jt - curt_t = RL_t - b_0
    storage     s_jt = s_j,t-1 + eta_c c_jt dt - d_jt dt / eta_d
    bounds      0 <= y_tk <= width_k (last segment open ended), 0 <= c, d <= P_j, 0 <= s <= E_j
    objective   min  dt * sum_t sum_k pi_k y_tk  +  eps * throughput
Curtailment curt_t is free energy spilled when residual load falls below the curve's lowest point.
"""

from __future__ import annotations

from dataclasses import dataclass

import highspy
import numpy as np
from scipy import sparse

_INF = highspy.kHighsInf


@dataclass(frozen=True)
class StorageUnit:
    """Aggregated storage class in the system (e.g. all 2-hour batteries)."""

    name: str
    power_gw: float
    energy_gwh: float
    eta_charge: float = 0.93
    eta_discharge: float = 0.93


class FleetDispatch:
    """Re-usable planner LP for a fixed horizon, curve discretisation and number of storage units."""

    def __init__(self, n_steps: int, dt_h: float, breakpoints_gw: np.ndarray, n_units: int, wear_cost: float = 0.5):
        self.T, self.dt, self.J = int(n_steps), float(dt_h), int(n_units)
        self.b = np.asarray(breakpoints_gw, float)
        self.K = len(self.b) - 1
        self.wear = wear_cost  # EUR/MWh on discharge: tiny, only to break ties and stop idle cycling
        self._build()

    def _build(self):
        T, J, K, dt = self.T, self.J, self.K, self.dt
        # column layout: [c (J*T) | d (J*T) | s (J*T) | curt (T) | y (T*K)]
        self.oc, self.od, self.os = 0, J * T, 2 * J * T
        self.ocurt = 3 * J * T
        self.oy = self.ocurt + T
        n = self.oy + T * K
        m = T + J * T
        rows, cols, vals = [], [], []
        t = np.arange(T)
        for j in range(J):
            rows += list(t) + list(t)
            cols += list(self.oc + j * T + t) + list(self.od + j * T + t)
            vals += [-1.0] * T + [1.0] * T
            r0 = T + j * T
            rows += list(r0 + t) + list(r0 + t[1:]) + list(r0 + t) + list(r0 + t)
            cols += (
                list(self.os + j * T + t)
                + list(self.os + j * T + t[:-1])
                + list(self.oc + j * T + t)
                + list(self.od + j * T + t)
            )
            vals += [1.0] * T + [-1.0] * (T - 1) + [0.0] * T + [0.0] * T  # efficiencies set at solve time
        rows += list(t)
        cols += list(self.ocurt + t)
        vals += [-1.0] * T
        tt = np.repeat(t, K)
        rows += list(tt)
        cols += list(self.oy + np.arange(T * K))
        vals += [1.0] * (T * K)
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(m, n))
        self.n, self.m = n, m
        # keep explicit structure to refresh efficiency coefficients per solve
        self._A_base = A.tocsc()
        widths = np.diff(self.b)
        up = np.full(n, _INF)
        ywid = np.tile(widths, T)
        ywid[K - 1 :: K] = _INF  # last segment open ended (scarcity)
        up[self.oy :] = ywid
        self._up = up
        self._h = None
        self._eta_key = None

    def _model(self, units: list[StorageUnit]):
        """(Re)build the HiGHS model when efficiencies change (they enter the matrix)."""
        key = tuple((u.eta_charge, u.eta_discharge) for u in units)
        if self._h is not None and key == self._eta_key:
            return self._h
        T, J, dt = self.T, self.J, self.dt
        A = self._A_base.tolil()
        t = np.arange(T)
        for j, u in enumerate(units):
            r0 = T + j * T
            A[r0 + t, self.oc + j * T + t] = -u.eta_charge * dt
            A[r0 + t, self.od + j * T + t] = dt / u.eta_discharge
        A = A.tocsc()
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.n, self.m
        lp.col_cost_ = np.zeros(self.n)
        lp.col_lower_ = np.zeros(self.n)
        lp.col_upper_ = self._up.copy()
        lp.row_lower_ = np.zeros(self.m)
        lp.row_upper_ = np.zeros(self.m)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.passModel(lp)
        self._h, self._eta_key = h, key
        self._cidx = np.arange(self.n, dtype=np.int32)
        self._ridx = np.arange(self.m, dtype=np.int32)
        return h

    def solve(
        self,
        residual_load_gw: np.ndarray,
        segment_prices: np.ndarray,
        units: list[StorageUnit],
        soc0: np.ndarray | None = None,
    ) -> dict:
        """Dispatch the fleet for one horizon.

        residual_load_gw: (T,) load minus must-take generation, before storage.
        segment_prices: (K,) or (T, K) non-decreasing price of each curve segment, EUR/MWh
            (time-varying when fuel prices change inside the horizon). Negative prices are
            floored at zero for the planner so that it never "pays" to spill.
        Returns net load after storage, charge, discharge, SoC and curtailment.
        """
        T, J, K, dt = self.T, self.J, self.K, self.dt
        assert len(units) == J
        h = self._model(units)
        pi = np.maximum(np.broadcast_to(np.asarray(segment_prices, float), (T, K)), 0.0)
        cost = np.zeros(self.n)
        cost[self.oy :] = (pi * dt * 1000.0).reshape(-1)  # GW x h x EUR/MWh x 1000 = EUR
        cost[self.od : self.os] = self.wear * dt * 1000.0
        up = self._up.copy()
        for j, u in enumerate(units):
            up[self.oc + j * T : self.oc + (j + 1) * T] = u.power_gw
            up[self.od + j * T : self.od + (j + 1) * T] = u.power_gw
            up[self.os + j * T : self.os + (j + 1) * T] = u.energy_gwh
        rhs = np.zeros(self.m)
        rhs[:T] = np.asarray(residual_load_gw, float) - self.b[0]
        if soc0 is not None:
            for j in range(J):
                rhs[T + j * T] = min(float(soc0[j]), units[j].energy_gwh)
        h.changeColsCost(self.n, self._cidx, cost)
        h.changeColsBounds(self.n, self._cidx, np.zeros(self.n), up)
        h.changeRowsBounds(self.m, self._ridx, rhs, rhs)
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError("Fleet LP not optimal")
        x = np.asarray(h.getSolution().col_value)
        c = x[self.oc : self.od].reshape(J, T)
        d = x[self.od : self.os].reshape(J, T)
        s = x[self.os : self.ocurt].reshape(J, T)
        curt = x[self.ocurt : self.oy]
        net = np.asarray(residual_load_gw, float) + c.sum(0) - d.sum(0) + curt
        return {"net_load_gw": net, "charge_gw": c, "discharge_gw": d, "soc_gwh": s, "curtail_gw": curt}
