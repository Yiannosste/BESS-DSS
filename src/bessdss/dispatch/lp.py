"""Linear-programming dispatch of a battery co-located with a PV park behind one connection point.

The model is a price-taker LP solved with HiGHS. One model instance is built once per
(horizon length, structure) and then re-used for every day by updating only costs, bounds and
right-hand sides, which makes a 15-minute year solve in well under a second.

Unified AC/DC-coupled formulation (all powers in MW, energies in MWh, dt in hours):

    variables per interval t
        e_t   PV energy sent directly to the grid, measured at the AC side
        cp_t  battery charging from PV, measured at the battery bus
        cg_t  battery charging from the grid, measured at the grid side
        d_t   battery discharging, measured at the battery bus
        s_t   usable state of charge at the end of the interval (0 .. E_usable)

    constraints
        (PV)     a_e * e_t + cp_t                <= pv_avail_t
        (EXPORT) e_t + k_d * d_t                 <= export_cap_t
        (POWER)  cp_t + k_g * cg_t + d_t         <= P_batt          (also blocks most
                                                                    simultaneous charge/discharge)
        (SOC)    s_t = (1-sigma) s_{t-1} + eta_c dt (cp_t + k_g cg_t) - dt d_t / eta_d
        (CYCLE)  sum_{t in day} dt d_t           <= throughput_cap  (warranty limit per day, optional)
        (TERM)   s_T >= s_terminal

    objective (maximise)
        sum_t dt * ( r_pv_t e_t + (r_dis_t k_d - c_deg) d_t - c_imp_t cg_t )

AC-coupled: a_e = 1, k_d = k_g = 1, pv_avail = min(pv_dc*eta_inv, P_inv) (clipped energy is lost).
DC-coupled: a_e = 1/eta_inv, k_d = k_g = eta_inv, pv_avail = pv_dc, export_cap = min(cap, P_inv)
            (energy above the inverter limit can be stored: clipping recapture).

Physical-sense safeguards
    * (POWER) bounds charging plus discharging by the rated power. A solution with both in the
      same interval is therefore physically realisable by time-sharing: the converter charges for
      a fraction c/P of the interval and discharges for d/P of it. A naive model without this row
      would let the battery do both at full power, which is impossible.
    * Such intra-interval cycling only pays when a rule makes it pay: a negative import cost
      (paid to burn energy) or a support clawback that energy routed through the battery escapes.
      It is reported as a diagnostic (`cycled_within_interval_mwh`) so that it is never hidden.
      With exact=True those horizons are re-solved as a MILP with one binary per interval
      (strictly one direction per interval); it is slower and used for verification.
    * Curtailment is the free slack of the PV constraint, so negative remuneration never forces export.

The LP duals are returned because they are the exact marginal values of power and energy
capacity (envelope theorem). The sizing module uses them as gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import highspy
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

_INF = highspy.kHighsInf


@dataclass(frozen=True)
class DispatchParams:
    """Static battery and coupling parameters seen by the dispatch model."""

    power_mw: float  # battery power rating at the battery bus
    energy_usable_mwh: float  # usable energy this year (nameplate x SoC window x SoH)
    eta_charge: float = 0.93
    eta_discharge: float = 0.93
    self_discharge_per_h: float = 0.0
    coupling: str = "ac"  # "ac" or "dc"
    eta_inverter: float = 0.98  # PV inverter efficiency (used for DC coupling factors)
    degradation_cost_eur_mwh: float = 0.0  # per MWh discharged at the battery bus
    max_cycles_per_day: float | None = None  # full equivalent cycles per day (warranty), None = no cap

    @property
    def k(self) -> float:
        """Conversion from battery-bus power to AC grid power (1 for AC coupling)."""
        return self.eta_inverter if self.coupling == "dc" else 1.0

    @property
    def a_e(self) -> float:
        return 1.0 / self.eta_inverter if self.coupling == "dc" else 1.0


@dataclass
class DispatchResult:
    """Dispatch of one horizon. Arrays have length T (the committed part of the horizon)."""

    e: np.ndarray  # PV direct export, AC MW
    cp: np.ndarray  # PV charging, battery bus MW
    cg: np.ndarray  # grid charging, grid side MW
    d: np.ndarray  # discharge, battery bus MW
    soc: np.ndarray  # MWh at end of interval
    objective: float  # objective over the full solved horizon
    dual_power: np.ndarray  # EUR per MW per interval (marginal value of battery power)
    dual_energy: np.ndarray  # EUR per MWh per interval (marginal value of usable energy)
    dual_export: np.ndarray  # EUR per MW per interval (marginal value of export capacity)
    dual_cycle: float  # EUR per MWh of extra daily throughput allowance
    used_milp: bool = False
    cycled_within_interval_mwh: float = 0.0  # energy charged and discharged inside the same interval


class DispatchLP:
    """Re-usable HiGHS model for a fixed horizon length T and time step dt."""

    def __init__(self, n_steps: int, dt_h: float, params: DispatchParams, cycle_block_steps: int | None = None):
        self.T = int(n_steps)
        self.dt = float(dt_h)
        self.p = params
        # the warranty cycle limit is applied per block (one day): with a two-day horizon the first
        # day must not borrow the allowance of the second
        self.block = int(cycle_block_steps) if cycle_block_steps else self.T
        self.B = max(1, self.T // self.block)
        self._build()

    # ------------------------------------------------------------------ model construction
    def _matrices(self):
        T, dt, p = self.T, self.dt, self.p
        n = 5 * T
        ie, icp, icg, idd, isoc = (np.arange(T) + k * T for k in range(5))
        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.extend(np.broadcast_to(r, np.shape(c)).tolist() if np.ndim(r) == 0 else list(r))
            cols.extend(list(c))
            vals.extend(np.broadcast_to(v, np.shape(c)).tolist() if np.ndim(v) == 0 else list(v))

        t = np.arange(T)
        # (PV) rows 0..T-1
        add(t, ie, p.a_e)
        add(t, icp, 1.0)
        # (EXPORT) rows T..2T-1
        add(T + t, ie, 1.0)
        add(T + t, idd, p.k)
        # (POWER) rows 2T..3T-1
        add(2 * T + t, icp, 1.0)
        add(2 * T + t, icg, p.k)
        add(2 * T + t, idd, 1.0)
        # (SOC) rows 3T..4T-1 :  s_t - (1-sig)^dt s_{t-1} - eta_c dt (cp + k cg) + dt d/eta_d = rhs
        keep = (1.0 - p.self_discharge_per_h) ** dt
        add(3 * T + t, isoc, 1.0)
        add(3 * T + t[1:], isoc[:-1], -keep)
        add(3 * T + t, icp, -p.eta_charge * dt)
        add(3 * T + t, icg, -p.eta_charge * dt * p.k)
        add(3 * T + t, idd, dt / p.eta_discharge)
        # (CYCLE) rows 4T .. 4T+B-1, one per block
        add(4 * T + np.minimum(t // self.block, self.B - 1), idd, dt)
        A = sparse.csc_matrix((vals, (rows, cols)), shape=(4 * T + self.B, n))
        return A, keep

    def _build(self):
        T, p = self.T, self.p
        A, self._keep = self._matrices()
        self._A = A
        n, m = A.shape[1], A.shape[0]
        self.n, self.m = n, m
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = n, m
        lp.col_cost_ = np.zeros(n)
        lo = np.zeros(n)
        up = np.full(n, _INF)
        up[4 * T :] = p.energy_usable_mwh
        up[2 * T : 3 * T] = 0.0  # grid charging closed by default
        lp.col_lower_, lp.col_upper_ = lo, up
        rlo = np.full(m, -_INF)
        rup = np.zeros(m)
        rup[2 * T : 3 * T] = p.power_mw
        rlo[3 * T : 4 * T] = 0.0
        rup[3 * T : 4 * T] = 0.0
        rup[4 * T :] = _INF
        lp.row_lower_, lp.row_upper_ = rlo, rup
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        lp.sense_ = highspy.ObjSense.kMaximize
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.passModel(lp)
        self._h = h
        self._col_idx = np.arange(n, dtype=np.int32)
        self._row_idx = np.arange(m, dtype=np.int32)
        self._rlo, self._rup = rlo.copy(), rup.copy()
        self._clo, self._cup = lo.copy(), up.copy()

    # ------------------------------------------------------------------ solve
    def solve(
        self,
        pv_avail: np.ndarray,
        export_cap: np.ndarray,
        r_pv: np.ndarray,
        r_dis: np.ndarray,
        c_imp: np.ndarray | None = None,
        import_cap: np.ndarray | None = None,
        soc0: float = 0.0,
        soc_terminal: float = 0.0,
        commit_steps: int | None = None,
        exact: bool = False,
    ) -> DispatchResult:
        """Solve one horizon.

        pv_avail: PV power available (AC-coupled: AC after inverter limit; DC-coupled: DC), MW.
        export_cap: allowed net injection at the connection point, MW (already includes curtailment
            orders; for DC coupling the caller also applies the inverter limit).
        r_pv / r_dis: remuneration in EUR/MWh of directly exported PV energy / discharged energy.
        c_imp, import_cap: cost (EUR/MWh) and limit (MW) of grid charging; None = not allowed.
        """
        T, dt, p = self.T, self.dt, self.p
        pv_avail = np.maximum(np.asarray(pv_avail, float), 0.0)
        export_cap = np.maximum(np.asarray(export_cap, float), 0.0)
        cost = np.zeros(self.n)
        cost[0:T] = np.asarray(r_pv, float) * dt
        cost[3 * T : 4 * T] = (np.asarray(r_dis, float) * p.k - p.degradation_cost_eur_mwh) * dt
        cup = self._cup.copy()
        if c_imp is not None and import_cap is not None:
            cost[2 * T : 3 * T] = -np.asarray(c_imp, float) * dt
            cup[2 * T : 3 * T] = np.maximum(np.asarray(import_cap, float), 0.0)
        cup[4 * T :] = p.energy_usable_mwh
        clo = self._clo.copy()
        clo[5 * T - 1] = min(soc_terminal, p.energy_usable_mwh)
        rup = self._rup.copy()
        rlo = self._rlo.copy()
        rup[0:T] = pv_avail
        rup[T : 2 * T] = export_cap
        # clamp: a solution is only feasible to solver tolerance, so yesterday's final SoC can be -1e-9
        rhs0 = self._keep * min(max(soc0, 0.0), p.energy_usable_mwh)
        rlo[3 * T] = rhs0
        rup[3 * T] = rhs0
        if p.max_cycles_per_day is not None:
            rup[4 * T :] = p.max_cycles_per_day * p.energy_usable_mwh * (self.block * dt / 24.0)

        h = self._h
        h.changeColsCost(self.n, self._col_idx, cost)
        h.changeColsBounds(self.n, self._col_idx, clo, cup)
        h.changeRowsBounds(self.m, self._row_idx, rlo, rup)
        h.run()
        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            # a warm start from the previous day occasionally leaves the simplex in a bad state:
            # drop the basis and solve again from scratch before giving up
            h.clearSolver()
            h.run()
            status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            h.setOptionValue("solver", "ipm")
            h.clearSolver()
            h.run()
            status = h.getModelStatus()
            h.setOptionValue("solver", "choose")
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"Dispatch LP not optimal: {h.modelStatusToString(status)}")
        sol = h.getSolution()
        x = np.asarray(sol.col_value)
        row_dual = np.asarray(sol.row_dual)
        col_dual = np.asarray(sol.col_dual)
        obj = h.getInfo().objective_function_value
        used_milp = False

        charge = x[T : 2 * T] + p.k * x[2 * T : 3 * T]
        both = np.minimum(charge, x[3 * T : 4 * T])
        if exact and np.any(both > 1e-6):
            x, obj = self._solve_milp(cost, clo, cup, rlo, rup)
            used_milp = True
            both = np.zeros(T)

        C = T if commit_steps is None else int(commit_steps)
        sl = slice(0, C)
        # Marginal values. For a maximisation problem HiGHS reports duals with the sign of
        # d(objective)/d(rhs) for <= rows, so they are non-negative for binding capacity rows.
        dual_power = np.abs(row_dual[2 * T : 3 * T])[sl]
        dual_export = np.abs(row_dual[T : 2 * T])[sl]
        soc = x[4 * T :]
        at_upper = soc >= p.energy_usable_mwh - 1e-7
        dual_energy = np.where(at_upper, np.abs(col_dual[4 * T :]), 0.0)[sl]
        dual_cycle = float(np.abs(row_dual[4 * T :]).sum()) if p.max_cycles_per_day is not None else 0.0
        return DispatchResult(
            e=x[0:T][sl].copy(),
            cp=x[T : 2 * T][sl].copy(),
            cg=x[2 * T : 3 * T][sl].copy(),
            d=x[3 * T : 4 * T][sl].copy(),
            soc=soc[sl].copy(),
            objective=float(obj),
            dual_power=dual_power,
            dual_energy=dual_energy,
            dual_export=dual_export,
            dual_cycle=float(dual_cycle),
            used_milp=used_milp,
            cycled_within_interval_mwh=float(both[sl].sum() * dt),
        )

    def _solve_milp(self, cost, clo, cup, rlo, rup):
        """Exact re-solve with one binary per interval (1 = charging allowed, 0 = discharging allowed)."""
        T, p = self.T, self.p
        n = self.n
        # extra binaries z_t ; cp + k cg <= P z ; d <= P (1 - z)
        A = sparse.hstack([self._A, sparse.csc_matrix((self.m, T))]).tocsc()
        rows, cols, vals = [], [], []
        t = np.arange(T)
        rows += list(t) * 3
        cols += list(T + t) + list(2 * T + t) + list(n + t)
        vals += [1.0] * T + [p.k] * T + [-p.power_mw] * T
        rows += list(T + t) * 2
        cols += list(3 * T + t) + list(n + t)
        vals += [1.0] * T + [p.power_mw] * T
        B = sparse.csc_matrix((vals, (rows, cols)), shape=(2 * T, n + T))
        rhs_up = np.concatenate([np.zeros(T), np.full(T, p.power_mw)])
        cons = [
            LinearConstraint(A, np.where(np.isinf(rlo), -np.inf, rlo), np.where(rup >= _INF, np.inf, rup)),
            LinearConstraint(B, -np.inf, rhs_up),
        ]
        c = np.concatenate([-cost, np.zeros(T)])
        integrality = np.concatenate([np.zeros(n), np.ones(T)])
        lb = np.concatenate([clo, np.zeros(T)])
        ub = np.concatenate([np.where(cup >= _INF, np.inf, cup), np.ones(T)])
        res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub))
        if not res.success:
            raise RuntimeError(f"Dispatch MILP failed: {res.message}")
        return res.x[:n], float(-res.fun)
