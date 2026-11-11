"""Input schema of the DSS: the park, the support scheme, the regulatory switches.

Everything a user can change lives here. Regulatory switches carry the ID of the rule in
registers/regulatory.yaml that justifies their default, so a reader can trace every setting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SupportScheme(BaseModel):
    """How the park's energy is paid today."""

    kind: Literal["fit", "fip", "merchant", "ppa"]
    price_eur_mwh: float | None = Field(
        None, description="FiT tariff, FiP reference price (Τιμή Αναφοράς) or PPA price. Not used for merchant."
    )
    end_year: int | None = Field(None, description="Last calendar year of the support contract or PPA. After it the park is merchant.")
    ppa_covers_battery_output: bool = Field(
        False, description="PPA only: True if energy discharged from the battery is also bought at the PPA price."
    )

    @model_validator(mode="after")
    def _check(self):
        if self.kind != "merchant" and self.price_eur_mwh is None:
            raise ValueError(f"support scheme '{self.kind}' needs price_eur_mwh")
        return self


class Park(BaseModel):
    """An existing PV park."""

    name: str
    lat: float
    lon: float
    pv_dc_kwp: float = Field(..., gt=0, description="Installed module power (DC)")
    pv_ac_kw: float = Field(..., gt=0, description="Inverter AC power")
    grid_export_limit_kw: float = Field(..., gt=0, description="Maximum injection agreed in the connection terms")
    tilt_deg: float = 25.0
    azimuth_deg: float = 0.0  # 0 = south, 90 = west (PVGIS convention)
    tracking: Literal["fixed", "single_axis"] = "fixed"
    system_loss_pct: float = 14.0
    commissioning_year: int = 2020
    commissioning_date: str | None = Field(None, description="ISO date; if missing, 1 July of commissioning_year is assumed")
    static_injection_cap_frac: float = Field(
        1.0, gt=0, le=1,
        description="Existing permanent injection cap as share of installed power: 1.0 for parks operating before 04.07.2022, "
        "0.73 (distribution) or 0.72 (transmission) for later connection offers (REG-A-010)",
    )
    pv_degradation_per_year: float = 0.005
    grid_operator: Literal["deddie", "admie"] = "deddie"
    voltage_level: Literal["lv", "mv", "hv"] = "mv"
    support: SupportScheme
    curtailment_multiplier: float = Field(
        1.0, ge=0, description="Local curtailment relative to the system average (1 = same as system). User estimate."
    )
    measured_production_csv: str | None = Field(
        None, description="Optional CSV with columns ts_utc, power_kw (15 min or hourly) of real production"
    )
    eta_inverter: float = 0.98

    @property
    def dc_ac_ratio(self) -> float:
        return self.pv_dc_kwp / self.pv_ac_kw

    @property
    def commissioned(self) -> str:
        return self.commissioning_date or f"{self.commissioning_year}-07-01"

    def installed_power_mw(self, basis: str) -> float:
        """'Installed power of the PV' under the chosen legal reading, in MW."""
        return {"pv_dc_kwp": self.pv_dc_kwp, "pv_ac_kw": self.pv_ac_kw, "export_limit": self.grid_export_limit_kw}[basis] / 1000.0


class RegulatoryConfig(BaseModel):
    """Legal switches. Every default is the most defensible reading recorded in the regulatory
    register (registers/regulatory.yaml); the rule ID is given next to each field. Fields marked
    UNCLEAR have legally plausible alternatives and the study reports the result under each."""

    # --- station category after the battery is added (art. 11Γ ν. 4685/2020, REG-A-001, REG-A-003)
    station_category: Literal["11A", "11B"] = Field(
        "11A",
        description="11A: battery charges only from the PV, support contract kept for direct PV energy. "
        "11B: battery may also charge from the grid, operating aid is lost (REG-A-004, REG-A-005).",
    )
    eligibility_commissioning_after: str = Field("2019-07-04", description="11A with support only for parks commissioned after this date (REG-A-001)")
    pre2019_supported_park_eligible_11a: bool = Field(False, description="UNCLEAR (REG-A-002): the law is silent on older supported parks")
    # --- static injection caps
    cap_11a_frac_of_installed: float = Field(0.60, description="UNCLEAR (REG-A-009): max injection 07:00 to 19:00 as share of installed PV power. Law text says 'restriction 40 %', DEDDIE applies 60 %. Alternative 0.40")
    cap_11a_window_local: tuple[float, float] = (7.0, 19.0)
    cap_11b_profile_local: list[tuple[float, float, float]] = Field(
        default_factory=lambda: [(9, 10, 0.70), (10, 11, 0.35), (11, 15, 0.0), (15, 16, 0.35), (16, 17, 0.70)],
        description="(from hour, to hour, max injection as share of max injection power), local time (REG-A-011, interim)",
    )
    installed_power_basis: Literal["pv_dc_kwp", "pv_ac_kw", "export_limit"] = Field(
        "pv_dc_kwp", description="UNCLEAR: what 'installed power of the PV' means for the cap and the minimum capacity"
    )
    # --- sizing rules
    min_storage_hours: float = Field(1.0, description="Usable energy >= this many hours of installed PV power (REG-A-007)")
    min_capacity_rule_active: bool = Field(True, description="Industry asked to abolish the rule; not accepted so far (REG-A-008)")
    max_battery_power_frac_of_export: float = Field(1.0, description="Battery inverter power up to the park's max generation power (REG-A-013, REG-A-014)")
    # --- remuneration
    stored_energy_support: Literal["market_price", "support_price"] = Field(
        "market_price", description="Energy leaving the battery is paid at market prices through its own meter (REG-A-006)"
    )
    negative_price_rule: Literal["negative_more_than_2h", "nonpositive_all", "none"] = Field(
        "negative_more_than_2h",
        description="Sliding premium: aid not paid when the hourly day-ahead price is negative for more than two consecutive hours (REG-B-014). "
        "'nonpositive_all' is the stricter reading that EU guidelines point to (REG-B-017).",
    )
    negative_episode_scope: Literal["whole_episode", "from_third_hour"] = Field("whole_episode", description="UNCLEAR (REG-B-014)")
    fit_exposed_to_negative_price_rule: bool = Field(False, description="Fixed tariffs are paid regardless of the market price (REG-B-016)")
    fit_subject_to_curtailment: bool = Field(True, description="Technical curtailment orders apply to every controllable park (REG-B-019)")
    curtailment_compensation_frac: float = Field(0.0, ge=0, le=1, description="No compensation mechanism in force (REG-B-023, REG-B-025)")
    import_levies_eur_mwh: float = Field(0.0, description="11B: no ETMEAR or YKO on energy that is cycled back (REG-A-027)")
    dc_coupling_keeps_support: bool = Field(False, description="UNCLEAR (REG-A-026): press reports that a common inverter loses the support")
    balancing_services_allowed: bool = Field(False, description="Not counted for a small distribution-connected park (REG-B-033)")
    guarantee_eur_per_kw_tiers: list[tuple[float, float]] = Field(
        default_factory=lambda: [(1000.0, 42.0), (10000.0, 21.0), (100000.0, 14.0), (1e12, 7.0)],
        description="Letter of guarantee: (upper kW bound of the tier, EUR/kW) on the whole station power (REG-A-020)",
    )

    @property
    def grid_charging_allowed(self) -> bool:
        return self.station_category == "11B"


class BatteryOptions(BaseModel):
    """Technology assumptions for the candidate battery. Size is NOT here: it is an output."""

    coupling: Literal["ac", "dc"] = "ac"
    soc_min: float = 0.05
    soc_max: float = 0.95
    round_trip_efficiency_ac: float = 0.86
    aux_kw_per_mwh: float = 1.5
    max_cycles_per_year: float | None = 365.0
    eol_soh: float = 0.6
    degradation_model: Literal["linear", "blast_lfp_gr_250ah"] = "blast_lfp_gr_250ah"
    pack_life_factor: float = 0.775  # cell-to-pack life derating advised by the model authors (20 to 30 %)
    cell_temp_c: float = 25.0
    durations_h: list[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])


class Study(BaseModel):
    """One complete study request."""

    park: Park
    regulatory: RegulatoryConfig = Field(default_factory=RegulatoryConfig)
    battery: BatteryOptions = Field(default_factory=BatteryOptions)
    base_year: int = 2026
    cod_year: int = 2027  # first full year of battery operation
    life_years: int = 15
    scenarios: list[str] = Field(default_factory=list, description="Market scenario names; empty = all in the library")
    finance_overrides: dict = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Study:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))
