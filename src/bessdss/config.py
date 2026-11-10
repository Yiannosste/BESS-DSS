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


class RegulatoryConfig(BaseModel):
    """Legal switches. Defaults follow the most defensible reading recorded in the regulatory register."""

    grid_charging_allowed: bool = Field(False, description="May the battery absorb energy from the grid")
    import_levies_eur_mwh: float = Field(0.0, description="Regulated charges on energy absorbed from the grid")
    min_storage_hours: float = Field(1.0, description="Minimum usable energy in hours of the reference power")
    min_storage_reference: Literal["pv_ac_power", "battery_power", "export_limit"] = "pv_ac_power"
    max_battery_power_frac_of_export: float | None = Field(1.0, description="Upper limit on battery power relative to export limit")
    stored_energy_support: Literal["support_price", "market_price"] = Field(
        "support_price", description="FiT/FiP parks: what the energy discharged from the battery receives"
    )
    negative_price_rule: Literal["no_support_when_nonpositive", "support_always"] = Field(
        "no_support_when_nonpositive", description="Operating aid in intervals with zero or negative day-ahead price"
    )
    fit_exposed_to_negative_price_rule: bool = Field(False, description="Does the negative price rule also hit old FiT contracts")
    fit_subject_to_curtailment: bool = Field(True, description="Are FiT parks curtailed like the rest")
    curtailment_compensation_frac: float = Field(0.0, ge=0, le=1, description="Share of curtailed energy that is compensated")
    balancing_services_allowed: bool = False
    rule_ids: dict[str, str] = Field(default_factory=dict, description="setting name -> regulatory register ID")


class BatteryOptions(BaseModel):
    """Technology assumptions for the candidate battery. Size is NOT here: it is an output."""

    coupling: Literal["ac", "dc"] = "ac"
    soc_min: float = 0.05
    soc_max: float = 0.95
    round_trip_efficiency_ac: float = 0.86
    aux_kw_per_mwh: float = 1.5
    max_cycles_per_year: float | None = 365.0
    eol_soh: float = 0.6
    degradation_model: Literal["linear", "semi_empirical_lfp"] = "semi_empirical_lfp"
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
