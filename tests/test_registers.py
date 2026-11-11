"""Integrity of the registers (REQ-03, REQ-15, REQ-17): every number and rule the code uses must be
traceable to a register entry, and every dataset must have a provenance entry."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "registers"
SRC = ROOT / "src" / "bessdss"


def _yaml(name):
    return yaml.safe_load((REG / name).read_text(encoding="utf-8"))


def test_regulatory_register_integrity():
    rows = _yaml("regulatory.yaml")
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) and len(ids) >= 80
    for r in rows:
        assert r["status"] in {"in_force", "pending", "proposed", "unclear", "contradictory", "repealed"}, r["id"]
        assert r["confidence"] in {"primary_verified", "secondary_only", "unverified"}, r["id"]
        assert r["verified_on"], r["id"]
        assert r["source"]["citation"], r["id"]
        assert r["requirement"], r["id"]
        assert r["enforced_by"], r["id"]


def test_rule_ids_used_in_code_exist():
    ids = {r["id"] for r in _yaml("regulatory.yaml")}
    used = set()
    for f in list(SRC.rglob("*.py")) + [ROOT / "scripts" / "merge_regulatory.py"]:
        used |= set(re.findall(r"REG-[AB]-\d{3}", f.read_text(encoding="utf-8")))
    assert used, "the code should reference regulatory rules"
    assert used <= ids, f"unknown rule ids referenced in code: {sorted(used - ids)}"


def test_assumptions_used_by_code_exist():
    names = {r["name"] for r in _yaml("assumptions.yaml")}
    used = set()
    for f in SRC.rglob("*.py"):
        used |= set(re.findall(r"A\.(?:central|low_high|entry)\(\"([a-z0-9_]+)\"\)", f.read_text(encoding="utf-8")))
    assert used and used <= names, f"assumptions used in code but missing from the register: {sorted(used - names)}"
    for r in _yaml("assumptions.yaml"):
        assert r["confidence"] in {"sourced", "derived", "expert_guess"}, r["id"]
        assert r["source"], r["id"]


def test_every_dataset_has_provenance():
    prov = _yaml("data_provenance.yaml")
    files = {Path(p["file"]).name for p in prov if p.get("file")}
    blob = " ".join(str(p.get("file", "")) + " " + str(p.get("description", "")) + " " + str(p.get("checks", "")) for p in prov)
    data_dir = ROOT / "data" / "processed"
    if not data_dir.exists():
        pytest.skip("no processed data in this checkout")
    for f in data_dir.iterdir():
        if f.is_file() and f.suffix in {".parquet", ".csv"}:
            assert f.name in files or f.name in blob, f"{f.name} has no provenance entry"
    for p in prov:
        assert p["kind"].split()[0] in {"real", "modelled", "synthetic", "mixed"}, p["id"]


def test_decision_log_and_difficulties_shape():
    for d in _yaml("decision_log.yaml"):
        assert {"id", "decision", "stage", "options", "chosen", "status", "date"} <= set(d), d.get("id")
        assert d["status"] in {"Πρόταση", "Αποφασίστηκε", "Αντικαταστάθηκε"}
        assert len(d["options"]) >= 2, d["id"]
    for d in _yaml("difficulties.yaml"):
        assert {"id", "what", "where", "severity", "affects", "action", "open"} <= set(d), d.get("id")
