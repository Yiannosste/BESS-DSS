"""Merge the two research parts into registers/regulatory.yaml and attach, for every rule the
system acts on, HOW it is enforced (code location, setting, test). Rules that are context only
are marked as such, so nobody assumes the tool checks them."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENFORCED = {
    "REG-A-001": "bessdss.dispatch.simulate.effective_support_kind (ρύθμιση eligibility_commissioning_after). Έλεγχος συμμόρφωσης CHK-01. Test: test_eligibility_cutoff_for_supported_parks.",
    "REG-A-002": "Ρύθμιση pre2019_supported_park_eligible_11a (προεπιλογή false). Η μελέτη δείχνει το αποτέλεσμα και με την αντίθετη ανάγνωση. CHK-01.",
    "REG-A-003": "Ρύθμιση station_category. Το εργαλείο αξιολογεί και τις δύο κατηγορίες (bessdss.sizing.search.candidate_grid).",
    "REG-A-004": "RegulatoryConfig.grid_charging_allowed = False για 11Α: το όριο απορρόφησης στο LP είναι μηδέν (bessdss.dispatch.lp). Test: test_no_grid_charging_by_default.",
    "REG-A-005": "effective_support_kind: στο 11Β το πάρκο γίνεται merchant. Test: test_11b_profile_and_loss_of_support.",
    "REG-A-006": "bessdss.dispatch.simulate.remuneration: r_dis = τιμή αγοράς (ρύθμιση stored_energy_support). Test: test_fit_is_flat_and_expires.",
    "REG-A-007": "bessdss.sizing.search.legal_minimum_energy_mwh: κανένα υποψήφιο μέγεθος κάτω από το όριο. CHK-02.",
    "REG-A-008": "Ρύθμιση min_capacity_rule_active για το σενάριο κατάργησης.",
    "REG-A-009": "bessdss.dispatch.simulate.static_cap_mw (ρύθμιση cap_11a_frac_of_installed 0,60, εναλλακτική 0,40). Test: test_11a_static_cap_and_no_grid_charging. CHK-05.",
    "REG-A-010": "Park.static_injection_cap_frac για το πάρκο πριν από τη μετατροπή, και αντικατάστασή του από το όριο κατηγορίας μετά.",
    "REG-A-011": "static_cap_mw με cap_11b_profile_local. Test: test_11b_profile_and_loss_of_support.",
    "REG-A-012": "Οι εντολές περικοπής εφαρμόζονται σε όλη την έγχυση του σταθμού, και στη μπαταρία (simulate_with_battery). Test: test_curtailment_blocks_discharge_too.",
    "REG-A-013": "Κοινό όριο έγχυσης στο LP (γραμμή EXPORT). Test: test_shared_export_limit. CHK-03.",
    "REG-A-014": "Ρύθμιση max_battery_power_frac_of_export = 1,0 (η αυστηρότερη ανάγνωση, του ΑΔΜΗΕ). CHK-03.",
    "REG-A-020": "bessdss.study.guarantee_cost: η προμήθεια της εγγυητικής μπαίνει στο αρχικό κόστος. CHK-09.",
    "REG-A-024": "Έλεγχος συμμόρφωσης CHK-07 (υποχρέωση τηλεελέγχου). Το κόστος είναι μέσα στο σταθερό CAPEX.",
    "REG-A-025": "Έλεγχος συμμόρφωσης CHK-08 (μετρητές). Το κόστος είναι μέσα στο σταθερό CAPEX.",
    "REG-A-026": "effective_support_kind: με κοινό μετατροπέα η ενίσχυση θεωρείται χαμένη (ρύθμιση dc_coupling_keeps_support). CHK-06.",
    "REG-A-027": "Ρύθμιση import_levies_eur_mwh = 0 για την ενέργεια φόρτισης του 11Β.",
    "REG-A-035": "Έλεγχος συμμόρφωσης CHK-11 (εκκρεμείς αλλαγές).",
    "REG-A-037": "Έλεγχος συμμόρφωσης CHK-11.",
    "REG-A-038": "Έλεγχος συμμόρφωσης CHK-11.",
    "REG-A-039": "Ρύθμιση finance_overrides.capex_subsidy_frac. Η μελέτη υπολογίζει την επιδότηση που μηδενίζει το NPV.",
    "REG-A-017": "Έλεγχος συμμόρφωσης CHK-10 (διαδικασία ΔΕΔΔΗΕ).",
    "REG-A-018": "Έλεγχος συμμόρφωσης CHK-10.",
    "REG-B-001": "SupportScheme.end_year: μετά τη λήξη το πάρκο είναι merchant. Test: test_fit_is_flat_and_expires.",
    "REG-B-005": "remuneration (fip): τιμή αγοράς συν (Τιμή Αναφοράς μείον μηνιαία τιμή αναφοράς ΦΒ του μοντέλου). Ρύθμιση fip_two_way. Test: test_fip_two_way_premium_can_be_negative.",
    "REG-B-007": "Ίδιο με REG-A-006.",
    "REG-B-009": "Ίδιο με REG-A-002.",
    "REG-B-014": "bessdss.dispatch.simulate.aid_suspended (ρύθμιση negative_price_rule = negative_more_than_2h). Test: test_fip_sliding_premium_and_negative_price_rule.",
    "REG-B-016": "Ρύθμιση fit_exposed_to_negative_price_rule = False.",
    "REG-B-017": "Ρύθμιση negative_price_rule = nonpositive_all ως αυστηρότερη εναλλακτική στη ρυθμιστική ευαισθησία.",
    "REG-B-019": "Οι περικοπές του συστήματος γίνονται όριο έγχυσης του πάρκου (curtail_frac x curtailment_multiplier).",
    "REG-B-025": "Ρύθμιση curtailment_compensation_frac = 0.",
    "REG-B-029": "Υπόθεση ASM-F07 (αμοιβή ΦοΣΕ) στο οικονομικό μοντέλο.",
    "REG-B-033": "Ρύθμιση balancing_services_allowed = False: τα έσοδα εξισορρόπησης δεν μετρώνται.",
    "REG-B-035": "SupportScheme.end_year.",
    "REG-B-039": "Προειδοποίηση στον έλεγχο συμμόρφωσης όταν δίνεται επιδότηση CAPEX (CHK-12).",
    "REG-B-041": "Σημεία αγκύρωσης των σεναρίων (registers/scenarios.yaml).",
    "REG-B-042": "Σημεία εκκίνησης των σεναρίων (registers/scenarios.yaml).",
    "REG-B-046": "Υπόθεση ASM-F10.",
    "REG-B-047": "Υπόθεση ASM-F11.",
    "REG-B-050": "Η εκκαθάριση που υλοποιεί το remuneration είναι η πρόταση αυτής της εγγραφής. Είναι δική μας κατασκευή, όχι κείμενο νόμου.",
}

def main():
    rows = []
    for part in ("regulatory_part_A.yaml", "regulatory_part_B.yaml"):
        rows += yaml.safe_load((ROOT / "registers" / part).read_text(encoding="utf-8"))
    for r in rows:
        r["enforced_by"] = ENFORCED.get(r["id"], "Πλαίσιο. Δεν επιβάλλεται από τον κώδικα.")
    missing = set(ENFORCED) - {r["id"] for r in rows}
    assert not missing, missing
    out = ROOT / "registers" / "regulatory.yaml"
    out.write_text(
        "# Ρυθμιστικό μητρώο (ενοποιημένο). Παράγεται από τα μέρη Α και Β με το scripts/merge_regulatory.py.\n"
        + yaml.safe_dump(rows, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )
    print("rules", len(rows), "enforced", sum(1 for r in rows if not r["enforced_by"].startswith("Πλαίσιο")))

if __name__ == "__main__":
    main()
