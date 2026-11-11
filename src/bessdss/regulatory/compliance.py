"""Regulatory compliance check for one park and one battery design.

Every check names the rule(s) of registers/regulatory.yaml it rests on, with the date the rule
was verified and how sure we are of it. Status: pass | fail | warn | info.
'warn' is used when the law is unclear or a change is pending: the tool never turns an unclear
rule into a silent pass.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from bessdss.config import Study
from bessdss.dispatch.simulate import effective_support_kind

REGISTER = Path(__file__).resolve().parents[3] / "registers" / "regulatory.yaml"


@lru_cache(maxsize=2)
def load_register(path: str = str(REGISTER)) -> dict[str, dict]:
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {r["id"]: r for r in rows}


def _rule_refs(ids: list[str]) -> list[dict]:
    reg = load_register()
    out = []
    for i in ids:
        r = reg[i]
        out.append(dict(id=i, title=r["title"], citation=r["source"]["citation"], status=r["status"],
                        confidence=r["confidence"], verified_on=r["verified_on"], url=r["source"].get("url", "")))
    return out


def check(study: Study, category: str, power_mw: float, energy_mwh: float) -> list[dict]:
    park, reg, b = study.park, study.regulatory.model_copy(update={"station_category": category}), study.battery
    usable = energy_mwh * (b.soc_max - b.soc_min)
    installed = park.installed_power_mw(reg.installed_power_basis)
    export = park.grid_export_limit_kw / 1000.0
    out: list[dict] = []

    def add(cid, title, status, text, rules):
        out.append(dict(id=cid, title=title, status=status, text=text, rules=_rule_refs(rules)))

    supported = park.support.kind in ("fit", "fip") and (park.support.end_year is None or park.support.end_year >= study.cod_year)
    after = park.commissioned > reg.eligibility_commissioning_after
    if category == "11A":
        if not supported:
            add("CHK-01", "Επιλεξιμότητα για σταθμό 11Α", "pass",
                "Το πάρκο δεν έχει λειτουργική ενίσχυση σε ισχύ, άρα η προσθήκη μπαταρίας επιτρέπεται χωρίς όριο ημερομηνίας.", ["REG-A-001", "REG-B-036"])
        elif after:
            add("CHK-01", "Επιλεξιμότητα για σταθμό 11Α με διατήρηση της ενίσχυσης", "pass",
                f"Το πάρκο τέθηκε σε λειτουργία στις {park.commissioned}, μετά τις {reg.eligibility_commissioning_after}. Η σύμβαση ενίσχυσης μένει για την ενέργεια που εγχέει απευθείας το ΦΒ.", ["REG-A-001", "REG-B-008"])
        elif reg.pre2019_supported_park_eligible_11a:
            add("CHK-01", "Επιλεξιμότητα για σταθμό 11Α με διατήρηση της ενίσχυσης", "warn",
                f"Το πάρκο λειτουργεί από {park.commissioned}, πριν από τις {reg.eligibility_commissioning_after}. Η μελέτη έτρεξε με την υπόθεση ότι κρατά την ενίσχυση. Ο νόμος δεν το προβλέπει ρητά. Χρειάζεται γραπτή επιβεβαίωση από ΔΑΠΕΕΠ ή ΡΑΑΕΥ.", ["REG-A-002", "REG-B-009"])
        else:
            add("CHK-01", "Επιλεξιμότητα για σταθμό 11Α με διατήρηση της ενίσχυσης", "fail",
                f"Το πάρκο λειτουργεί από {park.commissioned}, πριν από τις {reg.eligibility_commissioning_after}. Το άρθρο 11Γ δεν καλύπτει τέτοια πάρκα. Η μελέτη υπολογίζει τη μπαταρία ΧΩΡΙΣ την ενίσχυση (το πάρκο πουλά σε τιμές αγοράς).", ["REG-A-001", "REG-A-002", "REG-B-009"])
    else:
        add("CHK-01", "Σταθμός 11Β: απώλεια λειτουργικής ενίσχυσης", "warn" if supported else "pass",
            "Ο σταθμός 11Β δεν μπορεί να έχει σύμβαση λειτουργικής ενίσχυσης. " + ("Το πάρκο χάνει τη σύμβασή του από την ημέρα ηλέκτρισης ως 11Β." if supported else "Το πάρκο δεν έχει ενίσχυση, άρα δεν χάνει κάτι."),
            ["REG-A-005", "REG-B-011"])

    need = reg.min_storage_hours * installed if reg.min_capacity_rule_active else 0.0
    add("CHK-02", "Ελάχιστη ωφέλιμη χωρητικότητα", "pass" if usable + 1e-9 >= need else "fail",
        f"Απαιτούνται τουλάχιστον {need:.2f} MWh ωφέλιμα ({reg.min_storage_hours:g} ώρα επί {installed:.2f} MW εγκατεστημένης ισχύος ΦΒ, βάση: {reg.installed_power_basis}). Το σχέδιο έχει {usable:.2f} MWh ωφέλιμα στην αρχή ζωής. Ο νόμος δεν λέει αν το όριο ισχύει και μετά τη γήρανση.",
        ["REG-A-007", "REG-A-008"])

    pmax = reg.max_battery_power_frac_of_export * export
    add("CHK-03", "Ισχύς μπαταρίας και μέγιστη ισχύς έγχυσης", "pass" if power_mw <= pmax + 1e-9 else "fail",
        f"Ισχύς μπαταρίας {power_mw:.2f} MW, όριο {pmax:.2f} MW. Η μέγιστη ισχύς έγχυσης του σταθμού μένει {export:.2f} MW: ΦΒ και μπαταρία τη μοιράζονται.",
        ["REG-A-013", "REG-A-014"])

    if category == "11A":
        add("CHK-04", "Φόρτιση από το δίκτυο", "pass", "Το μοντέλο λειτουργίας δεν επιτρέπει απορρόφηση από το δίκτυο. Ο ΔΕΔΔΗΕ ζητά αισθητήρες κατεύθυνσης.", ["REG-A-004"])
        cap = min(export, reg.cap_11a_frac_of_installed * installed)
        add("CHK-05", "Μόνιμο όριο έγχυσης 07:00 έως 19:00", "warn",
            f"Η μελέτη εφαρμόζει μέγιστη έγχυση {cap:.2f} MW ({reg.cap_11a_frac_of_installed:.0%} της εγκατεστημένης ισχύος) τις ώρες 07:00 έως 19:00. Ο νόμος γράφει «περιορισμός 40%» και ο ΔΕΔΔΗΕ εφαρμόζει έγχυση έως 60%. Να επιβεβαιωθεί στην προσφορά σύνδεσης. Η μελέτη δείχνει το αποτέλεσμα και με τις δύο αναγνώσεις.",
            ["REG-A-009", "REG-A-010", "REG-B-013"])
    else:
        add("CHK-04", "Φόρτιση από το δίκτυο", "pass", "Επιτρέπεται. Στην ενέργεια που ξαναδίνεται δεν μπαίνει ΕΤΜΕΑΡ ούτε ΥΚΩ.", ["REG-A-005", "REG-A-027"])
        add("CHK-05", "Ωριαία όρια έγχυσης σταθμού 11Β", "warn",
            "Η μελέτη εφαρμόζει μηδενική έγχυση 11:00 έως 15:00 και μειωμένη 09:00 έως 11:00 και 15:00 έως 17:00. Τα όρια είναι προσωρινά, μέχρι να βγει ειδική απόφαση για το δίκτυο διανομής.",
            ["REG-A-011", "REG-A-035"])

    if b.coupling == "dc":
        kind = effective_support_kind(park, reg, study.cod_year, True, "dc")
        add("CHK-06", "Κοινός μετατροπέας (DC coupling) και ενίσχυση", "warn",
            "Με κοινό μετατροπέα δεν υπάρχει χωριστή μέτρηση ΦΒ και μπαταρίας όπως τη ζητά ο ΔΕΔΔΗΕ για σταθμούς με ενίσχυση. " + ("Η μελέτη θεωρεί ότι η ενίσχυση χάνεται." if kind == "merchant" and supported else "Η μελέτη έτρεξε με διατήρηση της ενίσχυσης, κάτι που δεν επιβεβαιώνεται."),
            ["REG-A-026", "REG-A-015"])
    else:
        add("CHK-06", "Τοπολογία σύνδεσης", "info", "Χωριστός μετατροπέας μπαταρίας (AC coupling). Δεσμεύει νέο ηλεκτρικό χώρο, άρα το αίτημα μπαίνει στη σειρά προτεραιότητας του Διαχειριστή.", ["REG-A-015", "REG-A-018"])

    add("CHK-07", "Τηλεποπτεία και τηλεέλεγχος", "info",
        "Ο σταθμός με αποθήκευση πρέπει να ανταλλάσσει σήματα και μετρήσεις σε πραγματικό χρόνο με το Κέντρο Ελέγχου, με δαπάνη του ιδιοκτήτη."
        + (" Το πάρκο είναι άνω των 400 kW, άρα η υποχρέωση ισχύει ήδη." if park.pv_ac_kw > 400 else ""), ["REG-A-024", "REG-B-021"])
    add("CHK-08", "Μετρητές", "info",
        "Για 11Α με ενίσχυση στη Μέση Τάση: δύο πρόσθετοι πιστοποιημένοι μετρητές (έξοδος ΦΒ και έξοδος μπαταρίας). Για 11Β: ένας μετρητής ιδιοκατανάλωσης." , ["REG-A-025", "REG-A-006"])
    kw = park.grid_export_limit_kw
    rate = next(eur for upper, eur in reg.guarantee_eur_per_kw_tiers if kw <= upper)
    add("CHK-09", "Εγγυητική επιστολή", "info", f"Περίπου {kw * rate:,.0f} ευρώ ({rate:g} ευρώ/kW σε όλη την ισχύ του σταθμού). Στο κόστος της μελέτης μπαίνει μόνο η προμήθεια της τράπεζας.".replace(",", "."), ["REG-A-020"])
    if park.grid_operator == "deddie":
        add("CHK-10", "Διαδικασία", "info", "Αίτημα στην πλατφόρμα ΑΠΕ του ΔΕΔΔΗΕ το πρώτο δεκαήμερο του μήνα. Αιτήματα που δεσμεύουν νέο ηλεκτρικό χώρο εξετάζονται μετά τα 900 MW αυτόνομων μπαταριών. Δεν υπάρχει προθεσμία απάντησης: ο χρόνος είναι κίνδυνος του έργου.", ["REG-A-016", "REG-A-017", "REG-A-018"])
    else:
        add("CHK-10", "Διαδικασία", "info", "Αίτημα στον ΑΔΜΗΕ. Χρειάζεται νέα βεβαίωση ηλέκτρισης και νέα δοκιμαστική λειτουργία.", ["REG-A-016", "REG-A-019"])
    add("CHK-11", "Αλλαγές που εκκρεμούν", "warn",
        "Εκκρεμούν: απόφαση για τα όρια έγχυσης στο δίκτυο διανομής, νομοσχέδιο για ευέλικτες συμβάσεις σύνδεσης και για χρέωση όλου του κόστους σύνδεσης στον παραγωγό, πλαίσιο επιμερισμού περικοπών, πρόγραμμα επιδότησης μπαταριών. Το μητρώο πρέπει να ξαναελεγχθεί πριν από κάθε πραγματική απόφαση.",
        ["REG-A-035", "REG-A-037", "REG-A-038", "REG-A-039", "REG-B-022", "REG-B-023"])
    if study.finance_overrides.get("capex_subsidy_frac", 0) > 0 and supported and category == "11A":
        add("CHK-12", "Επιδότηση κεφαλαίου και λειτουργική ενίσχυση", "warn", "Η επενδυτική ενίσχυση μειώνει κατά κανόνα τη λειτουργική ενίσχυση (συντελεστής απομείωσης). Δεν είναι σαφές αν επιδότηση μόνο για τη μπαταρία μειώνει την ταρίφα του πάρκου. Η μελέτη ΔΕΝ το αφαιρεί.", ["REG-B-039"])
    if park.support.kind == "fip" and category == "11A":
        add("CHK-13", "Κανόνας αρνητικών τιμών", "info", "Από 01.06.2026 η προσαύξηση δεν πληρώνεται όταν η ωριαία τιμή είναι αρνητική για πάνω από δύο συνεχόμενες ώρες. Οι μηδενικές τιμές δεν μετρούν. Η μελέτη δείχνει και την αυστηρότερη ανάγνωση (κάθε μη θετική τιμή).", ["REG-B-014", "REG-B-017"])
    return out


def summary_status(checks: list[dict]) -> str:
    st = {c["status"] for c in checks}
    if "fail" in st:
        return "fail"
    return "warn" if "warn" in st else "pass"
