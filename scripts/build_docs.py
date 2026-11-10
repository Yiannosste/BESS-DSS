"""Render the YAML registers into Greek markdown documents under docs/.

The YAML files are the single source of truth (Decision Log D-002). Run after every change:
    python scripts/build_docs.py
The CI checks that the rendered files are up to date.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "registers"
DOCS = ROOT / "docs"
HEADER = "<!-- Παράγεται αυτόματα από το {src} με το scripts/build_docs.py. Μην το αλλάζεις με το χέρι. -->\n\n"


def _load(name: str):
    p = REG / name
    if not p.exists():
        return []
    return yaml.safe_load(p.read_text(encoding="utf-8")) or []


def _cell(x) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return "<br>".join(f"• {_cell(i)}" for i in x)
    if isinstance(x, dict):
        return "<br>".join(f"{k}: {_cell(v)}" for k, v in x.items() if v not in (None, "", []))
    return str(x).replace("|", "\\|").replace("\n", " ").strip()


def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_cell(r.get(k)) for k, _ in cols) + " |")
    return "\n".join(out) + "\n"


def decision_log() -> str:
    rows = _load("decision_log.yaml")
    s = HEADER.format(src="registers/decision_log.yaml")
    s += "# Decision Log\n\n"
    s += (
        "Ίδιες στήλες με το Decision Log του Notion. Κάθε απόφαση που αλλάζει τι φτιάχνουμε, πώς το "
        "φτιάχνουμε ή ποια δεδομένα χρησιμοποιούμε γράφεται εδώ τη στιγμή που παίρνεται, μαζί με τις "
        "επιλογές που απορρίφθηκαν.\n\n"
    )
    s += f"Σύνολο αποφάσεων: {len(rows)}\n\n"
    s += _table(
        rows,
        [
            ("id", "ID"),
            ("decision", "Απόφαση"),
            ("stage", "Stage"),
            ("options", "Επιλογές που εξετάστηκαν"),
            ("chosen", "Τι επιλέχθηκε και γιατί"),
            ("status", "Κατάσταση"),
            ("date", "Ημερομηνία"),
        ],
    )
    return s


def difficulties() -> str:
    rows = _load("difficulties.yaml")
    s = HEADER.format(src="registers/difficulties.yaml")
    s += "# Μητρώο δυσκολιών\n\n"
    s += (
        "Κάθε δυσκολία που συναντήσαμε, από παντού: νόμος, δεδομένα, μοντελοποίηση, υπολογισμός, "
        "επικύρωση, εργαλεία, προϊόν. Τίποτα δεν κρύβεται και τίποτα δεν ωραιοποιείται.\n\n"
    )
    open_n = sum(1 for r in rows if str(r.get("open", "")).startswith("ναι"))
    part_n = sum(1 for r in rows if str(r.get("open", "")).startswith("μερικ"))
    s += f"Σύνολο: {len(rows)}. Ανοιχτές: {open_n}. Μερικώς ανοιχτές: {part_n}.\n\n"
    s += _table(
        rows,
        [
            ("id", "ID"),
            ("domain", "Τομέας"),
            ("what", "Τι είναι"),
            ("where", "Πού εμφανίστηκε"),
            ("severity", "Σοβαρότητα"),
            ("affects", "Τι επηρεάζει"),
            ("action", "Τι κάναμε"),
            ("open", "Ανοιχτό;"),
            ("lesson", "Μάθημα για τις ομάδες"),
        ],
    )
    return s


def regulatory() -> str:
    rows = _load("regulatory.yaml")
    s = HEADER.format(src="registers/regulatory.yaml")
    s += "# Ρυθμιστικό μητρώο\n\n"
    s += (
        "Κάθε κανόνας με την ακριβή επίσημη πηγή του, την ημερομηνία, τι απαιτεί και πώς τον επιβάλλει "
        "το σύστημα. Η στήλη «Βεβαιότητα» λέει αν διαβάσαμε την πρωτογενή πηγή (primary_verified), αν "
        "βασιζόμαστε σε δευτερογενή (secondary_only) ή αν δεν επιβεβαιώθηκε (unverified). Κανένας "
        "κανόνας δεν είναι επινοημένος. Όπου κάτι είναι ασαφές ή εκκρεμεί, γίνεται ρύθμιση του εργαλείου.\n\n"
    )
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r.get("status", "?")] = by_status.get(r.get("status", "?"), 0) + 1
    s += "Σύνολο κανόνων: " + str(len(rows)) + ". Ανά κατάσταση: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items())) + ".\n\n"
    topics: dict[str, list[dict]] = {}
    for r in rows:
        topics.setdefault(r.get("topic", "other"), []).append(r)
    for topic, items in topics.items():
        s += f"## {topic}\n\n"
        for r in items:
            src = r.get("source", {}) or {}
            s += f"### {r.get('id')}: {r.get('title', '')}\n\n"
            s += f"- **Τι απαιτεί:** {_cell(r.get('requirement'))}\n"
            s += f"- **Πηγή:** {_cell(src.get('citation'))} ({_cell(src.get('type'))})"
            if src.get("url"):
                s += f" [σύνδεσμος]({src['url']})"
            s += "\n"
            if src.get("quote"):
                s += f"- **Απόσπασμα:** «{_cell(src.get('quote'))}»\n"
            s += f"- **Ημερομηνία έκδοσης:** {_cell(r.get('date_issued'))}. **Κατάσταση:** {_cell(r.get('status'))}. "
            s += f"**Βεβαιότητα:** {_cell(r.get('confidence'))}. **Επαληθεύτηκε:** {_cell(r.get('verified_on'))}\n"
            s += f"- **Ποιους αφορά:** {_cell(r.get('applies_to'))}\n"
            s += f"- **Επίδραση στο μοντέλο:** {_cell(r.get('model_impact'))}\n"
            sp = r.get("suggested_parameter") or {}
            if sp.get("name"):
                s += f"- **Ρύθμιση:** `{sp.get('name')}` = `{sp.get('default')}`"
                if sp.get("alternatives"):
                    s += " (εναλλακτικές: " + ", ".join(f"`{a}`" for a in sp["alternatives"]) + ")"
                s += "\n"
            if r.get("enforced_by"):
                s += f"- **Πώς το επιβάλλει το σύστημα:** {_cell(r.get('enforced_by'))}\n"
            if r.get("notes"):
                s += f"- **Σημειώσεις:** {_cell(r.get('notes'))}\n"
            s += "\n"
    return s


def assumptions() -> str:
    rows = _load("assumptions.yaml")
    s = HEADER.format(src="registers/assumptions.yaml")
    s += "# Μητρώο υποθέσεων\n\n"
    s += (
        "Κάθε υπόθεση με την τιμή ή το εύρος της, την πηγή της και το πόσο ευαίσθητο είναι το αποτέλεσμα "
        "σε αυτήν. Ο κώδικας διαβάζει τις προεπιλεγμένες τιμές από το ίδιο αρχείο.\n\n"
    )
    doms: dict[str, list[dict]] = {}
    for r in rows:
        doms.setdefault(r.get("domain", "other"), []).append(r)
    for dom, items in doms.items():
        s += f"## {dom}\n\n"
        s += _table(
            items,
            [
                ("id", "ID"),
                ("name", "Παράμετρος"),
                ("description", "Περιγραφή"),
                ("unit", "Μονάδα"),
                ("low", "Χαμηλή"),
                ("central", "Κεντρική"),
                ("high", "Υψηλή"),
                ("as_of", "Ημερομηνία"),
                ("source", "Πηγή"),
                ("confidence", "Είδος"),
                ("sensitivity", "Ευαισθησία αποτελέσματος"),
            ],
        )
        s += "\n"
    return s


def data_provenance() -> str:
    rows = _load("data_provenance.yaml")
    s = HEADER.format(src="registers/data_provenance.yaml")
    s += "# Μητρώο προέλευσης δεδομένων\n\n"
    s += (
        "Κάθε σύνολο δεδομένων: πηγή, άδεια, ανάλυση, περίοδος, γνωστά κενά, και αν είναι πραγματικό "
        "(real), μοντελοποιημένο (modelled) ή συνθετικό (synthetic). Συνθετικά δεδομένα δεν παρουσιάζονται "
        "ποτέ ως πραγματικά.\n\n"
    )
    for r in rows:
        s += f"### {r.get('id')}: {r.get('dataset')}\n\n"
        for k, label in [
            ("description", "Περιγραφή"),
            ("file", "Αρχείο"),
            ("source", "Πηγή"),
            ("url", "URL"),
            ("licence", "Άδεια"),
            ("kind", "Είδος"),
            ("resolution", "Ανάλυση"),
            ("period", "Περίοδος"),
            ("rows", "Γραμμές"),
            ("known_gaps", "Γνωστά κενά"),
            ("retrieved_on", "Λήψη"),
            ("checks", "Έλεγχοι"),
        ]:
            if r.get(k) not in (None, ""):
                s += f"- **{label}:** {_cell(r.get(k))}\n"
        s += "\n"
    return s


def innovation() -> str:
    rows = _load("innovation_log.yaml")
    s = HEADER.format(src="registers/innovation_log.yaml")
    s += "# Ημερολόγιο τεχνικής έρευνας και καινοτομίας\n\n"
    s += (
        "Κάθε ιδέα συγκρίνεται με ένα απλό, κατανοητό baseline. Κρατάμε ό,τι το ξεπερνά με στοιχεία. "
        "Ό,τι δοκιμάστηκε και απορρίφθηκε μένει εδώ με τον λόγο.\n\n"
    )
    s += _table(rows, [("id", "ID"), ("idea", "Ιδέα"), ("verdict", "Ετυμηγορία"), ("evidence_short", "Στοιχείο")])
    s += "\n"
    for r in rows:
        s += f"## {r.get('id')}: {r.get('idea')}\n\n"
        for k, label in [
            ("what", "Τι είναι"),
            ("where", "Πού εφαρμόζεται"),
            ("baseline", "Baseline"),
            ("experiment", "Πείραμα"),
            ("result", "Αποτέλεσμα"),
            ("verdict", "Ετυμηγορία"),
            ("why", "Γιατί"),
            ("sources", "Πηγές"),
            ("code", "Κώδικας"),
        ]:
            if r.get(k) not in (None, "", []):
                s += f"- **{label}:** {_cell(r.get(k))}\n"
        s += "\n"
    return s


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    outputs = {
        "DECISION_LOG.md": decision_log(),
        "DIFFICULTIES.md": difficulties(),
        "REGULATORY_REGISTER.md": regulatory(),
        "ASSUMPTIONS.md": assumptions(),
        "DATA_PROVENANCE.md": data_provenance(),
        "INNOVATION_LOG.md": innovation(),
    }
    for name, text in outputs.items():
        (DOCS / name).write_text(text, encoding="utf-8")
        print(f"wrote docs/{name} ({len(text)} chars)")


if __name__ == "__main__":
    main()
