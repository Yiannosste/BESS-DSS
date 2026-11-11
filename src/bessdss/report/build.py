"""Greek study report (self-contained HTML) built from results.json.

Charts follow one small system: fixed categorical colour order, thin marks, one axis per chart,
recessive grid, direct labels where few series, and a table next to every chart that matters.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948")
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"], C["red"]]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
STATUS = {"pass": ("✔", "#1b7f3b", "Εντάξει"), "fail": ("✖", "#b3261e", "Δεν πληρείται"), "warn": ("!", "#9a6700", "Θέλει προσοχή"), "info": ("i", "#52514e", "Ενημέρωση")}
VERDICT = {
    "yes": ("Ναι, αξίζει", "Η επένδυση βγαίνει θετική και στο κακό σενάριο (P90)."),
    "only_with_risk": ("Αξίζει μόνο με ρίσκο", "Ο μέσος όρος είναι θετικός, αλλά στο κακό σενάριο (P90) η επένδυση χάνει."),
    "no": ("Όχι, δεν αξίζει σήμερα", "Κανένα μέγεθος μπαταρίας δεν βγάζει θετική αναμενόμενη αξία με τους κανόνες και τα σενάρια της μελέτης."),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def eur(x, digits=0) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "δ/υ"
    s = f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s + " €"


def num(x, digits=1, suffix="") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "δ/υ"
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".") + suffix


def pct(x, digits=1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "δ/υ"
    return num(100 * x, digits, "%")


def _keur(v, _pos=None):
    return f"{v / 1000:,.0f}".replace(",", ".")


# ----------------------------------------------------------------------------- figures
def fig_npv(res) -> str:
    npv = np.array(res["distributions"]["npv"])
    b = res["best"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.hist(npv / 1000, bins=50, color=C["blue"], alpha=0.85, edgecolor=SURFACE, linewidth=0.6)
    for val, lab, col in ((b["npv_p90"], "P90", C["red"]), (b["npv_p50"], "P50", INK), (b["npv_mean"], "μέσος", C["orange"])):
        ax.axvline(val / 1000, color=col, lw=1.6)
        ax.text(val / 1000, ax.get_ylim()[1] * 0.97, f" {lab}", color=INK, va="top", ha="left", fontsize=8.5)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Καθαρή παρούσα αξία (χιλ. €)")
    ax.set_ylabel("Αριθμός διαδρομών")
    ax.set_title("Κατανομή της καθαρής παρούσας αξίας")
    return _png(fig)


def fig_revenue(res) -> str:
    d = res["distributions"]
    yrs = res["design_detail"]["years"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.fill_between(yrs, np.array(d["revenue_p10"]) / 1000, np.array(d["revenue_p90"]) / 1000, color=C["blue"], alpha=0.18, lw=0, label="εύρος P90 έως P10")
    ax.plot(yrs, np.array(d["revenue_p50"]) / 1000, color=C["blue"], lw=2, label="P50")
    ax.plot(yrs, np.array(d["revenue_p10"]) / 1000, color=C["red"], lw=1.4, label="κακός χρόνος (P90)")
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    ax.set_ylabel("χιλ. € ανά έτος (σταθερές τιμές)")
    ax.set_title("Πρόσθετο έσοδο από τη μπαταρία ανά έτος")
    ax.legend(loc="upper right", fontsize=8.5)
    return _png(fig)


def fig_tornado(res) -> str:
    rows = res["risk"]["tornado"][::-1]
    base = rows[0]["base"]
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(rows) + 1.0))
    for i, r in enumerate(rows):
        ax.barh(i, (r["low"] - base) / 1000, left=base / 1000, color=C["orange"], height=0.55)
        ax.barh(i, (r["high"] - base) / 1000, left=base / 1000, color=C["blue"], height=0.55)
    ax.set_yticks(range(len(rows)), [r["factor"] for r in rows], fontsize=8.5)
    ax.axvline(base / 1000, color=INK, lw=1)
    ax.axvline(0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Αναμενόμενη NPV (χιλ. €). Πορτοκαλί: δυσμενές άκρο, μπλε: ευνοϊκό άκρο")
    ax.set_title("Ποιος παράγοντας μετακινεί περισσότερο το αποτέλεσμα")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def fig_sensitivity(res) -> str:
    s = res["risk"]["sensitivity_npv"]
    lab = res["risk"]["labels"]
    items = sorted(s.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    ax.barh([lab[k] for k, _ in items], [100 * v for _, v in items], color=C["blue"], height=0.55)
    for i, (_, v) in enumerate(items):
        ax.text(100 * v + 1, i, f"{100 * v:.0f}%", va="center", color=INK, fontsize=8.5)
    ax.set_xlabel("Μερίδιο της διασποράς της NPV που εξηγεί ο παράγοντας μόνος του (%)")
    ax.set_title("Από πού έρχεται η αβεβαιότητα")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(100 * max(s.values()) * 1.15, 10))
    return _png(fig)


def fig_dscr(res) -> str:
    d = res["distributions"]
    n = len(d["dscr_p50"])
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(x, d["dscr_p50"], color=C["blue"], lw=2, label="P50")
    ax.plot(x, d["dscr_p10"], color=C["red"], lw=1.6, label="κακός χρόνος (P90)")
    ax.axhline(d["target_dscr"], color=INK, lw=1, ls="--", label=f"απαίτηση τράπεζας {d['target_dscr']:.2f}")
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Έτος δανείου")
    ax.set_ylabel("DSCR")
    ax.set_title("Δείκτης κάλυψης δανείου (μόνο οι ροές της μπαταρίας)")
    ax.legend(fontsize=8.5)
    return _png(fig)


def fig_decomposition(res) -> str:
    dd = res["design_detail"]
    names = dd["scen_names"]
    i = names.index("central") if "central" in names else 0
    yrs = dd["rep_years"]
    g, p = np.array(dd["gross"][i]) / 1000, np.array(dd["penalty"][i]) / 1000
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    w = 0.36
    x = np.arange(len(yrs))
    ax.bar(x - w / 2 - 0.02, g, w, color=C["blue"], label="αξία της μπαταρίας")
    ax.bar(x + w / 2 + 0.02, p, w, color=C["orange"], label="κόστος μετατροπής του σταθμού")
    ax.plot(x, g + p, color=INK, lw=1.6, marker="o", ms=4, label="καθαρό πρόσθετο έσοδο")
    ax.set_xticks(x, [str(y) for y in yrs])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("χιλ. € ανά έτος")
    ax.set_title(f"Από πού έρχεται η αξία ({res['meta']['scenarios'][names[i]]['label']})")
    ax.legend(fontsize=8.5)
    ax.grid(axis="x", visible=False)
    return _png(fig)


def fig_days(res) -> str:
    days = res["operation"]["days"]
    titles = {"spring_surplus": "Ανοιξιάτικη μέρα με πλεόνασμα", "summer": "Καλοκαιρινή μέρα", "winter": "Χειμωνιάτικη μέρα"}
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.5, 1.0]})
    for j, key in enumerate(("spring_surplus", "summer", "winter")):
        d = days[key]
        h = np.array(d["local_hour"])
        o = np.argsort(h)
        h = h[o]
        f = lambda k, d=d, o=o: np.array(d[k])[o]  # noqa: E731
        a = axes[0, j]
        a.step(h, f("price"), color=INK, lw=1.3, where="post")
        a.set_title(titles[key], fontsize=9.5)
        if j == 0:
            a.set_ylabel("Τιμή (€/MWh)")
        a = axes[1, j]
        a.fill_between(h, 0, f("pv"), color=C["yellow"], alpha=0.35, lw=0, step="post", label="παραγωγή ΦΒ")
        a.step(h, f("export"), color=C["blue"], lw=1.8, where="post", label="έγχυση σταθμού")
        a.step(h, f("charge"), color=C["aqua"], lw=1.4, where="post", label="φόρτιση")
        a.step(h, f("discharge"), color=C["orange"], lw=1.4, where="post", label="εκφόρτιση")
        if j == 0:
            a.set_ylabel("Ισχύς (MW)")
            a.legend(fontsize=7.5, loc="upper left")
        a = axes[2, j]
        a.fill_between(h, 0, f("soc"), color=C["violet"], alpha=0.3, lw=0, step="post")
        a.step(h, f("soc"), color=C["violet"], lw=1.4, where="post")
        a.set_xlabel("Ώρα Ελλάδας")
        a.set_xticks([0, 6, 12, 18, 24])
        if j == 0:
            a.set_ylabel("Ενέργεια στη μπαταρία (MWh)")
    fig.tight_layout()
    return _png(fig)


def fig_market(res) -> str:
    rows = res["market"]
    labels = {n: v["label"] for n, v in res["meta"]["scenarios"].items()}
    labels["frozen_2026"] = "Παγωμένη στο 2026"
    names = [n for n in labels if any(r["scenario"] == n for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, key, title in ((axes[0], "tb4", "Ημερήσια διαφορά τιμών (4 ακριβές μείον 4 φθηνές ώρες, €/MWh)"), (axes[1], "zero_hours", "Ώρες με μηδενική ή αρνητική τιμή"), (axes[2], "bess_gw", "Μπαταρίες στη χώρα (GW)")):
        for k, n in enumerate(names):
            pts = [(r["year"], r[key]) for r in rows if r["scenario"] == n]
            xs, ys = zip(*sorted(pts), strict=True)
            style = dict(color=INK2, ls="--", lw=1.2) if n == "frozen_2026" else dict(color=SERIES[k % 8], lw=1.8)
            ax.plot(xs, ys, marker="o", ms=3, label=labels[n], **style)
        ax.set_title(title, fontsize=8.8)
    axes[0].legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    return _png(fig)


def fig_soh(res) -> str:
    dd = res["design_detail"]
    labels = {n: v["label"] for n, v in res["meta"]["scenarios"].items()}
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    x = [dd["years"][0] - 1] + dd["years"]
    for k, n in enumerate(dd["scen_names"]):
        ax.plot(x, 100 * np.array(dd["soh"][k]), color=SERIES[k % 8], lw=1.6, label=labels[n])
    eol = res["study"]["battery"]["eol_soh"] * 100
    ax.axhline(eol, color=INK, lw=1, ls="--")
    ax.text(x[0], eol + 0.6, "όριο εγγύησης", fontsize=8, color=INK)
    ax.set_ylabel("Υγεία μπαταρίας (% της αρχικής χωρητικότητας)")
    ax.set_title("Φθορά της μπαταρίας")
    ax.legend(fontsize=7.5)
    return _png(fig)


# ----------------------------------------------------------------------------- html
CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#0b0b0b;background:#fcfcfb;max-width:980px;margin:0 auto;padding:24px 16px 60px;line-height:1.5;font-size:15px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:34px 0 8px;border-bottom:1px solid #e6e5e0;padding-bottom:4px}h3{font-size:16px;margin:20px 0 6px}
.sub{color:#52514e;margin:0 0 18px}.verdict{border:1px solid #e6e5e0;border-left:6px solid VCOL;border-radius:8px;padding:14px 18px;margin:14px 0;background:#fff}
.verdict b{font-size:20px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}
.tile{border:1px solid #e6e5e0;border-radius:8px;padding:10px 12px;background:#fff}.tile .k{color:#52514e;font-size:12.5px}.tile .v{font-size:19px;font-weight:650}
table{border-collapse:collapse;width:100%;margin:8px 0 14px;font-size:13.5px}th,td{border-bottom:1px solid #e6e5e0;padding:5px 8px;text-align:left;vertical-align:top}
th{color:#52514e;font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}img{max-width:100%;height:auto;display:block;margin:10px 0}img.sm{max-width:640px;width:100%}img.md{max-width:760px;width:100%}
.note{color:#52514e;font-size:13.5px}.st{display:inline-block;min-width:1.4em;text-align:center;font-weight:700}.small{font-size:12.5px;color:#52514e}
.warnbox{border:1px solid #e6e5e0;border-left:6px solid #9a6700;border-radius:8px;padding:10px 14px;background:#fff;margin:10px 0}
@media print{body{font-size:12px}h2{break-after:avoid}img{break-inside:avoid}}
"""


def _table(headers, rows, num_cols=()):
    h = "".join(f"<th class='{'n' if i in num_cols else ''}'>{html.escape(str(c))}</th>" for i, c in enumerate(headers))
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td class='{'n' if i in num_cols else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def build_report(res: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    st, b, park = res["study"], res["best"], res["study"]["park"]
    scen = res["meta"]["scenarios"]
    v_title, v_text = VERDICT[res["verdict"]]
    vcol = {"yes": "#1b7f3b", "only_with_risk": "#9a6700", "no": "#b3261e"}[res["verdict"]]
    cat_text = {"11A": "σταθμός 11Α (φορτίζει μόνο από το ΦΒ, κρατά την ενίσχυση)", "11B": "σταθμός 11Β (φορτίζει και από το δίκτυο, χωρίς ενίσχυση)"}[b["category"]]
    support = {"fit": "Σταθερή τιμή (FiT)", "fip": "Διαφορική προσαύξηση (ΣΕΔΠ)", "merchant": "Χωρίς ενίσχυση (αγορά)", "ppa": "Διμερές συμβόλαιο (PPA)"}[park["support"]["kind"]]
    be = res["risk"]["break_even"]
    P = []
    P.append(f"<h1>Μελέτη μπαταρίας: {html.escape(park['name'])}</h1>")
    P.append(f"<p class='sub'>Εργαλείο bessdss {res['meta']['version']}. Έναρξη λειτουργίας μπαταρίας {st['cod_year']}, ορίζοντας {st['life_years']} έτη, {res['meta']['n_paths']} διαδρομές Monte Carlo. Όλα τα ποσά εσόδων σε σταθερά ευρώ {st['base_year']}, η NPV σε ευρώ έτους επένδυσης.</p>")
    lead = "Το λιγότερο κακό μέγεθος" if res["verdict"] == "no" else "Προτεινόμενο μέγεθος"
    P.append(f"<div class='verdict'><b>{v_title}</b><br>{v_text}<br><span class='note'>{lead}: <b>{num(b['power_mw'], 2)} MW / {num(b['energy_mwh'], 2)} MWh</b> ({num(b['duration_h'], 1)} ώρες ωφέλιμα), {cat_text}.</span></div>".replace("VCOL", vcol))
    tiles = [
        ("Κόστος εγκατάστασης", eur(b["capex_eur"])), ("NPV, μέσος όρος", eur(b["npv_mean"])), ("NPV στο κακό σενάριο (P90)", eur(b["npv_p90"])),
        ("Πιθανότητα ζημιάς", pct(b["prob_npv_negative"], 0)), ("IRR έργου (P50)", pct(b["irr_p50"])), ("Απόσβεση (P50)", num(b["payback_p50"], 1, " έτη") if np.isfinite(b["payback_p50"]) else "ποτέ μέσα στη ζωή"),
        ("DSCR στο κακό σενάριο", num(b["dscr_min_p90"], 2)), ("Πρόσθετο έσοδο 1ου έτους (P50)", eur(b["revenue_y1_p50"])),
        ("Κόστος ανά MWh που εκφορτίζει (LCOS)", eur(b["lcos_p50"])), ("CAPEX που μηδενίζει την NPV", eur(be["break_even_capex_eur_per_kwh"]) + "/kWh"),
    ]
    P.append("<div class='tiles'>" + "".join(f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in tiles) + "</div>")
    if res["compliance_status"] == "fail":
        P.append("<div class='warnbox'><b>Ρυθμιστικός έλεγχος: υπάρχει όρος που δεν πληρείται.</b> Δείτε την ενότητα 7.</div>")
    P.append("<p class='note'>P50: το αποτέλεσμα που ξεπερνιέται στις μισές περιπτώσεις. P90: το αποτέλεσμα που ξεπερνιέται στις 9 από τις 10, δηλαδή ένας κακός συνδυασμός αγοράς, καιρού και κόστους. Η μελέτη είναι εργαλείο υποστήριξης απόφασης και όχι επενδυτική ή νομική συμβουλή.</p>")

    # 1 park
    P.append("<h2>1. Το πάρκο και τα δεδομένα του</h2>")
    P.append(_table(["Στοιχείο", "Τιμή"], [
        ["Τοποθεσία", f"{park['lat']}, {park['lon']}"], ["Ισχύς πλαισίων / μετατροπέων", f"{num(park['pv_dc_kwp'], 0)} kWp / {num(park['pv_ac_kw'], 0)} kW"],
        ["Μέγιστη ισχύς έγχυσης", f"{num(park['grid_export_limit_kw'], 0)} kW"], ["Έναρξη λειτουργίας", park.get("commissioning_date") or park["commissioning_year"]],
        ["Καθεστώς", support + (f", {num(park['support']['price_eur_mwh'], 1)} €/MWh έως {park['support']['end_year']}" if park["support"].get("price_eur_mwh") else "")],
        ["Διαχειριστής", park["grid_operator"].upper()], ["Υφιστάμενο όριο έγχυσης", pct(park["static_injection_cap_frac"], 0) + " της εγκατεστημένης ισχύος"],
        ["Παραγωγή ΦΒ", html.escape(res["pv"]["source"])],
        ["Ειδική παραγωγή P50 / P90", f"{num(res['pv']['yield_p50'], 0)} / {num(res['pv']['yield_p90'], 0)} kWh/kWp (19 έτη, πριν από τον μετατροπέα)"],
    ]))
    P.append("<p class='note'>Η παραγωγή είναι μοντελοποιημένη από δορυφορικά δεδομένα και όχι μέτρηση του πάρκου, εκτός αν ο ιδιοκτήτης έδωσε δικές του μετρήσεις. Οι τιμές αγοράς του μέλλοντος είναι σενάρια και όχι πρόβλεψη.</p>")

    # 2 sizing
    P.append("<h2>2. Ποιο μέγεθος</h2>")
    P.append(f"<p>Το μέγεθος είναι αποτέλεσμα της μελέτης. Ο νόμος ζητά ωφέλιμη χωρητικότητα τουλάχιστον μίας ώρας της εγκατεστημένης ισχύος του ΦΒ, άρα η μικρότερη επιτρεπτή μπαταρία εδώ είναι <b>{num(res['legal_minimum_energy_mwh'], 2)} MWh</b> ονομαστικά. Εξετάστηκαν {len(res['screening'])} συνδυασμοί ισχύος, χωρητικότητας και νομικής κατηγορίας. Οι καλύτεροι αξιολογήθηκαν ξανά σε 15λεπτη ανάλυση με όλα τα καιρικά έτη:</p>")
    rows = [[m["category"], num(m["power_mw"], 2), num(m["energy_mwh"], 2), num(m["duration_h"], 1), eur(m["capex_eur"]), eur(m["npv_mean"]), eur(m["npv_p90"]), pct(m["prob_npv_negative"], 0), pct(m["irr_p50"]), num(m["dscr_min_p90"], 2)] for m in res["finals"]]
    P.append(_table(["Κατηγορία", "MW", "MWh", "Ώρες", "CAPEX", "NPV μέσος", "NPV P90", "Πιθ. ζημιάς", "IRR P50", "DSCR P90"], rows, num_cols=range(1, 10)))
    mg = res["marginal"]
    P.append(f"<p>Οριακή αξία στο επιλεγμένο μέγεθος (από τα duals της βελτιστοποίησης): ένα επιπλέον MW ισχύος αλλάζει την αναμενόμενη NPV κατά <b>{eur(mg['dnpv_dpower'])}</b> και μία επιπλέον MWh χωρητικότητας κατά <b>{eur(mg['dnpv_denergy'])}</b>. Αρνητική τιμή στη χωρητικότητα σημαίνει ότι η μπαταρία θα ήταν καλύτερη μικρότερη, αν το επέτρεπε ο κανόνας της μίας ώρας.</p>")
    rows = [[m["category"], num(m["power_mw"], 2), num(m["energy_mwh"], 2), num(m["duration_h"], 1), eur(m["capex_eur"]), eur(m["npv_mean"]), eur(m["npv_p90"]), eur(m["revenue_y1_p50"])] for m in res["screening"]]
    P.append("<details><summary>Όλοι οι συνδυασμοί της πρώτης διαλογής (ωριαία ανάλυση, τρία καιρικά έτη)</summary>" + _table(["Κατηγορία", "MW", "MWh", "Ώρες", "CAPEX", "NPV μέσος", "NPV P90", "Έσοδο 1ου έτους"], rows, num_cols=range(1, 8)) + "</details>")

    # 3 value
    P.append("<h2>3. Από πού έρχεται η αξία</h2>")
    P.append("<p>Το πρόσθετο έσοδο έχει δύο μέρη. Το πρώτο είναι το κόστος της μετατροπής: το πάρκο γίνεται σταθμός με αποθήκευση και αποκτά μόνιμο όριο έγχυσης (και στην κατηγορία 11Β χάνει την ενίσχυση). Το δεύτερο είναι αυτό που κερδίζει η μπαταρία: ενέργεια που σώζεται από το όριο, από περικοπές και από ώρες μηδενικής τιμής, και μετακίνηση σε ακριβότερες ώρες.</p>")
    P.append(f"<img class='sm' src='{fig_decomposition(res)}' alt='Ανάλυση αξίας'>")
    dd = res["design_detail"]
    i = dd["scen_names"].index("central") if "central" in dd["scen_names"] else 0
    rows = [[y, eur(dd["base_rev"][i][k]), eur(dd["penalty"][i][k]), eur(dd["gross"][i][k]), num(dd["discharged"][i][k], 0), num(dd["fec"][i][k], 0), num(dd["lost_base"][i][k], 0), num(dd["lost_with"][i][k], 0)] for k, y in enumerate(dd["rep_years"])]
    P.append(_table(["Έτος", "Έσοδο πάρκου σήμερα", "Κόστος μετατροπής", "Αξία μπαταρίας", "Εκφόρτιση MWh", "Κύκλοι", "Χαμένη ενέργεια χωρίς μπαταρία MWh", "Χαμένη με μπαταρία MWh"], rows, num_cols=range(1, 8)))
    P.append(f"<img class='sm' src='{fig_revenue(res)}' alt='Έσοδο ανά έτος'>")
    P.append(f"<img class='sm' src='{fig_soh(res)}' alt='Φθορά'>")
    cyc = max(max(r) for r in dd["cycled_in"])
    if cyc > 5:
        P.append(f"<div class='warnbox'>Σε κάποια έτη η βέλτιστη λειτουργία περνά έως {num(cyc, 0)} MWh τον χρόνο από τη μπαταρία μέσα στο ίδιο 15λεπτο. Αυτό συμβαίνει όταν οι κανόνες το ανταμείβουν (π.χ. αποφυγή της επιστροφής προσαύξησης όταν η αγορά είναι πάνω από την τιμή αναφοράς). Είναι έσοδο που ο ρυθμιστής μπορεί να κλείσει.</div>")

    # 4 operation
    op = res["operation"]
    P.append("<h2>4. Πώς πρέπει να λειτουργεί</h2>")
    pol = op["policies"]
    pf = pol["perfect"]["gross_value"]
    P.append(f"<p>Τρεις τυπικές μέρες του {op['year']} στο κεντρικό σενάριο (καιρός του {op['weather_year']}). Η μπαταρία φορτίζει όταν η ενέργεια του ΦΒ αξίζει λίγο ή δεν χωρά στο όριο έγχυσης, και εκφορτίζει στις ακριβές ώρες, κυρίως μετά τις 19:00 όταν το όριο δεν ισχύει.</p>")
    P.append(f"<img src='{fig_days(res)}' alt='Τυπικές μέρες'>")
    P.append("<p>Η αξία εξαρτάται από το πόσο καλά ξέρει κανείς τις τιμές της επόμενης μέρας. Η ίδια χρονιά με τρεις τρόπους λειτουργίας:</p>")
    P.append(_table(["Τρόπος λειτουργίας", "Αξία μπαταρίας", "Ως ποσοστό της τέλειας γνώσης", "Κύκλοι"], [
        ["Τέλεια γνώση των τιμών (άνω όριο)", eur(pf), "100%", num(pol["perfect"]["fec"], 0)],
        ["Πρόγραμμα με τις χθεσινές τιμές (απλή πρόβλεψη)", eur(pol["persistence"]["gross_value"]), pct(pol["persistence"]["gross_value"] / pf, 0) if pf else "δ/υ", num(pol["persistence"]["fec"], 0)],
        ["Σταθερός κανόνας ρολογιού (χωρίς βελτιστοποίηση)", eur(pol["rules"]["gross_value"]), pct(pol["rules"]["gross_value"] / pf, 0) if pf else "δ/υ", num(pol["rules"]["fec"], 0)],
    ], num_cols=(1, 2, 3)))
    P.append("<p class='note'>Τα οικονομικά της μελέτης χρησιμοποιούν την τέλεια γνώση επί έναν συντελεστή πρόβλεψης (βλ. υποθέσεις), ώστε να μην είναι αισιόδοξα.</p>")

    # 5 scenarios
    P.append("<h2>5. Σενάρια αγοράς</h2>")
    P.append("<p>Τα σενάρια δεν είναι προβλέψεις. Σε κάθε σενάριο η ισχύς ΦΒ, αιολικών και μπαταριών, η ζήτηση και οι τιμές καυσίμων ακολουθούν μια συνεπή ιστορία, και κάθε έτος προσομοιώνεται με τον καιρό εννέα πραγματικών ετών, ώστε ηλιοφάνεια, τιμές και περικοπές να κινούνται μαζί. Όπου σημειώνεται, η ισχύς των μπαταριών της χώρας είναι ενδογενής: μπαίνουν νέες όσο η διαφορά τιμών καλύπτει το κόστος τους.</p>")
    P.append(f"<img src='{fig_market(res)}' alt='Σενάρια αγοράς'>")
    rows = [[html.escape(r["label"]), pct(r["weight"], 0), eur(r["npv_mean"]), eur(r["npv_p90"]), pct(r["irr_p50"]), num(r["dscr_min_p90"], 2), eur(r["revenue_y1"]), eur(r["revenue_last"])] for r in res["risk"]["per_scenario"]]
    P.append(_table(["Σενάριο", "Βάρος", "NPV μέσος", "NPV P90", "IRR P50", "DSCR P90", "Έσοδο 1ου έτους", "Έσοδο τελευταίου έτους"], rows, num_cols=range(1, 8)))
    P.append("<ul>" + "".join(f"<li><b>{html.escape(v['label'])}:</b> {html.escape(v['description'])}</li>" for v in scen.values()) + "</ul>")
    fz = res.get("frozen_2026")
    if fz:
        P.append(f"<div class='warnbox'><b>Γιατί δεν αρκεί η επανάληψη της ιστορίας.</b> Αν η αγορά έμενε για πάντα όπως το 2026 (ίδια ΦΒ, λίγες μπαταρίες, ακριβό αέριο), το ίδιο έργο θα είχε NPV {eur(fz['npv_mean'])} και πρόσθετο έσοδο πρώτου έτους {eur(fz['revenue_y1'])}, έναντι {eur(b['npv_mean'])} και {eur(b['revenue_y1_p50'])} στα σενάρια της μελέτης. Η διαφορά είναι το αποτέλεσμα των νέων μπαταριών και ΦΒ που συμπιέζουν τη διαφορά τιμών.</div>")

    # 6 risk
    P.append("<h2>6. Ανάλυση κινδύνου</h2>")
    P.append(f"<img class='sm' src='{fig_npv(res)}' alt='Κατανομή NPV'>")
    P.append(f"<img class='sm' src='{fig_sensitivity(res)}' alt='Πηγές αβεβαιότητας'>")
    P.append(f"<img class='md' src='{fig_tornado(res)}' alt='Tornado'>")
    dn = res["risk"]["downside"]
    lab = res["risk"]["labels"]
    worst_scen = max(dn["scenario"].items(), key=lambda kv: kv[1])
    shifts = sorted(((k, v) for k, v in dn.items() if k != "scenario"), key=lambda kv: -abs(kv[1]))
    P.append(f"<p><b>Τι οδηγεί το κακό αποτέλεσμα.</b> Στο χειρότερο 10% των διαδρομών, το {pct(worst_scen[1], 0)} ανήκει στο σενάριο «{html.escape(scen[worst_scen[0]]['label'])}». Οι υπόλοιποι παράγοντες αποκλίνουν από τον μέσο τους ως εξής (σε τυπικές αποκλίσεις): " + ", ".join(f"{lab[k]} {num(v, 2)}" for k, v in shifts) + ".</p>")
    P.append("<h3>Τραπεζική εικόνα</h3>")
    fin = res["finance"]
    P.append(f"<p>Δάνειο {pct(fin['gearing'], 0)} του κόστους, {fin['debt_tenor_years']} έτη, επιτόκιο {pct(fin['debt_rate'])}. Ο δείκτης υπολογίζεται μόνο στις πρόσθετες ροές της μπαταρίας, χωρίς τις ροές του πάρκου. Η τράπεζα κοιτά τον κακό χρόνο, όχι τον μέσο.</p>")
    P.append(f"<img class='sm' src='{fig_dscr(res)}' alt='DSCR'>")
    P.append(_table(["Μέγεθος", "Τιμή"], [
        ["Ελάχιστος DSCR, P50", num(b["dscr_min_p50"], 2)], ["Ελάχιστος DSCR στο κακό σενάριο (P90)", num(b["dscr_min_p90"], 2)],
        ["Πιθανότητα DSCR κάτω από 1 σε κάποιο έτος", pct(b["prob_dscr_below_1"], 0)], ["Απαίτηση τράπεζας (υπόθεση)", num(fin["target_dscr"], 2)],
        ["CAPEX που μηδενίζει την αναμενόμενη NPV", f"{eur(be['break_even_capex_eur'])} ({eur(be['break_even_capex_eur_per_kwh'])}/kWh, {pct(be['capex_multiplier_for_zero_npv'], 0)} του σημερινού)"],
        ["Επιδότηση κεφαλαίου που μηδενίζει την αναμενόμενη NPV", pct(be["subsidy_share_for_zero_npv"], 0) if be["subsidy_share_for_zero_npv"] <= 1 else "πάνω από 100%: δεν αρκεί καμία επιδότηση"],
    ]))

    # 7 compliance
    P.append("<h2>7. Ρυθμιστικός έλεγχος για αυτό το πάρκο</h2>")
    rows = []
    for c in res["compliance"]:
        icon, col, word = STATUS[c["status"]]
        refs = "<br>".join(f"<span class='small'>{r['id']}: {html.escape(r['citation'][:150])} (επαλήθευση {r['verified_on']}, {r['confidence']})</span>" for r in c["rules"])
        rows.append([f"<span class='st' style='color:{col}'>{icon}</span> {word}", f"<b>{html.escape(c['title'])}</b><br>{html.escape(c['text'])}<br>{refs}"])
    P.append(_table(["Κατάσταση", "Έλεγχος και κανόνας"], rows))
    P.append("<h3>Πώς αλλάζει το αποτέλεσμα με άλλη ανάγνωση των κανόνων</h3>")
    rows = [[html.escape(r["label"]), eur(r["npv_mean"]), eur(r["npv_p90"]), eur(r["revenue_y1"])] for r in res["regulatory_sensitivity"]]
    P.append(_table(["Ανάγνωση", "NPV μέσος", "NPV P90", "Έσοδο 1ου έτους"], rows, num_cols=(1, 2, 3)))
    P.append("<p class='note'>Υπολογισμός σε ωριαία ανάλυση με τρία καιρικά έτη, γι' αυτό η βασική γραμμή διαφέρει λίγο από τον κύριο πίνακα. Το πλήρες ρυθμιστικό μητρώο με τις πηγές είναι στο docs/REGULATORY_REGISTER.md.</p>")

    # 8 assumptions
    P.append("<h2>8. Υποθέσεις και αβεβαιότητα</h2>")
    kind = {"sourced": "από πηγή", "derived": "υπολογισμός από πηγές", "expert_guess": "εκτίμηση χωρίς πηγή"}
    rows = [[a["id"], html.escape(a["description"]), html.escape(str(a["unit"])), a["low"], a["central"], a["high"], kind.get(a["confidence"], a["confidence"])] for a in res["assumptions"]]
    P.append(_table(["ID", "Υπόθεση", "Μονάδα", "Χαμηλή", "Κεντρική", "Υψηλή", "Είδος"], rows, num_cols=(3, 4, 5)))
    P.append(f"<p>Κόστος φθοράς που βλέπει η βελτιστοποίηση: {num(res['cycle_cost_eur_mwh'], 1)} €/MWh που εκφορτίζεται (αξία της χωρητικότητας που χάνεται σε έναν κύκλο, από το μοντέλο φθοράς).</p>")
    P.append("<h3>Τι δεν κάνει αυτή η μελέτη</h3><ul>"
             "<li>Δεν προβλέπει τιμές. Το μοντέλο τιμών έχει σφάλμα περίπου ±25% στη διαφορά τιμών όταν δοκιμάζεται σε χρονιά που δεν έχει δει, και αυτό μπαίνει στην ανάλυση κινδύνου.</li>"
             "<li>Δεν μετρά έσοδα από την αγορά εξισορρόπησης ούτε από την ενδοημερήσια αγορά, ούτε το όφελος από μικρότερες αποκλίσεις.</li>"
             "<li>Οι περικοπές του συγκεκριμένου πάρκου δεν είναι γνωστές. Χρησιμοποιείται ο μέσος όρος του συστήματος επί έναν πολλαπλασιαστή που ορίζει ο χρήστης.</li>"
             "<li>Η παραγωγή μέσα στην ώρα είναι ομαλή (παρεμβολή), άρα δεν φαίνονται τα σύννεφα.</li>"
             "<li>Τα κόστη είναι εκτιμήσεις με εύρος. Πριν από απόφαση χρειάζεται πραγματική προσφορά.</li>"
             "<li>Οι κανόνες αλλάζουν συχνά. Η ημερομηνία επαλήθευσης κάθε κανόνα φαίνεται στην ενότητα 7.</li></ul>")
    P.append(f"<p class='small'>Χρόνος υπολογισμού {res['meta']['run_seconds']} s. Αναπαραγωγή: <code>bessdss study &lt;park.yaml&gt;</code>. Όλα τα νούμερα της αναφοράς υπάρχουν στο results.json.</p>")
    doc = f"<!doctype html><html lang='el'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Μελέτη μπαταρίας</title><style>{CSS.replace('VCOL', vcol)}</style></head><body>{''.join(P)}</body></html>"
    out = out_dir / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out


def rebuild(results_json: str | Path) -> Path:
    p = Path(results_json)
    return build_report(json.loads(p.read_text(encoding="utf-8")), p.parent)
