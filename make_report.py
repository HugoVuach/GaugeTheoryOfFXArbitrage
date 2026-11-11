"""Transforme les CSV de l'analyse en figures vectorielles et en tables LaTeX."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

INK = "#142639"
TEAL = "#087F79"
RUST = "#A14B31"
SLATE = "#5D7284"
GRID = "#DCE6EB"
PALE = "#C4D2DA"
BLUE = "#2D587C"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.4, "axes.labelsize": 9.4,
    "axes.titlesize": 11.2, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelcolor": INK, "xtick.color": SLATE, "ytick.color": SLATE,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.edgecolor": PALE, "grid.color": GRID, "grid.linewidth": 0.65,
    "legend.frameon": False, "legend.fontsize": 8.1, "text.color": INK,
    "pdf.fonttype": 42, "figure.facecolor": "white",
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
})


def integer(x):
    return f"{int(x):,}".replace(",", r"\,")


def decimal(x, places=2):
    return f"{x:.{places}f}".replace("-", r"\ensuremath{-}").replace(".", ",")


def percent(n, total, places=1):
    return decimal(100 * n / total, places) if total else "--"


def grouped(result_dir, file, group):
    frame = pd.read_csv(result_dir / f"{file}.csv")
    july = frame[frame["date"].str.startswith("2024-07")]
    return july.drop(columns="date").groupby(group).sum()


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.035,
                metadata={"Title": name, "Creator": "Analyse HistData — projet de jauge"})
    fig.savefig(out / f"{name}.pgf", backend="pgf")
    plt.close(fig)


def geometry_plot(july, out):
    fig, ax = plt.subplots(figsize=(7.9, 3.3), layout="constrained")
    for signal, color, label in (
        (False, BLUE, "Fourchettes compatibles"),
        (True, RUST, "Cycle positif après spread"),
    ):
        points = july[july["signal_bidask"] == signal]
        ax.scatter(points["defaut_hodge_pb"], points["dilatation_spread"],
                   s=10, alpha=0.30, linewidths=0, color=color, label=label)
    ax.axhline(1, color=SLATE, lw=1.1, ls="--", label="Seuil de compatibilité")
    french = FuncFormatter(lambda value, _: f"{value:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(french)
    ax.yaxis.set_major_formatter(french)
    ax.set(xlabel="Distance RMS au potentiel (pb log)",
           ylabel=r"Dilatation minimale $\Lambda_\star$",
           xlim=(0, None), ylim=(0, None))
    ax.set_title("Juillet 2024 — géométrie des taux et coût des conversions", loc="left")
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    save(fig, out, "geometrie_juillet")


def fee_age_plot(ages, fees, out, count_grid):
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.0), layout="constrained")
    ax = axs[0]
    ages_s = ages.index.to_numpy() / 1000
    ax.plot(ages_s, 100 * ages["n"] / count_grid, "o-", lw=2, ms=4.4, color=TEAL,
            label="Instants admissibles / grille")
    ax.plot(ages_s, 100 * ages["alertes"] / ages["n"], "s--", lw=1.8, ms=4.2, color=RUST,
            label="Alertes / instants admissibles")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 5, 30], ["0,1", "0,25", "0,5", "1", "2", "5", "30"])
    ax.set(xlabel="Âge maximal autorisé (s)", ylabel="Part (%)", ylim=(-3, 105))
    ax.set_title("A  Fraîcheur des six paires", loc="left")
    ax.axvline(1, color=SLATE, ls=":", lw=1.1)
    ax.grid(axis="y")
    ax.legend(loc="upper left")

    ax = axs[1]
    small_fees = fees.loc[:0.1]
    ax.plot(small_fees.index, small_fees["alertes"], "o-", color=TEAL, ms=4.6, lw=2)
    ax.set_xticks([0, 0.025, 0.05, 0.075, 0.1], ["0", "0,025", "0,05", "0,075", "0,1"])
    ax.set(xlabel="Frais ajoutés par conversion (pb log)", ylabel="Instants avec une alerte", ylim=(-25, 820))
    ax.set_title("B  Sensibilité aux frais", loc="left")
    ax.grid(axis="y")
    save(fig, out, "fraicheur_et_frais")


def cpi_plot(result_dir, out):
    frame = pd.read_csv(result_dir / "profil_ipc_2024-07-11.csv")
    t = frame["secondes"]
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 3.0), sharex=True, layout="constrained",
                            gridspec_kw={"height_ratios": [1.8, 1]})
    ax = axs[0]
    ax.plot(t, frame["marge_mid_pb"], color=TEAL, lw=1.45, label="Taux médians")
    ax.plot(t, frame["marge_bidask_pb"], color=RUST, lw=1.3, label="Bid–ask")
    ax.axhline(0, color=SLATE, lw=0.9, zorder=0)
    ax.set_ylabel("Meilleure moyenne\n(pb log / conversion)")
    ax.set_title("11 juillet 2024 — autour de la publication de l’IPC américain", loc="left")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y")
    ax = axs[1]
    ax.plot(t, frame["age_max_ms"] / 1000, color=BLUE, lw=1.25)
    ax.axhline(1, color=RUST, ls="--", lw=1, label="Seuil de 1 s")
    ax.set(xlabel="Temps par rapport à 12 h 30 UTC (s)", ylabel="Âge maximal (s)", xlim=(-30, 120))
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    for ax in axs:
        ax.axvline(0, color=SLATE, lw=1, ls=":")
    save(fig, out, "ipc_juillet")


def stale_plot(stale, out):
    fig, ax = plt.subplots(figsize=(7.9, 2.5), layout="constrained")
    x = stale.index / 1000
    ax.plot(x, stale["reel_creees"], "o-", color=TEAL, lw=2, ms=4.4, label="Flux réel — nouvelles alertes")
    ax.plot(x, stale["fausses_alertes_nul"], "s-", color=RUST, lw=1.9, ms=4.2,
            label="Modèle nul — fausses alertes")
    ax.plot(x, stale["reel_creees_age_1s"], "o--", color=TEAL, lw=1.4, ms=4,
            markerfacecolor="white", label="Flux réel — avec filtre d’âge")
    ax.plot(x, stale["fausses_alertes_nul_age_1s"], "s--", color=RUST, lw=1.4, ms=4,
            markerfacecolor="white", label="Modèle nul — avec filtre d’âge")
    ax.set(xlabel="Retard injecté sur EUR/USD (s)", ylabel="Instants avec une nouvelle alerte")
    ax.set_xticks([0, 0.5, 1, 2, 3, 4, 5], ["0", "0,5", "1", "2", "3", "4", "5"])
    ax.set_ylim(-45, 1250)
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    save(fig, out, "retard_eurusd")


def replay_plot(latency, out):
    fig, ax = plt.subplots(figsize=(7.9, 2.8), layout="constrained")
    x = np.arange(len(latency))
    yes = latency["brutes_positives"].to_numpy()
    evaluated = latency["brutes_evaluables"].to_numpy()
    nominal = latency["alertes_brutes"].to_numpy()
    ax.bar(x, yes, width=0.54, color=TEAL, label="Gain positif")
    ax.bar(x, evaluated - yes, bottom=yes, width=0.54, color=RUST, label="Gain nul ou négatif")
    ax.bar(x, nominal - evaluated, bottom=evaluated, width=0.54, color=PALE,
           label="Au moins une cotation trop ancienne")
    for i, number in enumerate(yes):
        ax.text(i, max(number / 2, 13), f"{number}", color="white", ha="center", va="center",
                fontsize=9, fontweight="bold")
    ax.set_xticks(x, [str(v) for v in latency.index])
    ax.set(xlabel="Délai par conversion (ms)", ylabel="Alertes initiales", ylim=(0, 835))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, borderaxespad=0.15)
    save(fig, out, "rejeu")


def table_days(daily, path):
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"Date & Âge $\leq 1$ s & Médian $+$ & Bid--ask $+$ & Triangle $+$ & $\mu_{\max}$\\",
        r"\midrule",
    ]
    for row in daily.itertuples():
        day = row.date[-2:].lstrip("0")
        if row.date == "2024-07-08":
            lines.append(r"\midrule")
        values = [f"{day} {row.phase}", integer(row.n_1s), integer(row.mid_1s),
                  integer(row.bidask_1s), integer(row.triangle_1s), decimal(row.marge_max_pb, 3)]
        lines.append(" & ".join(values) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_latency(latency, path):
    lines = [
        r"\begin{tabular}{@{}rrrrrr@{}}", r"\toprule",
        r"$\Delta$ (ms) & Rejouables & Positives & Positives / rejouables & Robustes & Couverture jointe\\",
        r"\midrule",
    ]
    for t, row in latency.iterrows():
        values = [integer(t), integer(row.brutes_evaluables), integer(row.brutes_positives),
                  percent(row.brutes_positives, row.brutes_evaluables) + r"\,\%",
                  integer(row.alertes_robustes),
                  percent(row.enveloppe_jointe_ok, row.enveloppe_evaluable) + r"\,\%"]
        lines.append(" & ".join(values) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def values_tex(result_dir, daily, latency, stale, fees, july_geometry, path):
    july = daily[daily["phase"] == "juillet"]
    june = daily[daily["phase"] == "juin"]
    summary = json.loads((result_dir / "synthese.json").read_text(encoding="utf-8"))
    late = latency.loc[100]
    late500 = latency.loc[500]
    delayed = stale.loc[500]
    delayed5 = stale.loc[5000]
    big_days = july[july["date"].isin(["2024-07-11", "2024-07-12"])]["bidask_1s"].sum()
    radii100 = pd.read_csv(result_dir / "rayons_calibres.csv").query("delai_ms == 100")["rayon_pb"]
    july_peak = max((e for e in summary["pics_par_jour"] if e["instant"].startswith("2024-07")),
                    key=lambda e: e["mean_pb"])
    geometry_positive = july_geometry["signal_bidask"]
    positive_hodge_median = july_geometry.loc[geometry_positive, "defaut_hodge_pb"].median()
    negative_above_median = (~geometry_positive & (july_geometry["defaut_hodge_pb"] > positive_hodge_median))
    def sci(x):
        a, b = f"{x:.2e}".split("e")
        return r"\ensuremath{" + a.replace(".", "{,}") + rf"\times 10^{{{int(b)}}}" + "}"
    values = {
        "TotalGrille": integer(daily["grille"].sum()),
        "TotalCommuns": integer(daily["n_30s"].sum()),
        "TotalEligibles": integer(daily["n_1s"].sum()),
        "TicksSource": integer(summary["lignes_dans_fenetres_avec_marges"]),
        "JuinMarge": decimal(june["marge_max_pb"].max(), 3),
        "JuilletGrille": integer(july["grille"].sum()), "JuilletEligibles": integer(july["n_1s"].sum()),
        "JuilletEligiblesPct": percent(july["n_1s"].sum(), july["grille"].sum()),
        "JuilletMed": integer(july["mid_1s"].sum()), "JuilletNet": integer(july["bidask_1s"].sum()),
        "JuilletNetPct": percent(july["bidask_1s"].sum(), july["n_1s"].sum(), 2),
        "JuilletCommun": integer(july["n_30s"].sum()), "JuilletCommunNet": integer(july["bidask_30s"].sum()),
        "JuilletCommunPct": percent(july["bidask_30s"].sum(), july["n_30s"].sum(), 2),
        "JuilletEpisodes": integer(july["episodes_1s"].sum()),
        "JuilletPicDeuxJours": integer(big_days),
        "JuilletMarge": decimal(july["marge_max_pb"].max(), 3),
        "JuilletGainPic": decimal(july_peak["gain_pb"], 3),
        "JuilletPicHeure": july_peak["instant"][11:23].replace(".", ","),
        "JuilletPicAge": integer(july_peak["age_ms"]),
        "JuilletFeeCinq": integer(fees.loc[0.05, "alertes"]),
        "JuilletFeeDix": integer(fees.loc[0.1, "alertes"]),
        "KarpErreur": sci(daily["erreur_karp_pb"].max()),
        "JaugeErreur": sci(summary["verifications"]["erreur_jauge_pb"]),
        "DualErreur": sci(summary["verifications"]["erreur_dual_programme_linéaire"]),
        "DilatationErreur": sci(summary["verifications"]["erreur_dual_dilatation_spread"]),
        "HodgeJuilletMed": decimal(july_geometry["defaut_hodge_pb"].median(), 3),
        "HodgeJuilletMedPositifs": decimal(positive_hodge_median, 3),
        "HodgeJuilletNonPositifs": integer((~geometry_positive).sum()),
        "HodgeJuilletNonPositifsForts": integer(negative_above_median.sum()),
        "DilatationJuilletMed": decimal(july_geometry["dilatation_spread"].median(), 3),
        "DilatationJuilletMax": decimal(july_geometry["dilatation_spread"].max(), 3),
        "ReplayCentOk": integer(late.brutes_positives),
        "ReplayCentEval": integer(late.brutes_evaluables),
        "ReplayCentPartInitiale": percent(late.brutes_positives, late.alertes_brutes),
        "ReplayCentPairesOk": percent(late.enveloppe_jointe_ok, late.enveloppe_evaluable, 2),
        "ReplayCentAlertesCouv": percent(late.enveloppe_alertes_ok, late.enveloppe_alertes_evaluable, 2),
        "ReplayCentStrictOk": integer(late.age_250_positives),
        "ReplayCentStrictTotal": integer(late.alertes_age_250),
        "ReplayCinqCentsOk": integer(late500.brutes_positives),
        "ReplayCinqCentsEval": integer(late500.brutes_evaluables),
        "ReplayCinqCentsPartInitiale": percent(late500.brutes_positives, late500.alertes_brutes),
        "RayonCentMin": decimal(radii100.min(), 3),
        "RayonCentMax": decimal(radii100.max(), 3),
        "RetardCohorte": integer(delayed.cohorte),
        "RetardReelCinqCents": integer(delayed.reel_creees),
        "RetardReelCinqCentsFiltre": integer(delayed.reel_creees_age_1s),
        "RetardReelCinqCentsDisparues": integer(delayed.reel_disparues),
        "RetardNulCinqCents": integer(delayed.fausses_alertes_nul),
        "RetardNulCinqCentsFiltre": integer(delayed.fausses_alertes_nul_age_1s),
        "RetardNulCinqSecs": integer(delayed5.fausses_alertes_nul),
        "RetardNulCinqSecsPct": percent(delayed5.fausses_alertes_nul, delayed5.cohorte, 2),
        "RetardReelCinqSecs": integer(delayed5.reel_creees),
    }
    text = ["% Généré par make_report.py depuis les résultats de l'analyse."]
    text += [rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items()]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def standalone_tex(template, result_dir, figures, output):
    """Intègre les tables et les tracés vectoriels dans un seul fichier LaTeX."""
    source = template.read_text(encoding="utf-8")
    source = source.replace(r"\input{source/experiences.tex}",
                            (template.parent / "experiences.tex").read_text(encoding="utf-8"))

    def inline_table(match):
        return (result_dir / (match.group(1) + ".tex")).read_text(encoding="utf-8").rstrip()

    source = re.sub(r"\\input\{results/([a-z0-9_]+)\.tex\}", inline_table, source)

    def inline_figure(match):
        width, name = match.groups()
        plot = (figures / f"{name}.pgf").read_text(encoding="utf-8")
        plot = plot[plot.index(r"\begingroup"):].rstrip()
        # Lire le code PGF avant le redimensionnement, pour préserver ses catcodes.
        return (r"\begin{lrbox}{\plotbox}" + "%\n" + plot + "\n"
                + r"\end{lrbox}" + "%\n"
                + rf"\resizebox{{{width}}}{{!}}{{\usebox{{\plotbox}}}}")

    source, count = re.subn(r"\\includegraphics\[width=([^\]]+)\]\{([a-z_]+)\.pdf\}", inline_figure, source)
    if count != 10 or r"\input{results/" in source or r"\input{source/" in source:
        raise ValueError("Les dix figures et toutes les tables doivent être intégrées")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")


def run(result_dir, figures, template, output):
    figures.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(result_dir / "par_jour.csv")
    ages = grouped(result_dir, "fraicheur", "age_max_ms")
    fees = grouped(result_dir, "frais", "frais_pb_conversion")
    latency = grouped(result_dir, "latence", "delai_ms")
    stale = grouped(result_dir, "retard_eurusd", "retard_ms")
    geometry = pd.read_csv(result_dir / "geometrie.csv")
    july_geometry = geometry[geometry["instant_utc"].str.startswith("2024-07")]
    geometry_plot(july_geometry, figures)
    fee_age_plot(ages, fees, figures, daily.query("phase == 'juillet'")["grille"].sum())
    cpi_plot(result_dir, figures)
    stale_plot(stale, figures)
    replay_plot(latency, figures)
    table_days(daily, result_dir / "table_jours.tex")
    table_latency(latency, result_dir / "table_latence.tex")
    values_tex(result_dir, daily, latency, stale, fees, july_geometry, result_dir / "valeurs.tex")
    from experience_report import run as report_experiences
    report_experiences(result_dir, figures)
    standalone_tex(template, result_dir, figures, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--template", type=Path, default=Path("source/note.tex"))
    parser.add_argument("--output", type=Path, default=Path("Note_Jauge_Arbitrage_FX_3_Experiences.tex"))
    args = parser.parse_args()
    run(args.results, args.figures, args.template, args.output)
