"""Figures, tableaux et valeurs du rapport des trois expériences."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

import make_report as style

LABELS = {
    "arcs_marginaux": "Arêtes, quantiles marginaux",
    "arcs_simultanes": "Arêtes, contrôle simultané",
    "cycles_simultanes": "Cycles, contrôle simultané",
}
COLORS = {"arcs_marginaux": style.SLATE, "arcs_simultanes": style.BLUE,
          "cycles_simultanes": style.TEAL}
FRENCH = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))


def scientific(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    mantissa = mantissa.replace(".", "{,}")
    return r"\ensuremath{" + mantissa + rf"\times10^{{{int(exponent)}}}" + "}"


def summary_two(result_dir):
    data = pd.read_csv(result_dir / "exp2_resultats.csv")
    totals = data.drop(columns=["date", "meilleure_borne_max_pb"]).groupby(
        ["quantile", "delai_ms", "methode"]).sum()
    totals["couverture_pct"] = 100 * totals.couverture_cycles_ok / totals.couverture_evaluable
    totals["couverture_arcs_pct"] = 100 * totals.couverture_arcs_ok / totals.couverture_evaluable
    return totals


def phase_plot(result_dir, figures):
    data = pd.read_csv(result_dir / "exp1_phase.csv")
    grid = data.pivot(index="demi_spread_pb", columns="alpha_pb", values="cycle_positif")
    x, y = grid.columns.to_numpy(), grid.index.to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.0), layout="constrained")
    ax = axs[0]
    ax.contourf(x, y, grid.to_numpy().astype(float), levels=[-.5, .5, 1.5],
                colors=[style.PALE, style.TEAL])
    ax.plot(4*y, y, "--", color=style.INK, lw=1)
    ax.plot(-4*y, y, "--", color=style.INK, lw=1)
    ax.set(xlabel=r"Période $\alpha$ (pb log)", ylabel="Demi-spread (pb log)",
           xlim=(-2, 2), ylim=(0, .6))
    ax.set_title("A  Région avec un cycle positif", loc="left")
    ax.text(0, .45, "Fourchettes\ncompatibles", ha="center", fontsize=9)
    ax.text(-1.6, .07, "Gain", color="white", fontsize=9)
    ax.text(1.45, .07, "Gain", color="white", fontsize=9)
    ax = axs[1]
    for s, color in ((.05, style.TEAL), (.10, style.RUST), (.30, style.BLUE)):
        ax.plot(x, np.abs(x)-4*s, color=color, lw=2,
                label=f"Demi-spread {s:.2f}".replace(".", ",") + " pb")
    ax.axhline(0, color=style.SLATE, lw=1, ls="--")
    ax.set(xlabel=r"Période $\alpha$ (pb log)", ylabel="Gain du tour favorable (pb log)")
    ax.set_title("B  Holonomie après coûts", loc="left")
    ax.legend(loc="upper center", fontsize=7.7)
    ax.grid(axis="y")
    for ax in axs:
        ax.xaxis.set_major_formatter(FRENCH)
        ax.yaxis.set_major_formatter(FRENCH)
    style.save(fig, figures, "experience_un_phase")


def certificate_plot(result_dir, figures):
    radii = pd.read_csv(result_dir / "exp2_rayons.csv").query("quantile == 0.99 and delai_ms > 0")
    totals = summary_two(result_dir).loc[.99]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.0), layout="constrained")
    ax = axs[0]
    for method, frame in radii.groupby("methode", sort=False):
        ax.plot(frame.delai_ms, frame.rayon_max_pb_conversion, "o-",
                color=COLORS[method], label=LABELS[method], ms=4)
        if method == "arcs_marginaux":
            ax.fill_between(frame.delai_ms, frame.rayon_min_pb_conversion,
                            frame.rayon_max_pb_conversion, alpha=.15, color=COLORS[method])
    ax.axhline(.09299445061363561, ls=":", color=style.RUST, lw=1.3)
    ax.set(xlabel="Délai par conversion (ms)", ylabel="Rayon (pb log / conversion)")
    ax.set_title("A  Marges apprises en juin", loc="left")
    ax.legend(loc="upper left", fontsize=6.9)
    ax.grid(axis="y")
    ax = axs[1]
    for method in LABELS:
        frame = totals.xs(method, level="methode").loc[[100,250,500,1000]]
        ax.plot(frame.index, frame.couverture_pct, "o-", color=COLORS[method],
                label=LABELS[method], ms=4)
    ax.axhline(99, color=style.RUST, lw=1, ls=":")
    ax.set(xlabel="Délai par conversion (ms)", ylabel="Couverture simultanée des cycles (%)",
           ylim=(98.8, 100.08))
    ax.set_title("B  Couverture observée en juillet", loc="left")
    ax.grid(axis="y")
    for ax in axs:
        ax.set_xticks([100,250,500,1000])
        ax.yaxis.set_major_formatter(FRENCH)
    style.save(fig, figures, "experience_deux_certificats")


def replicate_stats(frame, columns, group):
    means = frame.groupby(group)[columns].mean()
    std = frame.groupby(group)[columns].std()
    count = frame.groupby(group)[columns].count()
    half = 1.96 * std / np.sqrt(count)
    return means, half


def lag_plot(data, figures):
    base = data.query("delai_conversion_ms == 0 and not panne").copy()
    base["taux_total"] = 100 * base.fausses_alertes / base.n
    base["taux_frais"] = 100 * base.fausses_alertes_fraiches / base.n_frais
    means, half = replicate_stats(base, ["taux_total","taux_frais"],
                                 ["mise_a_jour_ms","retard_eurusd_ms"])
    fig, axs = plt.subplots(1, 2, figsize=(8,3.0), layout="constrained")
    for ax, col, title in zip(axs, ["taux_total","taux_frais"],
                             ["A  Toutes les observations","B  Filtre d'âge de 1 seconde"]):
        for update, color in ((20,style.TEAL),(100,style.RUST),(500,style.BLUE)):
            m, h = means.loc[update], half.loc[update]
            ax.errorbar(m.index, m[col], yerr=h[col], fmt="o-", capsize=2,
                        color=color, label=f"Actualisation tau = {update} ms", ms=4)
        ax.set(xlabel="Retard ajouté sur EUR/USD (ms)", ylabel="Fausses alertes instantanées (%)",
               ylim=(0,100))
        ax.set_title(title,loc="left")
        ax.grid(axis="y")
        ax.set_xticks([0,100,500,1000])
        ax.yaxis.set_major_formatter(FRENCH)
    axs[0].legend(loc="upper left",fontsize=7.1)
    style.save(fig, figures, "experience_trois_retards")


def execution_plot(data, figures):
    frame = data.query("mise_a_jour_ms == 100 and retard_eurusd_ms == 500 and not panne").copy()
    frame["part_positive"] = 100 * frame.rejeu_positifs / frame.rejeu_n
    means, half = replicate_stats(frame, ["part_positive","gain_moyen_pb_conversion"], "delai_conversion_ms")
    fig, axs = plt.subplots(1,2,figsize=(8,3.0),layout="constrained")
    for ax,col,title,ylabel in (
        (axs[0],"part_positive","A  Rejeux à gain positif","Part des alertes initiales (%)"),
        (axs[1],"gain_moyen_pb_conversion","B  Gain moyen réalisé","Moyenne (pb log / conversion)"),
    ):
        ax.errorbar(means.index, means[col], yerr=half[col], color=style.TEAL,
                    fmt="o-", ms=5, capsize=3, lw=2)
        ax.set(xlabel="Délai par conversion (ms)", ylabel=ylabel)
        ax.set_title(title, loc="left")
        ax.set_xticks([0,100,500,1000])
        ax.yaxis.set_major_formatter(FRENCH)
        ax.grid(axis="y")
    axs[0].set_ylim(-1,42)
    axs[1].axhline(-.1,color=style.RUST,ls=":",lw=1)
    style.save(fig, figures, "experience_trois_rejeu")


def trace_plot(result_dir, figures):
    frame = pd.read_csv(result_dir / "exp3_trace.csv")
    fig, axs = plt.subplots(2,1,figsize=(8,3.4),sharex=True,layout="constrained")
    axs[0].plot(frame.temps_s,frame.marge_observee_pb,color=style.TEAL,lw=1,
                label="Marge sur les cotations décalées")
    axs[0].plot(frame.temps_s,frame.marge_synchrone_pb,color=style.RUST,lw=1.5,
                label="Marge synchrone")
    axs[0].axhline(0,color=style.SLATE,lw=.7)
    axs[0].set_ylabel("Moyenne de cycle\n(pb log / conversion)")
    axs[0].legend(loc="upper right",fontsize=7.5)
    axs[0].set_title("Une trajectoire simulée — actualisation de 100 ms et retard de 500 ms",loc="left")
    axs[1].plot(frame.temps_s,frame.age_max_ms/1000,color=style.BLUE,lw=1)
    axs[1].axhline(1,color=style.RUST,lw=1,ls=":")
    axs[1].set(xlabel="Temps simulé (s)",ylabel="Âge maximal (s)")
    for ax in axs:
        ax.grid(axis="y")
        ax.yaxis.set_major_formatter(FRENCH)
    style.save(fig, figures, "experience_trois_trajectoire")


def tables(result_dir):
    totals = summary_two(result_dir)
    rows = [r"\begin{tabular}{@{}lrrrr@{}}",r"\toprule",
            r"Méthode & Rayon & Couv. des cycles & Acceptés & Rejetés\\",r"\midrule"]
    radii=pd.read_csv(result_dir/"exp2_rayons.csv").query("quantile == .99 and delai_ms == 100").set_index("methode")
    for method in LABELS:
        t=totals.loc[(.99,100,method)]
        rr=radii.loc[method]
        radius = style.decimal(rr.rayon_min_pb_conversion,3)
        if rr.rayon_max_pb_conversion != rr.rayon_min_pb_conversion:
            radius += "--" + style.decimal(rr.rayon_max_pb_conversion,3)
        short={"arcs_marginaux":"Arêtes marginales","arcs_simultanes":"Arêtes simultanées","cycles_simultanes":"Cycles simultanés"}[method]
        rows.append(" & ".join([short,radius,style.decimal(t.couverture_pct,2)+r"\,\%",
                               style.integer(t.acceptes),style.integer(t.rejetes)])+r"\\")
    rows += [r"\bottomrule",r"\end{tabular}"]
    (result_dir/"table_experience_deux.tex").write_text("\n".join(rows)+"\n")
    rows=[r"\begin{tabular}{@{}rrrrr@{}}",r"\toprule",
          r"Quantile & Rayon & Couv. & Rejetés & Rejets contredits\\",r"\midrule"]
    for q in (.90,.95,.99):
        t=totals.loc[(q,100,"cycles_simultanes")]
        radius=pd.read_csv(result_dir/"exp2_rayons.csv").query("quantile == @q and delai_ms == 100 and methode == 'cycles_simultanes'").iloc[0].rayon_min_pb_conversion
        rows.append(" & ".join([style.decimal(100*q,0)+r"\,\%",style.decimal(radius,3),
                               style.decimal(t.couverture_pct,2)+r"\,\%",style.integer(t.rejetes),
                               style.integer(t.rejets_avec_cycle_positif_rejouable)])+r"\\")
    rows += [r"\bottomrule",r"\end{tabular}"]
    (result_dir/"table_experience_deux_sensibilite.tex").write_text("\n".join(rows)+"\n")
    data=pd.read_csv(result_dir/"exp3_repliques.csv")
    selected=data.query("mise_a_jour_ms == 100 and retard_eurusd_ms == 500 and not panne")
    g=selected.groupby("delai_conversion_ms").agg(n=("rejeu_n","sum"),positive=("rejeu_positifs","sum"),mean=("gain_moyen_pb_conversion","mean"))
    rows=[r"\begin{tabular}{@{}rrrr@{}}",r"\toprule",
          r"Délai (ms) & Rejeux positifs & Part positive & Gain moyen\\",r"\midrule"]
    for delay,t in g.iterrows():
        rows.append(" & ".join([str(delay),style.integer(t.positive),
                               style.percent(t.positive,t.n,2)+r"\,\%",style.decimal(t["mean"],4)])+r"\\")
    rows += [r"\bottomrule",r"\end{tabular}"]
    (result_dir/"table_experience_trois.tex").write_text("\n".join(rows)+"\n")


def run(result_dir, figures):
    phase_plot(result_dir,figures)
    certificate_plot(result_dir,figures)
    data=pd.read_csv(result_dir/"exp3_repliques.csv")
    lag_plot(data,figures)
    execution_plot(data,figures)
    trace_plot(result_dir,figures)
    tables(result_dir)
    first=json.loads((result_dir/"exp1_validation.json").read_text())
    third=json.loads((result_dir/"exp3_validation.json").read_text())
    values={
        "ExpUnCycles":str(first["cycles_orientes_simples"]),
        "ExpUnGrille":style.integer(first["points_phase"]),
        "ExpUnPositifs":style.integer(first["points_avec_gain"]),
        "ExpUnJauge":scientific(first["erreur_periodes_sous_jauge_pb"]),
        "ExpTroisErreurJauge":scientific(third["erreur_max_jauge_temporelle_pb"]),
        "ExpTroisRepliques":str(third["repliques"]),
    }
    (result_dir/"experiences_valeurs.tex").write_text("\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key,value in values.items())+"\n")


if __name__=="__main__":
    run(Path("results"),Path("figures"))
