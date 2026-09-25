"""Grafici del modulo 2.4 (CAF GIAS): bias algoritmico e fairness.

Uso:  python genera_figure.py [cartella_output]
Genera sei PNG a 300 dpi, gia dimensionati per le slide (10 x 5,63 pollici).
Colori verificati con il validatore di accessibilita: blu #2a78d6 e oro
#b07a0a passano i controlli per chi ha difetti nella visione dei colori.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm, gaussian_kde
from sklearn.metrics import roc_curve, roc_auc_score

QUI = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else QUI
OUT.mkdir(parents=True, exist_ok=True)

BLU, ORO = "#2a78d6", "#b07a0a"          # identita dei due gruppi: oro = dove sta il problema
GRIGIO, GRIGIO_CH = "#9a9892", "#e6e5e1"  # riferimenti e verita
INK, INK2, ASSE = "#1a1a19", "#52514e", "#cfcdc6"

matplotlib.rcParams.update({
    "font.family": "Calibri", "font.size": 10.5,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 22,
    "axes.labelsize": 11, "axes.labelcolor": INK2,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.major.pad": 5, "ytick.major.pad": 5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": ASSE, "axes.linewidth": 0.9,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#edece8", "grid.linewidth": 0.9,
    "axes.axisbelow": True,
    "lines.linewidth": 2.2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 10.5,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300,
})
ALONE = [pe.withStroke(linewidth=3.5, foreground="white")]   # anello bianco dietro testi che toccano una linea


def virgola(x, dec=2):
    return f"{x:.{dec}f}".replace(".", ",")


def pct(ax, asse="both"):
    f = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    if asse in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if asse in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def sottotitolo(ax, testo):
    ax.text(0, 1.035, testo, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=INK2)


def punto(ax, x, y, colore, ms=7.5, z=5):
    ax.plot([x], [y], "o", ms=ms, color=colore, mec="white", mew=1.6, zorder=z)


def nota(fig, testo, lato="destra"):
    x, ha = (0.995, "right") if lato == "destra" else (0.02, "left")
    fig.text(x, 0.012, testo, ha=ha, va="bottom", fontsize=9, color=GRIGIO)


def legenda(fig, voci):
    maniglie = [Line2D([], [], color=c, lw=2.4, marker="o", ms=6.5, mec="white", mew=1.2) for _, c in voci]
    fig.legend(maniglie, [t for t, _ in voci], loc="upper right", bbox_to_anchor=(0.995, 0.995),
               ncol=len(voci), handlelength=1.8, columnspacing=1.6, handletextpad=0.6)


def salva(fig, nome):
    fig.savefig(OUT / f"{nome}.png")
    plt.close(fig)
    print("  scritto", nome)


sig = lambda z: 1 / (1 + np.exp(-z))
logit = lambda p: np.log(p / (1 - p))


# ---------------------------------------------------------------- F5B
def f5b():
    """Come si legge l'intercetta di calibrazione. Mezza slide."""
    fig = plt.figure(figsize=(4.5, 3.3))
    ax = fig.add_axes([0.15, 0.155, 0.8, 0.645])
    p = np.linspace(0.005, 0.995, 400)
    sotto, sopra = sig(logit(p) - 0.4), sig(logit(p) + 0.4)

    ax.fill_between(p, sotto, p, color=GRIGIO_CH, alpha=0.55, lw=0, zorder=1)
    ax.fill_between(p, p, sopra, color=GRIGIO_CH, alpha=0.55, lw=0, zorder=1)
    ax.plot(p, p, color=GRIGIO, lw=1.5, ls=(0, (4, 3)), zorder=2)
    ax.plot(p, sopra, color=INK2, zorder=3)
    ax.plot(p, sotto, color=INK2, zorder=3)

    # la lettura a 30 per cento
    y22, y39 = sig(logit(0.30) - 0.4), sig(logit(0.30) + 0.4)
    ax.plot([0.30, 0.30], [0, y39], color=INK2, lw=0.9, zorder=2)
    for y, c in ((y39, INK2), (0.30, GRIGIO), (y22, INK2)):
        punto(ax, 0.30, y, c, ms=7)
    ax.text(0.275, y39, f"vero {y39 * 100:.0f}%", ha="right", va="center", fontsize=10.5, color=INK,
            path_effects=ALONE, zorder=6)
    ax.text(0.325, y22, f"vero {y22 * 100:.0f}%", ha="left", va="center", fontsize=10.5, color=INK,
            path_effects=ALONE, zorder=6)
    ax.text(0.318, 0.035, "dice 30%", ha="left", va="bottom", fontsize=10.5, color=INK,
            fontweight="bold", path_effects=ALONE, zorder=6)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.canvas.draw()

    def etichetta(x, curva, testo, dy):                  # testo ruotato lungo la curva
        i = np.searchsorted(p, x)
        (x0, y0), (x1, y1) = ax.transData.transform([(p[i - 5], curva[i - 5]), (p[i + 5], curva[i + 5])])
        ax.text(x, curva[i] + dy, testo, rotation=np.degrees(np.arctan2(y1 - y0, x1 - x0)),
                rotation_mode="anchor", ha="center", va="center", fontsize=10.5, color=INK,
                path_effects=ALONE, zorder=6)
    etichetta(0.50, sopra, "minimizza  ·  +0,4", 0.058)
    etichetta(0.56, p, "dice il vero  ·  0", 0.0)
    etichetta(0.72, sotto, "esagera  ·  −0,4", -0.062)

    pct(ax)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Rischio dichiarato dal sistema")
    ax.set_ylabel("Rischio vero")
    ax.set_title("Come si legge l'intercetta")
    sottotitolo(ax, "Stesso rischio dichiarato, tre realtà diverse")
    salva(fig, "F5B_intercetta")


# ---------------------------------------------------------------- F5A
def dati_f5a(seed=7, n=5000):
    rng = np.random.default_rng(seed)
    out = {}
    for g, spostamento in (("A", 0.0), ("B", 0.8)):
        L = rng.normal(-1.5, 2.0, n)                   # stesso rischio vero nei due gruppi
        y = rng.random(n) < sig(L)
        out[g] = (y, sig(L + spostamento))             # B: il sistema gonfia il rischio, ordine identico
    return out


def curva_calibrazione(y, s, bins=10):
    q = np.quantile(s, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, s, side="right") - 1, 0, bins - 1)
    return (np.array([s[idx == k].mean() for k in range(bins)]),
            np.array([y[idx == k].mean() for k in range(bins)]))


def f5a():
    """Stesso ordine, numeri diversi. Slide intera."""
    d = dati_f5a()
    fig = plt.figure(figsize=(9.0, 3.2))
    axr = fig.add_axes([0.075, 0.2, 0.37, 0.55])
    axc = fig.add_axes([0.575, 0.2, 0.37, 0.55])

    axr.plot([0, 1], [0, 1], color=GRIGIO, lw=1.3, ls=(0, (4, 3)))
    auc = {}
    for g, c, lw in (("A", BLU, 3.4), ("B", ORO, 1.6)):   # B sottile sopra A: si vedono tutte e due
        y, s = d[g]
        fpr, tpr, _ = roc_curve(y, s)
        auc[g] = roc_auc_score(y, s)
        axr.plot(fpr, tpr, color=c, lw=lw)
    fig.canvas.draw()
    (x0, y0), (x1, y1) = axr.transData.transform([(0, 0), (1, 1)])
    axr.text(0.80, 0.745, "caso", rotation=np.degrees(np.arctan2(y1 - y0, x1 - x0)), ha="center",
             va="center", fontsize=10, color=GRIGIO)
    for k, (g, c) in enumerate((("A", BLU), ("B", ORO))):
        punto(axr, 0.52, 0.30 - k * 0.12, c)
        axr.text(0.56, 0.30 - k * 0.12, f"Gruppo {g}  ·  AUROC {virgola(auc[g])}", va="center",
                 fontsize=10.5, color=INK, path_effects=ALONE)
    axr.set_xlim(0, 1); axr.set_ylim(0, 1.0)
    pct(axr)
    axr.set_xticks([0, 0.5, 1]); axr.set_yticks([0, 0.5, 1])
    axr.set_xlabel("Sani segnalati per errore"); axr.set_ylabel("Malati trovati")
    axr.set_title("Mette in ordine i pazienti?")
    sottotitolo(axr, "AUROC: per i due gruppi è uguale")

    axc.plot([0, 1], [0, 1], color=GRIGIO, lw=1.3, ls=(0, (4, 3)))
    for g, c in (("A", BLU), ("B", ORO)):
        x, y = curva_calibrazione(*d[g])
        axc.plot(x, y, color=c, zorder=3)
        axc.plot(x, y, "o", ms=6, color=c, mec="white", mew=1.4, zorder=4)
    axc.text(0.05, 0.66, "A  ·  dice il vero", fontsize=10.5, color=INK, path_effects=ALONE)
    axc.text(0.60, 0.16, "B  ·  esagera", fontsize=10.5, color=INK, path_effects=ALONE)
    axc.set_xlim(0, 1); axc.set_ylim(0, 1.0)
    pct(axc)
    axc.set_xticks([0, 0.5, 1]); axc.set_yticks([0, 0.5, 1])
    axc.set_xlabel("Rischio dichiarato dal sistema"); axc.set_ylabel("Rischio vero")
    axc.set_title("Il rischio dichiarato è vero?")
    sottotitolo(axc, "Calibrazione: per B non lo è")

    nota(fig, "Dati simulati a scopo illustrativo")
    salva(fig, "F5A_stesso_ordine")
    return auc


# ---------------------------------------------------------------- F7
def dati_f7(seed=3, n=10000, quota=0.40):
    rng = np.random.default_rng(seed)
    eta = rng.uniform(20, 80, n)
    val = 100 + 0.5 * (eta - 50) + rng.normal(0, 12, n)

    def tara(score):                                   # sposta la costante finche ne manca il 40%
        a = 0.0
        for _ in range(80):
            a -= (sig(a + score).mean() - quota) * 4
        return rng.random(n) < sig(a + score)

    manca = {
        "caso": rng.random(n) < quota,
        "noto": tara(-2.0 * (eta - 50) / 17),          # mancano soprattutto i giovani
        "diverso": tara(1.5 * (val - 100) / 15),        # mancano soprattutto i valori alti
    }
    fasce = np.digitize(eta, np.linspace(20, 80, 13)[1:-1])   # media per fascia d'eta, pesata sulla popolazione
    corretta = sum((fasce == k).mean() * val[(fasce == k) & ~manca["noto"]].mean() for k in np.unique(fasce))
    return val, manca, corretta


def f7():
    """Tre modi di mancare. Slide intera."""
    val, manca, corretta = dati_f7()
    xs = np.linspace(40, 160, 400)
    kde = lambda v: gaussian_kde(v, bw_method=0.25)(xs) * len(v)
    tutti = kde(val)
    vera = val.mean()

    fig = plt.figure(figsize=(9.0, 3.2))
    titoli = [("caso", "Manca per caso", "Chi ha il dato somiglia a tutti"),
              ("noto", "Manca per un motivo noto", "Lo scostamento si corregge con l'età"),
              ("diverso", "Manca perché il paziente è diverso", "Lo scostamento non si corregge")]
    for k, (chiave, titolo, sotto) in enumerate(titoli):
        ax = fig.add_axes([0.035 + k * 0.325, 0.17, 0.285, 0.55])
        oss = val[~manca[chiave]]
        media = oss.mean()
        ax.fill_between(xs, tutti, color=GRIGIO_CH, lw=0)
        ax.plot(xs, tutti, color=GRIGIO, lw=1.2)
        ax.fill_between(xs, kde(oss), color=BLU, alpha=0.16, lw=0)
        ax.plot(xs, kde(oss), color=BLU, lw=2)
        top = tutti.max()
        ax.plot([vera, vera], [0, top * 1.06], color=GRIGIO, lw=1.3)
        ax.plot([media, media], [0, top * 1.06], color=BLU, lw=1.6)
        dist = media - vera
        if abs(dist) < 1:
            ax.text(vera, top * 1.09, "stessa media", ha="center", va="bottom", fontsize=10.5, color=INK)
        else:
            colore = ORO if chiave == "diverso" else INK2
            ax.annotate("", xy=(media, top * 1.03), xytext=(vera, top * 1.03),
                        arrowprops=dict(arrowstyle="-|>", color=colore, lw=1.6, shrinkA=0, shrinkB=0,
                                        mutation_scale=11))
            ax.text((vera + media) / 2, top * 1.09, f"{'+' if dist > 0 else '−'}{abs(dist):.0f} punti",
                    ha="center", va="bottom", fontsize=10.5, color=colore if chiave == "diverso" else INK,
                    fontweight="bold" if chiave == "diverso" else "normal")
        ax.set_xlim(45, 155); ax.set_ylim(0, top * 1.28)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
        ax.set_xticks([70, 100, 130])
        ax.set_title(titolo, fontsize=12.5)
        sottotitolo(ax, sotto)

    maniglie = [plt.Rectangle((0, 0), 1, 1, fc=GRIGIO_CH, ec=GRIGIO, lw=1),
                plt.Rectangle((0, 0), 1, 1, fc=matplotlib.colors.to_rgba(BLU, 0.16), ec=BLU, lw=1.6)]
    fig.legend(maniglie, ["tutti i pazienti", "chi ha il dato"], loc="upper right",
               bbox_to_anchor=(0.995, 0.995), ncol=2, handlelength=1.4, columnspacing=1.6)
    fig.text(0.5, 0.05, "Valore dell'esame", ha="center", va="bottom", fontsize=11, color=INK2)
    nota(fig, "Dati simulati · 10.000 pazienti · manca il 40% in ogni pannello")
    salva(fig, "F7_tre_modi_di_mancare")
    return val.mean(), {k: val[~m].mean() for k, m in manca.items()}, corretta


# ---------------------------------------------------------------- F13
F13_RIGHE = [
    ("Tutte le 1.388 colonne", 1.000),
    ("Senza i 135 argomenti legati al sesso", 0.988),
    ("Senza colonne con differenza oltre 30 punti", 0.954),
    ("Senza colonne con differenza oltre 10 punti", 0.857),
    ("Senza colonne con differenza oltre 5 punti", 0.732),
    ("Senza colonne con differenza oltre 2 punti", 0.586),
    ("Solo laboratorio ed esame fisico", 0.575),
]


def f13():
    """La scala della dimostrazione. Slide intera."""
    fig = plt.figure(figsize=(9.0, 3.2))
    ax = fig.add_axes([0.385, 0.155, 0.585, 0.64])
    ys = [7, 6, 5, 4, 3, 2, 0.6]
    ax.axvline(0.5, color=GRIGIO, lw=1.3, ls=(0, (4, 3)), zorder=1)
    ax.text(0.507, -0.45, "tirare una moneta", fontsize=10, color=GRIGIO, va="bottom")
    ax.axhline(1.3, color="#edece8", lw=1, zorder=0)
    for (testo, v), y in zip(F13_RIGHE, ys):
        chiave = "10 punti" in testo
        c = ORO if chiave else BLU
        ax.plot([0.5, v], [y, y], color="#dcdad4", lw=3.2, solid_capstyle="round", zorder=2)
        punto(ax, v, y, c, ms=10, z=4)
        ax.text(v + 0.013, y, virgola(v, 3), va="center", fontsize=10.5, color=INK,
                fontweight="bold" if chiave else "normal")
        ax.text(-0.015, y, testo, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10.8, color=INK, fontweight="bold" if chiave else "normal")
    ax.set_xlim(0.45, 1.055); ax.set_ylim(-0.55, 7.55)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.grid(False); ax.grid(True, axis="x", color="#f1f0ec", lw=0.9)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: virgola(v, 1)))
    ax.set_xlabel("Capacità di riconoscere il sesso (AUROC)")
    fig.text(0.02, 0.955, "Quanto bisogna mutilare il dato prima che il segnale sparisca",
             fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.88, "Il modello vede solo quali celle sono vuote, mai un valore clinico",
             fontsize=10.5, color=INK2, va="top")
    nota(fig, "NHANES 2013-2014 · 6.113 adulti · regressione logistica, 70/30 stratificato", lato="sinistra")
    salva(fig, "F13_quanto_mutilare")


# ---------------------------------------------------------------- F17
def f17():
    """L'impossibilita in una figura. Slide intera."""
    mu, soglia = 2.1232, 1.2816                        # sensibilita 80%, specificita 90% alla soglia dell'esempio
    t = np.linspace(-1.2, 3.6, 500)
    sens, fpr = 1 - norm.cdf(t - mu), 1 - norm.cdf(t)
    vpp = lambda prev: sens * prev / (sens * prev + fpr * (1 - prev))
    s0 = 1 - norm.cdf(soglia - mu)
    vA0, vB0 = [s0 * p / (s0 * p + (1 - norm.cdf(soglia)) * (1 - p)) for p in (0.20, 0.05)]

    fig = plt.figure(figsize=(9.0, 3.2))
    axs = fig.add_axes([0.075, 0.2, 0.37, 0.52])
    axv = fig.add_axes([0.575, 0.2, 0.37, 0.52])
    for ax in (axs, axv):
        ax.axvline(soglia, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 1.04)
        ax.set_xticks([]); pct(ax, "y"); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.text(0.0, -0.07, "← sistema più indulgente", transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color=INK2)
        ax.text(1.0, -0.07, "più severo →", transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=INK2)
        ax.text(soglia + 0.07, 0.025, "soglia dell'esempio", fontsize=9.5, color=INK2, va="bottom",
                path_effects=ALONE)

    axs.plot(t, sens, color=BLU, lw=6.2)               # due linee identiche: bordo blu, anima oro
    axs.plot(t, sens, color="white", lw=3.6)
    axs.plot(t, sens, color=ORO, lw=2.0)
    punto(axs, soglia, s0, BLU, ms=11); punto(axs, soglia, s0, ORO, ms=6, z=6)
    axs.text(soglia - 0.14, s0 - 0.06, "80% e 80%", ha="right", va="top", fontsize=10.5, color=INK,
             fontweight="bold", path_effects=ALONE)
    axs.text(2.55, 0.40, "identica nei\ndue gruppi", fontsize=10.5, color=INK, linespacing=1.1, path_effects=ALONE)
    axs.set_title("Trova i malati?")
    sottotitolo(axs, "Sensibilità: stessa per A e B, a ogni soglia")

    axv.plot(t, vpp(0.20), color=BLU)
    axv.plot(t, vpp(0.05), color=ORO)
    punto(axv, soglia, vA0, BLU, ms=8.5); punto(axv, soglia, vB0, ORO, ms=8.5)
    axv.text(soglia - 0.12, vA0, f"{vA0 * 100:.0f}%", ha="right", va="center", fontsize=10.5, color=INK,
             fontweight="bold", path_effects=ALONE)
    axv.text(soglia + 0.14, vB0 - 0.04, f"{vB0 * 100:.0f}%", ha="left", va="top", fontsize=10.5, color=INK,
             fontweight="bold", path_effects=ALONE)
    axv.text(-1.1, 0.36, "Gruppo A", fontsize=10.5, color=INK, path_effects=ALONE)
    axv.text(-1.1, 0.135, "Gruppo B", fontsize=10.5, color=INK, path_effects=ALONE)
    axv.set_title("Quando dice positivo, ha ragione?")
    sottotitolo(axv, "Valore predittivo: diverso a ogni soglia")

    legenda(fig, [("Gruppo A · malattia nel 20%", BLU), ("Gruppo B · malattia nel 5%", ORO)])
    nota(fig, "Stesso sistema per i due gruppi · cambia solo quanto è frequente la malattia")
    salva(fig, "F17_impossibilita")
    return s0, vA0, vB0


# ---------------------------------------------------------------- F21
def f21():
    """AUROC fermo, calibrazione in movimento. Dati reali del workshop, rieseguiti."""
    df = pd.read_csv(QUI / "dati" / "rerun_1c_xgboost_artrite.csv")
    ordine = ["S0_50-50", "S1_30-70", "S2_15-85", "S3_5-95"]
    etich = ["50%", "30%", "15%", "5%"]
    stat = lambda col: df.groupby("scenario")[col].agg(
        m="mean", lo=lambda s: s.quantile(0.025), hi=lambda s: s.quantile(0.975)).reindex(ordine)
    x = np.arange(4)

    fig = plt.figure(figsize=(9.0, 3.2))
    axa = fig.add_axes([0.075, 0.2, 0.36, 0.52])
    axc = fig.add_axes([0.575, 0.2, 0.36, 0.52])
    for ax, col in ((axa, "auroc"), (axc, "citl")):
        for sesso, c in (("M", BLU), ("F", ORO)):
            s = stat(f"{col}_{sesso}")
            ax.fill_between(x, s.lo, s.hi, color=c, alpha=0.14, lw=0)
            ax.plot(x, s.m, color=c)
            for xi, yi in zip(x, s.m):
                punto(ax, xi, yi, c, ms=7)
        ax.set_xticks(x); ax.set_xticklabels(etich)
        ax.set_xlim(-0.25, 3.55)
        ax.set_xlabel("Quota di donne nei dati di addestramento")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: virgola(v, 1)))

    sc = {k: stat(f"citl_{k}") for k in "FM"}
    axa.set_ylim(0.5, 1.0); axa.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axa.text(1.5, 0.9, "uomini e donne quasi sovrapposti", fontsize=10.5, color=INK, ha="center")
    axa.set_title("Mette in ordine?")
    sottotitolo(axa, "AUROC: quasi fermo")

    axc.axhline(0, color=GRIGIO, lw=1.3, ls=(0, (4, 3)), zorder=1)
    axc.set_ylim(-0.4, 1.0); axc.set_yticks([-0.4, 0, 0.4, 0.8])
    axc.text(3.14, -0.075, "dice il vero", fontsize=10, color=GRIGIO, va="center")
    axc.text(-0.2, 0.92, "↑ sottostima il rischio", fontsize=10, color=INK2, va="top")
    axc.text(-0.2, -0.33, "↓ sovrastima", fontsize=10, color=INK2, va="bottom")
    for k, nome in (("F", "Donne"), ("M", "Uomini")):
        v = sc[k].m.iloc[-1]
        axc.text(3.14, v, f"{nome} {'+' if v >= 0 else '−'}{virgola(abs(v))}", fontsize=10.5,
                 color=INK, va="center", fontweight="bold" if k == "F" else "normal")
    v0 = sc["F"].m.iloc[0]
    axc.text(0, v0 + 0.075, f"+{virgola(v0)}", fontsize=10.5, color=INK, va="bottom", ha="center",
             fontweight="bold", path_effects=ALONE)
    axc.set_title("Il rischio dichiarato è vero?")
    sottotitolo(axc, "Intercetta di calibrazione: si allontana per le donne")

    legenda(fig, [("Uomini", BLU), ("Donne", ORO)])
    nota(fig, "NHANES, XGBoost su artrite · media e intervallo al 95% su 30 ripetizioni · intercetta con pendenza fissata a 1")
    salva(fig, "F21_auroc_fermo_calibrazione_no")
    return sc


if __name__ == "__main__":
    print("Genero in", OUT)
    f5b()
    auc = f5a()
    print(f"    F5A AUROC A {auc['A']:.3f}  B {auc['B']:.3f}")
    vera, oss, corr = f7()
    print(f"    F7 media vera {vera:.1f}  osservate " + "  ".join(f"{k} {v:.1f}" for k, v in oss.items())
          + f"  corretta per eta {corr:.1f}")
    f13()
    s0, a, b = f17()
    print(f"    F17 sensibilita {s0:.3f}  VPP A {a:.3f}  B {b:.3f}")
    sc = f21()
    print("    F21 CITL donne", [round(v, 3) for v in sc["F"].m], " uomini", [round(v, 3) for v in sc["M"].m])
