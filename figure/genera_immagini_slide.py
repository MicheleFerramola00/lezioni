"""Immagini delle slide del modulo 2.4 (CAF GIAS), oltre ai sei grafici di genera_figure.py.

Uso:  python genera_immagini_slide.py [cartella_output]
Stesso stile e stessi colori di genera_figure.py. Dati:
  - NHANES 2013-2014 (pubblico): figure/dati/NHANES_2013_2014_master.csv, scaricato se manca
  - matrice di mancanza: dati/mancanza_nhanes_adulti.parquet
  - Seyyed-Kalantari et al., Nature Medicine 2021: CSV del repository degli autori (LalehSeyyed/Underdiagnosis_NatMed)
  - ARIES, Oberije et al., BMJ Health & Care Informatics 2025, tabella 3 (CC BY 4.0)
  - Agniel, Kohane, Weber, BMJ 2018: numeri citati testualmente dai risultati
  - Gichoya et al., preprint arXiv 2107.10356: AUC sulle immagini degradate
  - radiografia del torace di M. Haggstrom, CC0 (Wikimedia Commons)
  - NIH ChestX-ray14 (NIH Clinical Center, https://nihcc.app.box.com/v/ChestXray-NIHCC): metadati
    Data_Entry_2017_v2020.csv, scaricati se mancano; uso libero citando Wang et al., CVPR 2017
Le immagini marcate "illustrativo" usano numeri inventati per spiegare un meccanismo, e lo dichiarano.
"""
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageFilter
from scipy.stats import norm, gaussian_kde

from genera_figure import (BLU, ORO, GRIGIO, GRIGIO_CH, INK, INK2, ALONE, virgola, pct,
                           sottotitolo, punto, nota, salva as _salva, sig, logit, QUI)

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else QUI
OUT.mkdir(parents=True, exist_ok=True)
NAVY_SLIDE = "#001B5C"
ORO_SLIDE = "#D4A13A"
CXR = QUI / "dati" / "radiografia_torace_haggstrom_cc0.jpg"
REPO_URL = "https://github.com/MicheleFerramola00/lezioni/tree/caf-unilink-m2.4"


def salva(fig, nome, trasparente=False):
    fig.savefig(OUT / f"{nome}.png", transparent=trasparente)
    plt.close(fig)
    print("  scritto", nome)


def nhanes():
    f = QUI / "dati" / "NHANES_2013_2014_master.csv"
    if not f.exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/oliviariccomi/gender-bias-analysis/main/"
                                   "data/raw_dataset/NHANES_2013_2014_master.csv", f)
    d = pd.read_csv(f, low_memory=False)
    return d[d.age_years >= 18].copy()


def cxr14():
    f = QUI / "dati" / "Data_Entry_2017_v2020.csv"
    if not f.exists():
        urllib.request.urlretrieve("https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/"
                                   "data/Data_Entry_2017_v2020.csv", f)
    return pd.read_csv(f)


def titolo_fig(fig, titolo, sotto=None, y=0.955):
    fig.text(0.03, y, titolo, fontsize=13, fontweight="bold", color=INK, va="top")
    if sotto:
        fig.text(0.03, y - 0.085, sotto, fontsize=10.5, color=INK2, va="top")


def radiografia(lato=512):
    im = Image.open(CXR).convert("L")
    w, h = im.size
    q = min(w, h)
    im = im.crop(((w - q) // 2, 0, (w - q) // 2 + q, q))   # quadrato dalla parte alta: il torace
    return im.resize((lato, lato), Image.LANCZOS)


def cornice(ax, colore=GRIGIO, tratteggio=False, lw=1.2):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(colore); s.set_linewidth(lw)
        if tratteggio:
            s.set_linestyle((0, (4, 3)))


# ---------------------------------------------------------------- slide 2
def s02():
    fig = plt.figure(figsize=(4.5, 3.3))
    fig.text(0.03, 0.93, "COSA DICE IL REPORT", fontsize=9.5, fontweight="bold", color=INK2, va="top")
    fig.text(0.03, 0.78, "0,92", fontsize=40, fontweight="bold", color=NAVY_SLIDE, va="top")
    fig.text(0.03, 0.54, "AUROC su tutti\ni pazienti", fontsize=10.5, color=INK2, va="top", linespacing=1.15)
    # la scala per leggerlo: da 0,5 (una moneta) a 1 (perfetto)
    x0, x1, ys = 0.035, 0.32, 0.31
    xv = x0 + (0.92 - 0.5) / 0.5 * (x1 - x0)
    fig.add_artist(plt.Line2D([x0, x1], [ys, ys], color=GRIGIO_CH, lw=4, solid_capstyle="round"))
    fig.add_artist(plt.Line2D([x0, xv], [ys, ys], color=NAVY_SLIDE, lw=4, solid_capstyle="round"))
    fig.add_artist(plt.Line2D([xv], [ys], marker="o", ms=8, color=NAVY_SLIDE, mec="white", mew=1.5))
    for x, num, parola, ha in ((x0, "0,5", "una moneta", "left"), (x1, "1", "perfetto", "right")):
        fig.text(x, ys - 0.045, num, ha=ha, va="top", fontsize=9.5, color=INK, fontweight="bold")
        fig.text(x, ys - 0.105, parola, ha=ha, va="top", fontsize=9.5, color=INK2)
    fig.add_artist(plt.Line2D([0.36, 0.36], [0.12, 0.93], color=GRIGIO_CH, lw=1.2))
    fig.text(0.41, 0.93, "COSA SUCCEDE IN REPARTO", fontsize=9.5, fontweight="bold", color=INK2, va="top")
    fig.text(0.41, 0.84, "Diagnosi mancate ogni 100 malati", fontsize=10.5, color=INK2, va="top")
    ax = fig.add_axes([0.6, 0.14, 0.34, 0.6])
    gruppi = ["Gruppo A", "Gruppo B", "Gruppo C", "Gruppo D"]
    val = [8, 7, 9, 23]
    y = np.arange(4)[::-1]
    ax.barh(y, val, height=0.56, color=[BLU, BLU, BLU, ORO])
    for yi, v in zip(y, val):
        ax.text(v + 0.8, yi, str(v), va="center", fontsize=10.5, color=INK, fontweight="bold" if v == 23 else "normal")
    ax.set_yticks(y); ax.set_yticklabels(gruppi)
    ax.set_xlim(0, 28); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False); ax.grid(False)
    nota(fig, "Esempio illustrativo")
    salva(fig, "S02_report_e_reparto")


# ---------------------------------------------------------------- mappa delle tappe
TAPPE = [("Il dato che manca", "25 min · con demo dal vivo"), ("Quale equità", "10 min"),
         ("Tre casi nell'imaging", "30 min · con demo dal vivo"), ("Cosa si chiede in gara", "10 min")]


def mappa(attiva=None, scura=False):
    fig = plt.figure(figsize=(9.0, 1.15))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    xs = [0.11, 0.37, 0.63, 0.89]
    linea = "#6f7fa6" if scura else "#cfcdc6"
    ax.plot([xs[0], xs[-1]], [0.72, 0.72], color=linea, lw=3.2, solid_capstyle="round", zorder=1)
    for k, (x, (nome, durata)) in enumerate(zip(xs, TAPPE), start=1):
        if scura:
            if attiva == k:
                fc, ec, cn, cl, cd, peso = ORO_SLIDE, ORO_SLIDE, NAVY_SLIDE, "white", "#e9dcc0", "bold"
            elif attiva and k < attiva:
                fc, ec, cn, cl, cd, peso = "#c9d1e4", "#c9d1e4", NAVY_SLIDE, "#c9d1e4", "#9fabc7", "normal"
            else:
                fc, ec, cn, cl, cd, peso = NAVY_SLIDE, "#8a98bd", "#8a98bd", "#aab4cc", "#8a98bd", "normal"
        else:
            fc, ec, cn, cl, cd, peso = NAVY_SLIDE, NAVY_SLIDE, "white", INK, INK2, "bold"
        ax.plot([x], [0.72], "o", ms=25, mfc=fc, mec=ec, mew=2.2, zorder=2)   # marcatore: resta rotondo
        ax.text(x, 0.715, str(k), ha="center", va="center", fontsize=12, fontweight="bold", color=cn, zorder=3)
        ax.text(x, 0.40, nome, ha="center", va="center", fontsize=12, fontweight=peso, color=cl)
        ax.text(x, 0.13, durata, ha="center", va="center", fontsize=10, color=cd)
    return fig


def mappe():
    fig = mappa(); salva(fig, "S03_mappa_tappe")
    for k in range(1, 5):
        fig = mappa(attiva=k, scura=True); salva(fig, f"S_mappa_tappa{k}", trasparente=True)


# ---------------------------------------------------------------- slide 6 bis
def s06b():
    """Il gruppo sanguigno e chi lo ha misurato: numeri illustrativi, gli stessi del workshop Sanidays."""
    fasce = ["18-29 anni", "30-49 anni", "50-69 anni", "70 e oltre"]
    pannelli = (("Nella popolazione", "quota con gruppo 0", {"uomini": [45] * 4, "donne": [45] * 4}),
                ("Nel dataset", "quota con il gruppo sanguigno registrato",
                 {"uomini": [16, 37, 55, 56], "donne": [56, 69, 56, 59]}))
    fig = plt.figure(figsize=(4.5, 3.3))
    maniglie = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLU, ORO)]
    fig.legend(maniglie, ["uomini", "donne"], loc="upper right", bbox_to_anchor=(0.97, 0.995), ncol=2,
               handlelength=1.0, handleheight=0.9, columnspacing=1.2, fontsize=10)
    for k, (titolo, sotto, dati) in enumerate(pannelli):
        ax = fig.add_axes([0.05, 0.555 - k * 0.405, 0.9, 0.235])
        for j, (g, c) in enumerate((("uomini", BLU), ("donne", ORO))):
            xs = [i + (j - 0.5) * 0.34 for i in range(4)]
            ax.bar(xs, dati[g], width=0.3, color=c)
            if k == 1:
                for x, v in zip(xs, dati[g]):
                    ax.text(x, v + 3, f"{v}%", ha="center", va="bottom", fontsize=9, color=INK)
        if k == 0:
            ax.text(3.62, 47, "45% in ogni gruppo", ha="right", va="bottom", fontsize=9.5, color=INK)
        ax.set_ylim(0, 100); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
        ax.set_xlim(-0.6, 3.6); ax.set_xticks(range(4)); ax.set_xticklabels(fasce if k == 1 else [])
        ax.text(0, 1.08, titolo, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=INK)
        ax.text(1, 1.08, sotto, transform=ax.transAxes, fontsize=10, color=INK2, ha="right")
    nota(fig, "Esempio illustrativo")
    salva(fig, "S06B_gruppo_sanguigno")


# ---------------------------------------------------------------- slide 8
def s08():
    fig = plt.figure(figsize=(4.5, 3.3))
    titolo_fig(fig, "Chi arriva nel dataset", "Su 100 persone che avrebbero bisogno dell'esame")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.68]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 4.6)
    tappe = [("Hanno bisogno dell'esame", "", 100), ("Arrivano al servizio", "distanza, orari, lingua", 72),
             ("Ricevono la prescrizione", "sospetto clinico, come si raccontano", 55),
             ("Fanno l'esame", "spostarsi, capire la preparazione", 46),
             ("Entrano nel dataset", "codifica, sistema informativo", 40)]
    cx, largh = 0.74, 0.46
    for i, (nome, causa, n) in enumerate(tappe):
        y = 4 - i
        w = largh * n / 100
        ax.add_patch(Rectangle((cx - largh / 2, y - 0.3), largh, 0.6, fc=GRIGIO_CH, ec="none"))
        ax.add_patch(Rectangle((cx - w / 2, y - 0.3), w, 0.6, fc=ORO if i == 4 else BLU, ec="none"))
        ax.text(cx, y, str(n), ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(0.47, y + (0.12 if causa else 0), nome, ha="right", va="center", fontsize=10.5,
                color=INK, fontweight="bold" if i == 4 else "normal")
        if causa:
            ax.text(0.47, y - 0.2, causa, ha="right", va="center", fontsize=9, color=INK2, style="italic")
    nota(fig, "Numeri illustrativi")
    salva(fig, "S08_imbuto")


# ---------------------------------------------------------------- slide 9
def s09():
    d = nhanes()
    esami = ["bmi", "bp_systolic_1", "hba1c_pct", "cholesterol_serum_mgdl", "hdl_cholesterol_mgdl",
             "creatinine_mgdl", "albumin_gdl", "hemoglobin_gdl", "glucose_serum_mgdl"]
    completi = d[esami].notna().all(axis=1)
    fasce = pd.cut(d.age_years, [17, 39, 59, 79, 120], labels=["18-39 anni", "40-59 anni", "60-79 anni", "80 anni e oltre"])
    reddito = np.where(d.poverty_income_ratio < 1, "sotto la soglia di povertà",
                       np.where(d.poverty_income_ratio.isna(), None, "sopra la soglia di povertà"))
    righe = [(f, completi[fasce == f].mean()) for f in fasce.cat.categories]
    righe += [(r, completi[reddito == r].mean()) for r in ("sopra la soglia di povertà", "sotto la soglia di povertà")]
    fig = plt.figure(figsize=(4.5, 3.3))
    titolo_fig(fig, "Chi resta dopo il filtro", "Quota di pazienti con tutti e nove gli esami")
    ax = fig.add_axes([0.46, 0.15, 0.46, 0.6])
    ys = [6, 5, 4, 3, 1.4, 0.4]
    for (lab, v), y in zip(righe, ys):
        basso = lab in ("80 anni e oltre", "sotto la soglia di povertà")
        ax.plot([0.70, v], [y, y], color="#dcdad4", lw=3, solid_capstyle="round", zorder=1)
        punto(ax, v, y, ORO if basso else BLU, ms=9)
        ax.text(v + 0.009, y, f"{v * 100:.0f}%", va="center", fontsize=10.5, color=INK,
                fontweight="bold" if basso else "normal")
        ax.text(-0.03, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10.2,
                color=INK, fontweight="bold" if basso else "normal")
    ax.axhline(2.2, color="#edece8", lw=1)
    ax.set_xlim(0.70, 0.92); ax.set_ylim(-0.2, 6.6)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.grid(False); ax.grid(True, axis="x", color="#f1f0ec")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_xticks([0.70, 0.80, 0.90])
    nota(fig, "NHANES 2013-2014 · 6.113 adulti · esami previsti dal protocollo, uguali per tutti", lato="sinistra")
    salva(fig, "S09_casi_completi")
    return righe


# ---------------------------------------------------------------- slide 10
def s10(seed=11):
    rng = np.random.default_rng(seed)
    magg = rng.normal(100, 15, 8000)
    mino = rng.normal(118, 15, 2000)
    manca = rng.random(2000) < sig(-0.9 + 2.0 * (mino - 118) / 15)        # nel gruppo piccolo mancano i valori alti
    pool = np.concatenate([magg, mino[~manca]])                              # chi il dato ce l'ha
    riempito = mino.copy(); riempito[manca] = rng.choice(pool, manca.sum())
    xs = np.linspace(40, 180, 400)
    kd = lambda v: gaussian_kde(v, bw_method=0.3)(xs)
    fig = plt.figure(figsize=(4.5, 3.3))
    titolo_fig(fig, "Cosa fa l'imputazione a un gruppo", "Riempio i buchi con i valori di chi il dato ce l'ha")
    ax = fig.add_axes([0.05, 0.17, 0.9, 0.57])
    ax.fill_between(xs, kd(mino), color=GRIGIO_CH, lw=0)
    ax.plot(xs, kd(mino), color=GRIGIO, lw=1.6)
    ax.plot(xs, kd(riempito), color=ORO, lw=2.4)
    top = kd(mino).max()
    for v, c, t, ha in ((mino.mean(), GRIGIO, "vero", "left"), (riempito.mean(), ORO, "dopo", "right")):
        ax.plot([v, v], [0, top * 1.12], color=c, lw=1.4)
        ax.text(v + (1.5 if ha == "left" else -1.5), top * 1.14, f"{t} {v:.0f}", ha=ha, va="bottom",
                fontsize=10.5, color=INK, fontweight="bold" if t == "dopo" else "normal")
    m = magg.mean()
    ax.plot([m], [0], marker="^", ms=9, color=BLU, clip_on=False, zorder=5)
    ax.text(m, top * 0.07, "media del gruppo\npiù numeroso", ha="center", va="bottom", fontsize=9.5, color=INK2,
            linespacing=1.05, path_effects=ALONE)
    ax.text(150, top * 0.72, "valori veri\ndel gruppo", fontsize=10, color=INK2, linespacing=1.05)
    ax.text(58, top * 0.8, "dopo aver\nriempito i buchi", fontsize=10, color=INK, linespacing=1.05)
    ax.set_xlim(45, 175); ax.set_ylim(0, top * 1.35)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
    ax.set_xlabel("Valore dell'esame nel gruppo meno numeroso")
    nota(fig, "Dati simulati")
    salva(fig, "S10_imputazione")
    return mino.mean(), riempito.mean(), manca.mean()


# ---------------------------------------------------------------- slide 11
def s11():
    """Agniel, Kohane, Weber, BMJ 2018. Numeri citati testualmente:
    'more accurate than the test results in predicting survival in 118 of 174 tests (68%)';
    'The time interval between consecutive tests is the single most predictive variable for 76 of 210 (36%)
    tests, followed by the value of the test result in 56 (27%) tests, and the hour of the day in 47 (22%) tests.'"""
    fig = plt.figure(figsize=(4.5, 3.3))
    fig.text(0.03, 0.955, "68%", fontsize=34, fontweight="bold", color=ORO, va="top")
    fig.text(0.27, 0.93, "degli esami (118 su 174): il momento della\nrichiesta predice la sopravvivenza meglio\n"
                         "del risultato dell'esame", fontsize=10.5, color=INK, va="top", linespacing=1.15)
    fig.text(0.03, 0.62, "La variabile che da sola predice meglio, esame per esame", fontsize=10.5,
             fontweight="bold", color=INK, va="top")
    ax = fig.add_axes([0.42, 0.13, 0.5, 0.4])
    righe = [("Intervallo fra due esami", 76, ORO), ("Valore del risultato", 56, INK2),
             ("Ora del giorno", 47, ORO), ("Altre variabili", 31, "#cfcdc6")]
    y = np.arange(4)[::-1]
    ax.barh(y, [r[1] for r in righe], height=0.58, color=[r[2] for r in righe])
    for yi, (lab, n, _) in zip(y, righe):
        ax.text(n + 2, yi, f"{n}", va="center", fontsize=10.5, color=INK)
        ax.text(-0.03, yi, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10, color=INK)
    ax.set_yticks([]); ax.set_xticks([]); ax.set_xlim(0, 95)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(False)
    fig.text(0.42, 0.075, "numero di esami su 210", fontsize=9.5, color=INK2)
    nota(fig, "Agniel, Kohane, Weber · BMJ 2018")
    salva(fig, "S11_agniel")


# ---------------------------------------------------------------- slide 12
def s12():
    im = radiografia(700)
    fig = plt.figure(figsize=(4.5, 3.3))
    a1 = fig.add_axes([0.05, 0.14, 0.42, 0.66]); a2 = fig.add_axes([0.53, 0.14, 0.42, 0.66])
    a1.imshow(im, cmap="gray", aspect="auto"); cornice(a1, INK2)
    a2.set_facecolor("#f7f6f3"); cornice(a2, GRIGIO, tratteggio=True, lw=1.4)
    a2.text(0.5, 0.52, "nessun esame", ha="center", va="center", fontsize=12, color=INK2, transform=a2.transAxes)
    a2.text(0.5, 0.40, "nessun file, nessuna riga", ha="center", va="center", fontsize=9.5, color=GRIGIO,
            transform=a2.transAxes)
    fig.text(0.05, 0.90, "Nel dataset", fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.53, 0.90, "Fuori dal dataset", fontsize=12, fontweight="bold", color=INK, va="top")
    nota(fig, "Radiografia: M. Häggström, pubblico dominio (CC0)")
    salva(fig, "S12_paziente_mancante")


# ---------------------------------------------------------------- slide 13
def s13():
    M = pd.read_parquet(QUI.parent / "dati" / "mancanza_nhanes_adulti.parquet")
    y = M.pop("sesso_F").values
    M = M.loc[:, (M.mean() > 0.05) & (M.mean() < 0.95)]
    rng = np.random.default_rng(4)
    diff = M[y == 1].mean() - M[y == 0].mean()
    col = list(diff.sort_values().index[:30]) + list(rng.choice(diff.abs().sort_values().index[:600], 45, replace=False)) \
        + list(diff.sort_values().index[-30:])
    righe = np.concatenate([rng.choice(np.where(y == 1)[0], 40, replace=False), rng.choice(np.where(y == 0)[0], 40, replace=False)])
    X = M.iloc[righe][col].values.astype(float)
    fig = plt.figure(figsize=(4.3, 2.3))
    ax = fig.add_axes([0.12, 0.08, 0.86, 0.72])
    ax.imshow(X, cmap=matplotlib.colors.ListedColormap(["#eef0f4", "#2f3b4c"]), aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(39.5, color="white", lw=2)
    ax.text(-1.5, 19.5, "donne", rotation=90, ha="right", va="center", fontsize=10, color=INK2)
    ax.text(-1.5, 59.5, "uomini", rotation=90, ha="right", va="center", fontsize=10, color=INK2)
    fig.text(0.12, 0.965, "Il file della dimostrazione: solo buchi", fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.12, 0.87, "80 pazienti, 105 colonne · scuro = il dato manca", fontsize=9.8, color=INK2, va="top")
    salva(fig, "S13_matrice_buchi")


def qr():
    import qrcode
    q = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=20, border=2)
    q.add_data(REPO_URL); q.make(fit=True)
    q.make_image(fill_color=NAVY_SLIDE, back_color="white").save(OUT / "QR_repo.png")
    print("  scritto QR_repo")


# ---------------------------------------------------------------- slide 16
def s16():
    fig = plt.figure(figsize=(4.4, 3.3))
    titolo_fig(fig, "Tre richieste, tutte ragionevoli", "Ognuna chiede che i due gruppi siano uguali su una misura diversa")
    voci = [("Trova i malati", "sensibilità"), ("Positivo affidabile", "valore predittivo"),
            ("Rischio veritiero", "calibrazione")]
    for k, (t, s) in enumerate(voci):
        ax = fig.add_axes([0.04 + k * 0.325, 0.12, 0.27, 0.5])
        ax.bar([0, 1], [0.5, 0.5], width=0.42, color=[BLU, ORO])
        ax.plot([-0.35, 1.35], [0.58, 0.58], color=INK2, lw=1.1, ls=(0, (4, 3)))
        ax.text(0.5, 0.64, "=", ha="center", fontsize=16, fontweight="bold", color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["A", "B"])
        ax.set_yticks([]); ax.set_ylim(0, 1.02); ax.set_xlim(-0.55, 1.55)
        ax.spines["left"].set_visible(False); ax.grid(False)
        fig.text(0.04 + k * 0.325, 0.73, t, fontsize=11, fontweight="bold", color=INK, va="top")
        fig.text(0.04 + k * 0.325, 0.665, s, fontsize=10, color=INK2, va="top")
    salva(fig, "S16_tre_equita")


# ---------------------------------------------------------------- slide 20
def s20(seed=5):
    rng = np.random.default_rng(seed)
    xa = rng.uniform(0, 100, 9000); xb = rng.uniform(0, 100, 1000)
    ya = rng.random(9000) < sig((xa - 50) / 9)
    yb = rng.random(1000) < sig((xb - 30) / 14)
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([xa, xb]).reshape(-1, 1); Y = np.concatenate([ya, yb])
    m = LogisticRegression(C=1e6).fit(X, Y)
    xs = np.linspace(0, 100, 300)
    fig = plt.figure(figsize=(4.5, 3.3))
    titolo_fig(fig, "Una regola sola per due gruppi", "La regola imparata somiglia al gruppo più numeroso")
    ax = fig.add_axes([0.13, 0.16, 0.82, 0.58])
    ax.plot(xs, sig((xs - 50) / 9), color=BLU, lw=2.2)
    ax.plot(xs, sig((xs - 30) / 14), color=ORO, lw=2.2)
    ax.plot(xs, m.predict_proba(xs.reshape(-1, 1))[:, 1], color=INK, lw=3.2, ls=(0, (1, 0)), alpha=0.9)
    ax.text(2, 0.58, "gruppo piccolo\n(1 su 10)", fontsize=10, color=INK, linespacing=1.05, path_effects=ALONE)
    ax.annotate("gruppo grande (9 su 10)", xy=(57, sig(7 / 9)), xytext=(66, 0.5), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax.annotate("regola del sistema", xy=(53, float(m.predict_proba([[53]])[0, 1])), xytext=(66, 0.32),
                fontsize=10.5, color=INK, fontweight="bold", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.02)
    pct(ax, "y"); ax.set_yticks([0, 0.5, 1]); ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Valore clinico"); ax.set_ylabel("Probabilità di malattia")
    nota(fig, "Dati simulati")
    salva(fig, "S20_regola_unica")


# ---------------------------------------------------------------- slide 22
SK = [  # Seyyed-Kalantari et al. 2021, MIMIC-CXR: FPR della classe 'No Finding' e semiampiezza dell'intervallo
    ("Sesso", "Uomini", 0.219, 0.007, False), ("Sesso", "Donne", 0.250, 0.007, True),
    ("Etnia", "Bianchi", 0.170, 0.006, False), ("Etnia", "Neri", 0.276, 0.006, True),
    ("Etnia", "Ispanici", 0.275, 0.012, True),
    ("Assicurazione", "Privata o altra", 0.208, 0.005, False), ("Assicurazione", "Medicaid, basso reddito", 0.274, 0.011, True),
    ("Intersezione", "Uomini ispanici", 0.206, 0.010, False), ("Intersezione", "Donne ispaniche", 0.356, 0.016, True),
]


def s22():
    fig = plt.figure(figsize=(4.5, 2.8))
    titolo_fig(fig, "Sottodiagnosi per sottogruppo", "Pazienti con un reperto a cui il sistema dice «nessun reperto»",
               y=0.965)
    ax = fig.add_axes([0.46, 0.13, 0.46, 0.66])
    y, ys = 0, []
    prec = None
    for grp, lab, v, ci, serv in SK:
        if prec and grp != prec:
            y -= 0.55
        ys.append(y); prec = grp; y -= 1
    for (grp, lab, v, ci, serv), yi in zip(SK, ys):
        c = ORO if serv else BLU
        ax.plot([v - ci, v + ci], [yi, yi], color=c, lw=1.6)
        punto(ax, v, yi, c, ms=7)
        ax.text(v + ci + 0.006, yi, f"{v * 100:.0f}%", va="center", fontsize=9.5, color=INK,
                fontweight="bold" if lab == "Donne ispaniche" else "normal")
        ax.text(-0.03, yi, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.5,
                color=INK, fontweight="bold" if lab == "Donne ispaniche" else "normal")
    ax.set_xlim(0.14, 0.40); ax.set_ylim(ys[-1] - 0.7, 0.7)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.grid(False); ax.grid(True, axis="x", color="#f1f0ec")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_xticks([0.2, 0.3, 0.4])
    nota(fig, "Seyyed-Kalantari et al. · Nature Medicine 2021 · MIMIC-CXR", lato="sinistra")
    salva(fig, "S22_sottodiagnosi")


# ---------------------------------------------------------------- slide 23
def s23():
    d = nhanes()
    b = d[["bmi", "gender"]].dropna()
    xs = np.linspace(15, 50, 300)
    kd = lambda v: gaussian_kde(v, bw_method=0.35)(xs)
    fig = plt.figure(figsize=(4.5, 3.3))
    maniglie = [plt.Line2D([], [], color=c, lw=2.4) for c in (BLU, ORO)]
    fig.legend(maniglie, ["uomini", "donne"], loc="upper right", bbox_to_anchor=(0.97, 0.995), ncol=2,
               handlelength=1.4, columnspacing=1.2, fontsize=10)
    for k, (titolo, sub) in enumerate((("Nella popolazione", None), ("Nel dataset raccolto", "2.000 donne · 2.000 uomini"))):
        ax = fig.add_axes([0.05, 0.58 - k * 0.4, 0.9, 0.26])
        for g, c, lab in ((1, BLU, "uomini"), (2, ORO, "donne")):
            v = b.bmi[b.gender == g].values
            if k == 1:
                v = v[v <= 28] if g == 2 else v[v > 28]
            dens = kd(v)
            ax.fill_between(xs, dens, color=c, alpha=0.14, lw=0)
            ax.plot(xs, dens, color=c, lw=2)
        if k == 1:
            ax.axvline(28, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.text(19.5, ax.get_ylim()[1] * 0.78, "donne", fontsize=10, color=INK, ha="right")
            ax.text(35.5, ax.get_ylim()[1] * 0.78, "uomini", fontsize=10, color=INK)
        else:
            ax.text(38, ax.get_ylim()[1] * 0.55, "uomini e donne\nsi sovrappongono", fontsize=10, color=INK, linespacing=1.05)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
        ax.set_xlim(15, 50); ax.set_xticks([20, 25, 30, 35, 40, 45])
        if k == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Indice di massa corporea")
        ax.text(0, 1.06, titolo, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=INK)
        if sub:
            ax.text(1, 1.06, sub, transform=ax.transAxes, fontsize=10, color=INK2, ha="right")
    nota(fig, "NHANES 2013-2014 · adulti", lato="sinistra")
    salva(fig, "S23_due_centri")


# ---------------------------------------------------------------- slide 24
def s24():
    base = radiografia(700)
    port = Image.fromarray(np.clip(np.asarray(base, float) * 0.82 + 30
                                   + np.random.default_rng(2).normal(0, 9, (700, 700)), 0, 255).astype("uint8"))
    fig = plt.figure(figsize=(4.5, 3.3))
    a1 = fig.add_axes([0.05, 0.14, 0.42, 0.66]); a2 = fig.add_axes([0.53, 0.14, 0.42, 0.66])
    a1.imshow(port, cmap="gray", aspect="auto", vmin=0, vmax=255); cornice(a1, INK2)
    a2.imshow(base, cmap="gray", aspect="auto", vmin=0, vmax=255); cornice(a2, INK2)
    a1.text(0.07, 0.84, "PORTABLE  AP", transform=a1.transAxes, fontsize=9, color="white", fontweight="bold",
            family="Consolas", va="top")
    a1.add_patch(FancyBboxPatch((0.045, 0.765), 0.6, 0.1, transform=a1.transAxes, boxstyle="round,pad=0.01",
                                fc="none", ec=ORO, lw=2.2))
    a1.annotate("il modello vede\nquesto, chi referta no", xy=(0.4, 0.765), xytext=(0.42, 0.5),
                xycoords="axes fraction", textcoords="axes fraction", fontsize=9, color="white",
                fontweight="bold", ha="center", arrowprops=dict(arrowstyle="-|>", color=ORO, lw=1.6),
                path_effects=[pe.withStroke(linewidth=3, foreground="#1a1a19")])
    fig.text(0.05, 0.90, "Al letto del paziente", fontsize=11.5, fontweight="bold", color=INK, va="top")
    fig.text(0.53, 0.90, "In radiologia", fontsize=11.5, fontweight="bold", color=INK, va="top")
    nota(fig, "Ricostruzione illustrativa su una radiografia di pubblico dominio (M. Häggström, CC0)")
    salva(fig, "S24_marcatore")


# ---------------------------------------------------------------- slide 24 bis
def s24b():
    """Come è stata fatta la lastra: quota di radiografie fatte a letto (AP) per reperto, NIH ChestX-ray14."""
    d = cxr14()
    ap = d["View Position"].eq("AP")
    voci = [("Edema", "edema"), ("Consolidation", "consolidamento"), ("Pneumonia", "polmonite"),
            ("Effusion", "versamento pleurico")]
    righe = [(it, ap[d["Finding Labels"].str.contains(en)].mean()) for en, it in voci]
    righe.append(("nessun reperto", ap[d["Finding Labels"].eq("No Finding")].mean()))
    fig = plt.figure(figsize=(4.45, 2.38))
    fig.text(0.03, 0.965, "Come è stata fatta la lastra", fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.865, "Quota di radiografie fatte a letto (AP), per reperto", fontsize=9.8, color=INK2, va="top")
    ax = fig.add_axes([0.36, 0.13, 0.55, 0.62])
    ys = list(range(len(righe)))[::-1]
    for (nome, v), y in zip(righe, ys):
        colore = ORO if nome == "edema" else (GRIGIO if nome == "nessun reperto" else BLU)
        ax.barh(y, 1, height=0.56, color=GRIGIO_CH)
        ax.barh(y, v, height=0.56, color=colore)
        ax.text(v + 0.02, y, f"{v * 100:.0f}%", va="center", fontsize=10, color=INK,
                fontweight="bold" if nome == "edema" else "normal")
        ax.text(-0.03, y, nome, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10,
                color=INK, fontweight="bold" if nome == "edema" else "normal")
    ax.set_xlim(0, 1.12); ax.set_ylim(-0.6, len(righe) - 0.4)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    nota(fig, "NIH ChestX-ray14 · 112.120 radiografie · reperti estratti dai referti")
    salva(fig, "S24B_lastra_a_letto")
    return righe


# ---------------------------------------------------------------- slide 25
def s25():
    base = radiografia(512)
    rng = np.random.default_rng(1)
    rum = Image.fromarray(np.clip(np.asarray(base, float) + rng.normal(0, 38, (512, 512)), 0, 255).astype("uint8"))
    sfo = base.filter(ImageFilter.GaussianBlur(9))
    px4 = base.resize((4, 4), Image.BOX).resize((512, 512), Image.NEAREST)
    pannelli = [(base, "Originale", "oltre 0,95"), (rum, "Con rumore", "0,74–0,80"),
                (sfo, "Sfocata", "0,64–0,72"), (px4, "4 × 4 pixel", "sopra il caso")]
    fig = plt.figure(figsize=(4.5, 2.8))
    titolo_fig(fig, "Il modello riconosce l'etnia anche così", "AUROC dei modelli sulle immagini degradate", y=0.965)
    for k, (im, t, v) in enumerate(pannelli):
        ax = fig.add_axes([0.03 + k * 0.245, 0.24, 0.21, 0.46])
        ax.imshow(im, cmap="gray", aspect="auto", vmin=0, vmax=255); cornice(ax, INK2, lw=0.8)
        fig.text(0.03 + k * 0.245 + 0.105, 0.75, t, ha="center", fontsize=9.8, fontweight="bold", color=INK)
        fig.text(0.03 + k * 0.245 + 0.105, 0.15, v, ha="center", fontsize=10.5, color=INK,
                 fontweight="bold")
    nota(fig, "Gichoya et al., preprint arXiv 2107.10356 · radiografia: M. Häggström, CC0", lato="sinistra")
    salva(fig, "S25_degradazione")


# ---------------------------------------------------------------- slide 27
def s27():
    mu = 2.1232
    t = np.linspace(-1.2, 3.6, 600)
    sens, fpr = 1 - norm.cdf(t - mu), 1 - norm.cdf(t)
    vpp = lambda prev: sens * prev / (sens * prev + fpr * (1 - prev))
    obiettivo = 0.5
    ta = t[np.argmax(vpp(0.20) >= obiettivo)]; tb = t[np.argmax(vpp(0.05) >= obiettivo)]
    sa, sb = 1 - norm.cdf(ta - mu), 1 - norm.cdf(tb - mu)
    fig = plt.figure(figsize=(4.5, 2.8))
    titolo_fig(fig, "Una soglia per ogni popolazione",
               "Obiettivo: almeno un positivo su due davvero malato", y=0.965)
    ax = fig.add_axes([0.11, 0.2, 0.84, 0.52])
    ax.axhline(obiettivo, color=INK2, lw=1.1, ls=(0, (4, 3)))
    ax.plot(t, vpp(0.20), color=BLU); ax.plot(t, vpp(0.05), color=ORO)
    for tt, c, lab, s, xy, ha in ((ta, BLU, "tarata su A", sa, (ta - 0.1, 0.72), "right"),
                                  (tb, ORO, "tarata su B", sb, (tb + 0.1, 0.2), "left")):
        ax.plot([tt, tt], [0, obiettivo], color=c, lw=1.4)
        punto(ax, tt, obiettivo, c, ms=8)
        ax.text(*xy, f"{lab}\ntrova il {s * 100:.0f}% dei malati", fontsize=9.5, color=INK, ha=ha,
                linespacing=1.05, path_effects=ALONE)
    ax.text(-1.15, 0.53, "obiettivo", fontsize=9.5, color=INK2)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 1.03)
    pct(ax, "y"); ax.set_yticks([0, 0.5, 1]); ax.set_xticks([])
    maniglie = [plt.Line2D([], [], color=c, lw=2.4) for c in (BLU, ORO)]
    fig.legend(maniglie, ["popolazione A · malattia nel 20%", "popolazione B · 5%"], loc="upper right",
               bbox_to_anchor=(0.995, 0.8), ncol=2, handlelength=1.4, columnspacing=1.0, fontsize=9.3)
    ax.text(0, -0.07, "← soglia più indulgente", transform=ax.transAxes, fontsize=9.5, color=INK2, va="top")
    ax.text(1, -0.07, "più severa →", transform=ax.transAxes, fontsize=9.5, color=INK2, va="top", ha="right")
    salva(fig, "S27_taratura")
    return sa, sb


# ---------------------------------------------------------------- slide 30
ARIES = [  # Oberije et al., BMJ Health Care Inform 2025, tabella 3: tumori trovati ogni 1000 (IC 95%)
    ("Età", "Meno di 60 anni", (6.2, 5.8, 6.6), (6.1, 5.7, 6.5)),
    ("Età", "60 anni e oltre", (10.2, 9.7, 10.7), (10.0, 9.5, 10.6)),
    ("Densità", "Seno meno denso", (7.6, 7.2, 8.0), (7.5, 7.1, 7.9)),
    ("Densità", "Seno più denso", (8.9, 8.4, 9.4), (8.8, 8.3, 9.3)),
    ("Etnia", "Bianca", (9.9, 9.3, 10.5), (9.6, 9.0, 10.3)),
    ("Etnia", "Non bianca", (8.0, 7.4, 8.7), (7.9, 7.3, 8.6)),
    ("Centro", "Londra (RFL)", (8.9, 8.5, 9.3), (8.7, 8.3, 9.1)),
    ("Centro", "Nord-est (NED)", (8.9, 8.2, 9.6), (8.8, 8.1, 9.6)),
    ("Centro", "Gateshead", (6.6, 6.0, 7.4), (6.5, 5.9, 7.3)),
]


def s30():
    fig = plt.figure(figsize=(4.5, 2.8))
    titolo_fig(fig, "Tumori trovati ogni 1.000 donne", None, y=0.965)
    ax = fig.add_axes([0.37, 0.12, 0.58, 0.66])
    y, ys, prec = 0, [], None
    for grp, *_ in ARIES:
        if prec and grp != prec:
            y -= 0.5
        ys.append(y); prec = grp; y -= 1
    for (grp, lab, umani, ia), yi in zip(ARIES, ys):
        for (v, lo, hi), c, dy in ((umani, GRIGIO, 0.17), (ia, BLU, -0.17)):
            ax.plot([lo, hi], [yi + dy, yi + dy], color=c, lw=1.5)
            punto(ax, v, yi + dy, c, ms=6)
        ax.text(-0.03, yi, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9.3, color=INK)
    ax.set_xlim(5.4, 11.2); ax.set_ylim(ys[-1] - 0.7, 0.7)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.grid(False); ax.grid(True, axis="x", color="#f1f0ec")
    ax.set_xticks([6, 8, 10])
    maniglie = [plt.Line2D([], [], color=c, marker="o", ms=6, lw=1.5, mec="white") for c in (GRIGIO, BLU)]
    fig.legend(maniglie, ["doppia lettura dei radiologi", "con l'IA"], loc="upper left",
               bbox_to_anchor=(0.02, 0.885), ncol=2, handlelength=1.4, columnspacing=1.0, fontsize=9.5)
    nota(fig, "ARIES · Oberije et al., BMJ Health Care Inform 2025 · 306.839 mammografie", lato="sinistra")
    salva(fig, "S30_aries")


# ---------------------------------------------------------------- slide 31
def s31(seed=8):
    rng = np.random.default_rng(seed)
    mesi = np.arange(1, 25)
    a = 0.86 + rng.normal(0, 0.012, 24)
    b = np.where(mesi <= 12, 0.85, 0.85 - 0.03 * np.minimum(mesi - 12, 6)) + rng.normal(0, 0.014, 24)
    tutti = 0.85 * a + 0.15 * b
    fig = plt.figure(figsize=(4.5, 2.8))
    titolo_fig(fig, "Il sistema era equo. Poi", "Sensibilità mese per mese: la media non vede il gruppo B", y=0.965)
    ax = fig.add_axes([0.12, 0.17, 0.68, 0.56])
    ax.axhline(0.75, color=INK2, lw=1.1, ls=(0, (4, 3)))
    ax.axvline(12.5, color=GRIGIO, lw=1.2)
    ax.text(12.8, 0.955, "nuovo apparecchio", fontsize=9.5, color=INK2, va="top")
    ax.text(0.8, 0.735, "soglia di allarme", fontsize=9.5, color=INK2, va="top")
    ax.plot(mesi, tutti, color=GRIGIO, lw=2.6)
    ax.plot(mesi, a, color=BLU, lw=1.8)
    ax.plot(mesi, b, color=ORO, lw=2.2)
    for serie, lab, dy in ((tutti, "media", -0.012), (a, "gruppo A", 0.014), (b, "gruppo B", 0.0)):
        ax.text(24.6, serie[-1] + dy, lab, fontsize=9.5, color=INK, va="center")
    ax.set_xlim(1, 24); ax.set_ylim(0.62, 0.96)
    ax.set_xticks([1, 6, 12, 18, 24]); ax.set_xlabel("Mese")
    pct(ax, "y"); ax.set_yticks([0.7, 0.8, 0.9])
    nota(fig, "Esempio illustrativo")
    salva(fig, "S31_deriva")


if __name__ == "__main__":
    print("Genero in", OUT)
    s02(); mappe(); s06b(); s08()
    for lab, v in s09():
        print(f"    S09 {lab}: {v:.3f}")
    print("    S10 media vera {:.1f} dopo {:.1f} mancanti {:.0%}".format(*s10()))
    s11(); s12(); s13(); qr(); s16(); s20(); s22(); s23(); s24(); s24b(); s25()
    print("    S27 sensibilita tarata su A {:.2f}, su B {:.2f}".format(*s27()))
    s30(); s31()
