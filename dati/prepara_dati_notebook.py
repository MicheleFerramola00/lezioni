# Prepara i due file ridotti letti dai notebook, così i notebook non scaricano nulla da siti esterni.
# Uso: python dati/prepara_dati_notebook.py   (dalla radice del repo; legge i file già scaricati in figure/dati)
from pathlib import Path

import pandas as pd

QUI = Path(__file__).parent
SORGENTI = QUI.parent / "figure" / "dati"

# NIH ChestX-ray14, solo metadati: nessuna immagine
d = pd.read_csv(SORGENTI / "Data_Entry_2017_v2020.csv")
d.columns = ["immagine", "reperti", "controllo_n", "paziente", "eta", "sesso", "proiezione",
             "larghezza", "altezza", "pixel_x", "pixel_y"]
r = pd.DataFrame({"paziente": d.paziente, "eta": d.eta, "donna": (d.sesso == "F").astype("int8"),
                  "a_letto": (d.proiezione == "AP").astype("int8"),
                  "larghezza": d.larghezza, "altezza": d.altezza, "pixel_x": d.pixel_x, "pixel_y": d.pixel_y,
                  "nessun_reperto": (d.reperti == "No Finding").astype("int8")})
for en, it in (("Effusion", "versamento"), ("Pneumonia", "polmonite"), ("Consolidation", "consolidamento"),
               ("Edema", "edema")):
    r[it] = d.reperti.str.contains(en).astype("int8")
r.to_parquet(QUI / "radiografie_nih_metadati.parquet", index=False)
print("radiografie:", r.shape)

# NHANES 2013-2014, adulti dai 20 anni: artrite dichiarata, sesso, età, BMI
n = pd.read_csv(SORGENTI / "NHANES_2013_2014_master.csv", low_memory=False)
n = n[(n.age_years >= 20) & n.mc_arthritis_ever.isin([1, 2])].dropna(subset=["bmi"])
a = pd.DataFrame({"eta": n.age_years.astype(int), "bmi": n.bmi, "donna": (n.gender == 2).astype("int8"),
                  "artrite": (n.mc_arthritis_ever == 1).astype("int8")})
a.to_parquet(QUI / "artrite_nhanes_adulti.parquet", index=False)
print("artrite:", a.shape)
