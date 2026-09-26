# CAF GIAS, Universita degli Studi Link, modulo 2.4

**Bias algoritmico e fairness: gestione dei missing data e casi studio di fairness nell'imaging diagnostico**

Michele Ferramola | 1,5 ore | Modulo 2, Evidence-Based Medicine e Valutazione Metodologica

---

## Dimostrazione dal vivo

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/mancanza_e_fairness.ipynb)

`notebook/mancanza_e_fairness.ipynb` | durata in aula: **5 minuti**

Si colloca nel blocco *Gestione dei missing data* e dimostra una cosa sola:

> Il pattern di cio che manca riconosce il paziente, anche quando nessun valore clinico viene usato.

Il percorso della dimostrazione:

1. Si carica una matrice che contiene **solo** `True` e `False`, cioe dove il dato manca. Nessun esito clinico.
2. Si allena una regressione logistica su quella matrice e le si chiede di indovinare il sesso. AUROC 1.000.
3. Si accoglie l'obiezione ovvia (gravidanza, prostata, HPV) e si rimuovono tutti i domini sesso-specifici. AUROC 0.988.
4. Si alza il filtro fino ad eliminare ogni colonna il cui tasso di mancanza differisce fra i sessi piu del 10 per cento. AUROC 0.857.
5. Si verifica dove vive il segnale: nel laboratorio quasi non c'e (AUROC 0.575). Vive nel questionario, cioe nel **processo**, non nello strumento.

Non serve alcuna configurazione: il notebook scarica il dato da questo branch ed e autosufficiente.

## Dati

`dati/mancanza_nhanes_adulti.parquet` | 1,8 MB | 6.113 righe, 1.810 colonne

Derivato da **NHANES 2013-2014**, indagine pubblica statunitense su salute e nutrizione, limitato agli adulti di eta pari o superiore a 18 anni.

Il file contiene **esclusivamente** la maschera di mancanza (`True` se il valore e assente) piu la colonna `sesso_F` usata come bersaglio. Nessun valore clinico, nessun identificativo. Questo non e solo una cautela: e parte dell'argomento della lezione.

Il dato grezzo di partenza proviene dal repository pubblico [oliviariccomi/gender-bias-analysis](https://github.com/oliviariccomi/gender-bias-analysis), a sua volta costruito sui file pubblici NHANES dei CDC.

## Risultati attesi

| Cosa viene rimosso | Colonne usate | AUROC |
|---|---|---|
| niente | 1.388 | **1.000** |
| tutti i domini sesso-specifici (riproduttivo, HPV, sessuale, prostata, gravidanza): 135 colonne | 1.253 | **0.988** |
| ogni colonna con divario di mancanza oltre 30 punti | 1.254 | 0.954 |
| ogni colonna con divario oltre 10 punti | 1.221 | **0.857** |
| ogni colonna con divario oltre 5 punti | 1.125 | 0.732 |
| ogni colonna con divario oltre 2 punti | 970 | 0.586 |
| tutto tranne laboratorio ed esame fisico | 70 | **0.575** |

Valori ottenuti con `random_state=42` e ripartizione 70/30 stratificata.

## Grafici della lezione

`figure/` contiene i sei grafici inseriti nelle slide, in PNG a 300 dpi gia dimensionati per il formato 16:9 del template, e il codice che li genera:

```
cd figure
python genera_figure.py
```

| File | Slide | Cosa mostra |
|---|---|---|
| `F5B_intercetta.png` | 5 | Come si legge l'intercetta di calibrazione |
| `F5A_stesso_ordine.png` | 5 bis | Stesso AUROC, calibrazione diversa |
| `F7_tre_modi_di_mancare.png` | 7 | MCAR, MAR, MNAR e cosa succede alla media |
| `F13_quanto_mutilare.png` | 13 bis | La scala della dimostrazione dal vivo |
| `F17_impossibilita.png` | 17 bis | Stessa sensibilita, valore predittivo diverso a ogni soglia |
| `F21_auroc_fermo_calibrazione_no.png` | 21 | AUROC fermo, calibrazione in movimento (dati reali) |

Colori: blu `#2a78d6` e oro `#b07a0a`, verificati per la leggibilita da parte di chi ha difetti nella visione dei colori. L'oro indica sempre il gruppo trattato peggio.

### F21 e l'intercetta di calibrazione

F21 usa i dati di `figure/dati/`, prodotti rieseguendo il caso 1 del workshop con `figure/dati/riesegui_caso1_workshop.py` (circa due minuti; scarica da solo notebook e dati). La riesecuzione e deterministica e riproduce i CSV al bit.

Il notebook del workshop calcola l'intercetta di calibrazione con la pendenza libera. I CSV riportano entrambe le versioni: `cal_int_notebook_*` (come nel workshop) e `citl_*` (definizione standard, pendenza fissata a 1), piu la pendenza `slope_*`. F21 usa `citl_*`.

### Immagini delle slide

Le altre immagini del mazzo (prefisso `S`, piu `QR_repo.png`) si rigenerano con:

```
cd figure
python genera_immagini_slide.py
```

Il file NHANES del workshop (`figure/dati/NHANES_2013_2014_master.csv`) non e nel repository: lo script lo scarica da solo se manca. Le immagini `illustrative` e `simulate` lo dichiarano in basso; le altre usano dati reali o i numeri pubblicati negli articoli citati.

| File | Slide | Cosa mostra | Fonte |
|---|---|---|---|
| `S02_report_e_reparto.png` | 2 | Il report dice 0,92, il reparto vede un gruppo | illustrativo |
| `S03_mappa_tappe.png`, `S_mappa_tappa1..4.png` | 3 e divisori | Le quattro tappe della lezione | |
| `S08_imbuto.png` | 8 | Chi arriva nel dataset | illustrativo |
| `S09_casi_completi.png` | 9 | Chi resta dopo il filtro dei casi completi | NHANES 2013-2014 |
| `S10_imputazione.png` | 10 | Cosa fa l'imputazione a un gruppo | simulato |
| `S11_agniel.png` | 11 | Il momento della richiesta predice meglio del risultato | Agniel et al. 2018 |
| `S12_paziente_mancante.png` | 12 | In imaging manca un paziente intero | radiografia CC0 |
| `S13_matrice_buchi.png`, `QR_repo.png` | 13 | Il file della dimostrazione: solo buchi | NHANES 2013-2014 |
| `S16_tre_equita.png` | 16 | Tre richieste di equita | |
| `S20_regola_unica.png` | 20 | Una regola sola per due gruppi | simulato |
| `S22_sottodiagnosi.png` | 22 | Sottodiagnosi per sottogruppo | Seyyed-Kalantari et al. 2021 |
| `S23_due_centri.png` | 23 | Stesso numero di persone, due popolazioni | NHANES 2013-2014 |
| `S24_marcatore.png` | 24 | Il marcatore del portatile | ricostruzione su radiografia CC0 |
| `S25_degradazione.png` | 25 | Etnia riconosciuta su immagini degradate | Gichoya et al., preprint |
| `S27_taratura.png` | 27 | Una soglia per ogni popolazione | curve teoriche |
| `S30_aries.png` | 30 | Tumori trovati ogni 1.000 donne, radiologi e IA | ARIES 2025 |
| `S31_deriva.png` | 31 | La deriva dopo il nuovo apparecchio | illustrativo |

La radiografia `figure/dati/radiografia_torace_haggstrom_cc0.jpg` e di Mikael Häggström, in pubblico dominio (CC0), da Wikimedia Commons.

## Riferimenti della lezione

- Agniel D, Kohane IS, Weber GM. *Biases in electronic health record data due to processes within the healthcare system: retrospective observational study*. BMJ 2018;361:k1479.
- Seyyed-Kalantari L et al. *Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations*. Nature Medicine 2021.
- Zech JR et al. *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs*. PLOS Medicine 2018.
- Gichoya JW et al. *AI recognition of patient race in medical imaging: a modelling study*. Lancet Digital Health 2022. Preprint: arXiv 2107.10356.
- Oberije C et al. *Assessing artificial intelligence in breast screening with stratified results on 306 839 mammograms across geographic regions, age, breast density and ethnicity: the ARIES study*. BMJ Health & Care Informatics 2025;32:e101318.
- Larrazabal AJ et al. *Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis*. PNAS 2020.
- *FUTURE-AI: international consensus guideline for trustworthy and deployable artificial intelligence in healthcare*. BMJ 2025.
- *Tackling algorithmic bias and promoting transparency in health datasets: the STANDING Together consensus recommendations*. Lancet Digital Health 2024.
- Regolamento (UE) 2024/1689 (AI Act), art. 10.
