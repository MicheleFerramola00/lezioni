# CAF GIAS, Universita degli Studi Link, modulo 2.4

**Bias algoritmico e fairness: gestione dei missing data e casi studio di fairness nell'imaging diagnostico**

Michele Ferramola | 1,5 ore | Modulo 2, Evidence-Based Medicine e Valutazione Metodologica

---

## Notebook

Cinque notebook, un esempio per tipo. Girano in Colab senza configurazione e leggono i dati solo da questo branch: in aula non si scarica nulla da siti esterni. Ogni passo riporta il numero atteso.

### Dimostrazioni dal vivo

| Notebook | Slide | Il messaggio |
|---|---|---|
| [Demo LIVE 1 · Un file che contiene solo dati mancanti](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/Demo_LIVE_1_un_file_di_soli_dati_mancanti.ipynb) | «Un file che contiene solo dati mancanti» | La sola mappa di quali dati mancano riconosce il sesso del paziente: AUROC 1,000, e 0,857 anche togliendo ogni colonna con una differenza oltre 10 punti. |
| [Demo LIVE 2 · Senza guardare i polmoni](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/Demo_LIVE_2_senza_guardare_i_polmoni.ipynb) | «Senza guardare i polmoni» | La sola proiezione della lastra trova l'edema con AUROC 0,756, senza guardare l'immagine. |

Circa 4 minuti ciascuna, una cella alla volta.

### Approfondimenti, da rifare a casa

| Notebook | Slide | Il messaggio |
|---|---|---|
| [Approfondimento 1 · I tre approcci ai dati mancanti](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/Approfondimento_1_tre_approcci_ai_dati_mancanti.ipynb) | Approcci 1, 2 e 3 | Ogni approccio regge sotto un'ipotesi precisa su perché il dato manca. |
| [Approfondimento 2 · Stessa soglia, malattia più o meno frequente](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/Approfondimento_2_stessa_soglia_prevalenze_diverse.ipynb) | «Non possono valere tutte e tre insieme», «Stesso modello, stessa soglia, due risultati» | Recall e specificity uguali, precision 67% contro 30%. |
| [Approfondimento 3 · Poche donne nei dati](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/Approfondimento_3_poche_donne_nei_dati.ipynb) | «Prima prova: poche donne nei dati» | L'AUROC resta fermo, il rischio delle donne viene sottostimato. |

## Dati

`dati/mancanza_nhanes_adulti.parquet` | 1,8 MB | 6.113 righe, 1.810 colonne

Derivato da **NHANES 2013-2014**, indagine pubblica statunitense su salute e nutrizione, limitato agli adulti di eta pari o superiore a 18 anni.

Il file contiene **esclusivamente** la maschera di mancanza (`True` se il valore e assente) piu la colonna `sesso_F` usata come bersaglio. Nessun valore clinico, nessun identificativo. Questo non e solo una cautela: e parte dell'argomento della lezione.

`dati/valori_nhanes_adulti.parquet` | 6.113 righe, 7 colonne

Gli stessi adulti, nello stesso ordine, con pochi valori clinici per la parte 2 della dimostrazione: eta, sesso, pressione sistolica (media delle letture), BMI, glicemia, HbA1c, diabete dichiarato. NHANES e di pubblico dominio.

`dati/artrite_nhanes_adulti.parquet` | 5.507 righe, 4 colonne

Adulti NHANES dai 20 anni con artrite dichiarata, sesso, età e BMI, per l'approfondimento 3.

`dati/radiografie_nih_metadati.parquet` | 0,7 MB | 112.120 righe

**NIH ChestX-ray14** (NIH Clinical Center, https://nihcc.app.box.com/v/ChestXray-NIHCC), solo metadati, nessuna immagine: paziente, età, sesso, proiezione (a letto o in piedi), dimensioni e risoluzione dell'immagine, cinque reperti. Ridotto dal file `Data_Entry_2017_v2020.csv`. Uso libero citando Wang X. et al., *ChestX-ray8*, IEEE CVPR 2017.

I due file ridotti si rigenerano con `python dati/prepara_dati_notebook.py`.

Il dato grezzo di partenza dei due file NHANES proviene dal repository pubblico [oliviariccomi/gender-bias-analysis](https://github.com/oliviariccomi/gender-bias-analysis), a sua volta costruito sui file pubblici NHANES dei CDC.

## Risultati attesi

### Demo LIVE 1

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

### Approfondimento 1

Errore sulla media della pressione sistolica (vera: 122.7 mmHg su 5.712 adulti), in mmHg:

| Come manca | Quanti mancano | Casi completi | Imputazione con eta e sesso |
|---|---|---|---|
| MCAR, per caso | 40% | +0.1 | +0.1 |
| MAR, ai giovani | 42% | +2.6 | -0.2 |
| MNAR, a chi ha la pressione alta | 37% | -4.8 | -4.0 |

Riconoscere il diabete con un modello allenato nell'ospedale A (AUROC):

| Il modello | Ospedale A | Ospedale B |
|---|---|---|
| usa la mancanza della glicemia come informazione | 0.906 | 0.816 |
| stima la glicemia mancante da eta e BMI | 0.849 | 0.858 |

Nell'ospedale A la glicemia si chiede a chi e sospetto di diabete, nel B a tutti sopra i 50 anni. Tutto con `numpy.random.default_rng(0)`.

### Demo LIVE 2

Verifica su 33.082 radiografie di 9.242 pazienti mai visti in addestramento (691 con edema), `GroupShuffleSplit(test_size=0.3, random_state=0)`:

| Chi | Casi di edema | AUROC della sola proiezione |
|---|---|---|
| tutti | 691 | 0.756 |
| donne | 351 | 0.77 |
| uomini | 340 | 0.74 |
| sotto i 30 anni | 113 | 0.75 |
| 30-49 anni | 238 | 0.79 |
| 50-69 anni | 306 | 0.74 |
| 70 anni e oltre | 34 | 0.69 |

Con proiezione, eta, sesso e apparecchio (`HistGradientBoostingClassifier`, `random_state=0`): 0.787. Precision della sola proiezione: 4.8%, con l'edema nel 2.1% delle radiografie.

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
| `S06B_gruppo_sanguigno.png` | 6 bis | Il gruppo sanguigno e chi lo ha misurato | illustrativo |
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
| `S24B_lastra_a_letto.png` | 24 bis | Radiografie fatte a letto, per reperto | NIH ChestX-ray14 |
| `S24C_per_gruppo.png` | 24 ter | La sola proiezione, gruppo per gruppo | NIH ChestX-ray14 |
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
- Wang X et al. *ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases*. IEEE CVPR 2017.
- Larrazabal AJ et al. *Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis*. PNAS 2020.
- *FUTURE-AI: international consensus guideline for trustworthy and deployable artificial intelligence in healthcare*. BMJ 2025.
- *Tackling algorithmic bias and promoting transparency in health datasets: the STANDING Together consensus recommendations*. Lancet Digital Health 2024.
- Regolamento (UE) 2024/1689 (AI Act), art. 10.
