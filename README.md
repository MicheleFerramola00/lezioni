# CAF GIAS, Università degli Studi Link, modulo 2.4

**Bias algoritmico e fairness: gestione dei missing data e casi studio di fairness nell'imaging diagnostico**

Michele Ferramola | 1,5 ore | Modulo 2, Evidence-Based Medicine e Valutazione Metodologica

---

## Slide

Le slide della lezione: [in PDF](slide/M2.4_Ferramola_slide.pdf), da leggere nel browser, oppure [in PowerPoint](slide/M2.4_Ferramola_slide.pptx), con le note del presentatore (circa 8 MB).

## Notebook

Cinque notebook: due dimostrazioni fatte in aula e tre approfondimenti da rifare a casa. Si aprono in Colab con un clic, senza installare nulla, e i dati sono già in questo repository. Ogni passo riporta il risultato atteso.

### Dimostrazioni dal vivo

| Notebook | Slide | Il messaggio |
|---|---|---|
| [Demo LIVE 1 · Un file che contiene solo dati mancanti](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/Demo_LIVE_1_un_file_di_soli_dati_mancanti.ipynb) | «Un file che contiene solo dati mancanti» | La sola mappa dei dati mancanti riconosce il sesso del paziente: AUROC 1,000. Resta 0,857 anche togliendo ogni colonna in cui il dato manca a un sesso più spesso che all'altro, oltre 10 punti percentuali. |
| [Demo LIVE 2 · Senza guardare i polmoni](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/Demo_LIVE_2_senza_guardare_i_polmoni.ipynb) | «Senza guardare i polmoni» | La sola proiezione della lastra, a letto o in piedi, trova l'edema con AUROC 0,756, senza guardare l'immagine. |

### Approfondimenti, da rifare a casa

| Notebook | Slide | Il messaggio |
|---|---|---|
| [Approfondimento 1 · I tre approcci ai dati mancanti](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/Approfondimento_1_tre_approcci_ai_dati_mancanti.ipynb) | Approcci 1, 2 e 3 | Ogni approccio regge solo sotto un'ipotesi precisa sul perché il dato manca. |
| [Approfondimento 2 · Stessa soglia, malattia più o meno frequente](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/Approfondimento_2_stessa_soglia_prevalenze_diverse.ipynb) | «Non possono valere tutte e tre insieme» | Recall e specificity uguali, precision 67% contro 30%. |
| [Approfondimento 3 · Poche donne nei dati](https://colab.research.google.com/github/MicheleFerramola00/lezioni/blob/caf-unilink-m2.4/notebook/Approfondimento_3_poche_donne_nei_dati.ipynb) | «Il modello minimizza l'errore medio», «Prima prova: poche donne nei dati» | L'AUROC resta fermo, il rischio delle donne viene sottostimato. |

## Dati

I notebook leggono quattro file, tutti in `dati/`: dati pubblici e anonimizzati.

`dati/mancanza_nhanes_adulti.parquet` | 1,8 MB | 6.113 righe, 1.810 colonne

Da **NHANES 2013-2014**, l'indagine pubblica statunitense su salute e nutrizione, solo adulti dai 18 anni. Contiene solo la mappa dei dati mancanti (`True` dove il valore manca) e la colonna `sesso_F`, che il modello deve riconoscere: nessun valore clinico, nessun identificativo. Lo usa la Demo LIVE 1.

`dati/valori_nhanes_adulti.parquet` | 6.113 righe, 7 colonne

Gli stessi adulti, nello stesso ordine, con pochi valori clinici: età, sesso, pressione sistolica (media delle letture), BMI, glicemia, HbA1c, diabete dichiarato. Lo usa l'Approfondimento 1.

`dati/artrite_nhanes_adulti.parquet` | 5.507 righe, 4 colonne

Adulti NHANES dai 20 anni: artrite dichiarata, sesso, età e BMI. Lo usa l'Approfondimento 3.

`dati/radiografie_nih_metadati.parquet` | 0,7 MB | 112.120 righe

**NIH ChestX-ray14** (NIH Clinical Center, https://nihcc.app.box.com/v/ChestXray-NIHCC), solo metadati, nessuna immagine: paziente, età, sesso, proiezione (a letto o in piedi), dimensioni e risoluzione dell'immagine, cinque reperti. Ridotto dal file `Data_Entry_2017_v2020.csv`. Uso libero citando Wang X. et al., *ChestX-ray8*, IEEE CVPR 2017. Lo usa la Demo LIVE 2.

I file NHANES vengono dal repository pubblico [oliviariccomi/gender-bias-analysis](https://github.com/oliviariccomi/gender-bias-analysis), costruito sui file pubblici NHANES dei CDC. I file delle radiografie e dell'artrite si rigenerano con `python dati/prepara_dati_notebook.py`.

## Risultati attesi delle dimostrazioni

Negli approfondimenti il risultato atteso è scritto nel notebook, a ogni passo.

### Demo LIVE 1

| Cosa viene tolto | Colonne usate | AUROC |
|---|---|---|
| niente | 1.388 | **1,000** |
| gli argomenti legati al sesso (riproduttivo, HPV, sessuale, prostata, gravidanza): 135 colonne | 1.253 | **0,988** |
| ogni colonna con una differenza di mancanza fra i sessi oltre 30 punti | 1.254 | 0,954 |
| ogni colonna con una differenza oltre 10 punti | 1.221 | **0,857** |
| ogni colonna con una differenza oltre 5 punti | 1.125 | 0,732 |
| ogni colonna con una differenza oltre 2 punti | 970 | 0,586 |

Valori ottenuti con `random_state=42` e ripartizione 70/30 stratificata.

### Demo LIVE 2

Verifica su 33.082 radiografie di 9.242 pazienti mai visti in addestramento (691 con edema), `GroupShuffleSplit(test_size=0.3, random_state=0)`:

| Chi | Casi di edema | AUROC della sola proiezione |
|---|---|---|
| tutti | 691 | 0,756 |
| donne | 351 | 0,77 |
| uomini | 340 | 0,74 |
| sotto i 30 anni | 113 | 0,75 |
| 30-49 anni | 238 | 0,79 |
| 50-69 anni | 306 | 0,74 |
| 70 anni e oltre | 34 | 0,69 |

Con proiezione, età, sesso e apparecchio (`HistGradientBoostingClassifier`, `random_state=0`): 0,787. Precision della sola proiezione: 4,8%, con l'edema nel 2,1% delle radiografie.

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
