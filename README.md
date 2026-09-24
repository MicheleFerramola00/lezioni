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
| niente | 1.388 | 1.000 |
| le 120 colonne strutturalmente sesso-specifiche | 1.268 | 0.998 |
| tutti i domini sesso-specifici (riproduttivo, HPV, sessuale, prostata, gravidanza) | 1.253 | 0.988 |
| ogni colonna con divario di mancanza oltre 30 punti | 1.254 | 0.954 |
| ogni colonna con divario oltre 10 punti | 1.221 | 0.857 |
| ogni colonna con divario oltre 5 punti | 1.125 | 0.732 |
| ogni colonna con divario oltre 2 punti | 970 | 0.586 |
| tutto tranne laboratorio ed esame fisico | 110 | 0.575 |

Valori ottenuti con `random_state=42` e ripartizione 70/30 stratificata.

## Riferimenti della lezione

- Agniel D, Kohane IS, Weber GM. *Biases in electronic health record data due to processes within the healthcare system*. BMJ 2018.
- Seyyed-Kalantari L et al. *Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations*. Nature Medicine 2021.
- Zech JR et al. *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs*. PLOS Medicine 2018.
- Gichoya JW et al. *AI recognition of patient race in medical imaging: a modelling study*. Lancet Digital Health 2022.
- Larrazabal AJ et al. *Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis*. PNAS 2020.
- *FUTURE-AI: international consensus guideline for trustworthy and deployable artificial intelligence in healthcare*. BMJ 2025.
- *Tackling algorithmic bias and promoting transparency in health datasets: the STANDING Together consensus recommendations*. Lancet Digital Health 2024.
- Regolamento (UE) 2024/1689 (AI Act), art. 10.
