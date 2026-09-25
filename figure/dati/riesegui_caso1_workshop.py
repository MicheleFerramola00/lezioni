# Uso: python riesegui_caso1_workshop.py   (da questa cartella; circa 2 minuti, serve xgboost)
#
# Riesegue il caso 1 del workshop (notebook Fase_3_4 del repository pubblico
# oliviariccomi/gender-bias-analysis) aggiungendo l'intercetta di calibrazione
# standard (pendenza fissata a 1) accanto a quella del notebook (pendenza
# libera), piu la pendenza. Produce i due CSV usati da genera_figure.py per F21.
import json, os, time, urllib.request
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

BASE = "https://raw.githubusercontent.com/oliviariccomi/gender-bias-analysis/main/"
for nome, url in (("Fase_3_4_Modelli_Performance_Gap.ipynb", BASE + "Fase_3_4_Modelli_Performance_Gap.ipynb"),
                  ("NHANES_2013_2014_master.csv", BASE + "data/raw_dataset/NHANES_2013_2014_master.csv")):
    if not os.path.exists(nome):
        print("scarico", nome); urllib.request.urlretrieve(url, nome)
nb = json.load(open("Fase_3_4_Modelli_Performance_Gap.ipynb", encoding="utf-8"))
g = {"display": print}

# celle di preparazione e definizione del notebook (niente installazioni, niente grafici)
for i in (5, 6, 8, 11, 15, 17, 19, 22, 24, 31, 33, 34, 47):
    exec(compile("".join(nb["cells"][i]["source"]), f"<cella {i}>", "exec"), g)

from sklearn.linear_model import LogisticRegression


def cal_metrics(y, p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); y = np.asarray(y, float)
    lp = np.log(p / (1 - p))
    free = LogisticRegression(C=1e10, solver="lbfgs").fit(lp.reshape(-1, 1), y)   # come il notebook
    a = 0.0                                   # standard: logit(y) = a + 1*logit(p), Newton sulla sola costante
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp)))
        step = (y - q).sum() / (q * (1 - q)).sum()
        a += step
        if abs(step) < 1e-10:
            break
    return {"int_free": float(free.intercept_[0]), "slope": float(free.coef_[0][0]),
            "citl": float(a), "oe": float(y.mean() / p.mean())}


def run(fit_fn, target, features, n_boot=30, base=1000):
    rec = []
    for si, (sname, cfg) in enumerate(g["SCENARIOS"].items()):
        pool_b = g["pool_train"].dropna(subset=[target])
        for b in range(n_boot):
            rng_b = np.random.default_rng(g["SEED"] + base * b + si)   # deterministico, al posto di hash()
            train_b = g["make_scenario"](pool_b, cfg["n_F"], cfg["n_M"], rng_b)
            out = fit_fn(train_b, g["test_fixed"], features, target)
            res, proba, y, grp = out[0], out[1], out[2], out[3]
            r = {"scenario": sname, "boot": b, "auroc_F": res["auroc_F"], "auroc_M": res["auroc_M"],
                 "cal_int_notebook_F": res["cal_int_F"], "cal_int_notebook_M": res["cal_int_M"]}
            for gcode, lab in ((2, "F"), (1, "M")):
                m = grp == gcode
                for k, v in cal_metrics(y[m], proba[m]).items():
                    r[f"{k}_{lab}"] = v
            rec.append(r)
        print(f"  {sname}: {n_boot} fit", flush=True)
    return pd.DataFrame(rec)


t = time.time()
run(g["fit_eval_1c"], g["TARGET_1C"], g["features_1c"], base=2000).to_csv("rerun_1c_xgboost_artrite.csv", index=False)
run(g["fit_eval_1b"], g["TARGET_1B"], g["features_1b"], base=1000).to_csv("rerun_1b_logreg_diabete.csv", index=False)
print(f"fatto in {time.time() - t:.0f}s")
