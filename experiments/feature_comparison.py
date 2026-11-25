#!/usr/bin/env python3
import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from window_cache import get_feats, eer_thr, bal_acc, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--asv-root", required=True, help="folder containing the ASVspoof2019 LA protocols and flac")
ap.add_argument("--n-train", type=int, default=2000, help="per class (2000 -> 4000 files)")
ap.add_argument("--n-dev", type=int, default=1500)
ap.add_argument("--n-eval", type=int, default=4000)
a = ap.parse_args()

SETS = {"MFCC (120)": ["mfcc"], "Spectral (24)": ["spectral"], "F0 (8)": ["f0"],
        "Spectral+F0 (32)": ["spectral", "f0"], "MFCC+Spectral+F0 (152)": ["mfcc", "spectral", "f0"]}
CONDS = ["clean", "telephone", "telephone_codec"]


def X_for(split, n, cond, kinds):
    parts = [get_feats(a.asv_root, split, n, cond, k) for k in kinds]
    return np.hstack([p[0] for p in parts]), parts[0][1]


def models():
    return {"LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
            "RF": RandomForestClassifier(300, n_jobs=-1, class_weight="balanced", random_state=0),
            "SVM": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced"))}


def score(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") and not isinstance(m[-1] if hasattr(m, "steps") else m, SVC) \
        else m.decision_function(X)


rows = []
for fs, kinds in SETS.items():
    Xtr, ytr = X_for("train", a.n_train, "clean", kinds)
    Xdv, ydv = X_for("dev", a.n_dev, "clean", kinds)
    for name, m in models().items():
        m.fit(Xtr, ytr)
        _, thr = eer_thr(ydv, score(m, Xdv))              # frozen on clean dev
        for c in CONDS:
            Xe, ye = X_for("eval", a.n_eval, c, kinds)
            s = score(m, Xe)
            e, _ = eer_thr(ye, s)
            rows.append(dict(features=fs, model=name, condition=c, EER=100 * e,
                             AUC=roc_auc_score(ye, s), bal_acc_frozen=100 * bal_acc(ye, s, thr)))
        print(fs, name, "done", flush=True)

df = pd.DataFrame(rows)
out = ROOT / "outputs"; out.mkdir(exist_ok=True)
df.to_csv(out / "feature_comparison.csv", index=False)
print(df.pivot_table(index=["features", "model"], columns="condition", values="EER").round(1))
