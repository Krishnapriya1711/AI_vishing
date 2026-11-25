#!/usr/bin/env python3
"""Train / tune / evaluate on the cached features from features/extract_features.py.

Protocol (matches the README):
    train  -> fit models
    dev    -> early stopping (CNN) and the decision threshold (EER point), then FROZEN
    eval   -> touched once per model / condition / window / seed
    VISHGUARD -> external evaluation only (1 s windows), window level + call level

Two training regimes:
    clean : train on clean audio only            (measures the telephone gap)
    aug   : train on clean + telephone + codec   (measures what augmentation recovers)

Models: Logistic Regression, Random Forest, SVM (RBF) on MFCC statistics, and the
LightweightCNN on log-Mel. Primary metrics are EER and AUC; accuracy / balanced
accuracy / F1 use the frozen dev threshold (ASVspoof is ~90% spoof, so plain
accuracy is not meaningful on its own).

Multiple seeds: the seed changes CNN initialisation / batch order and Random Forest
randomness. Logistic Regression and SVM are deterministic and run once. The training
subset is the same for every seed. 95% CIs are bootstrap intervals over eval samples
(the caches hold one centre crop per file, so one window = one utterance).

At 1 s, the first seed's models and frozen dev thresholds are saved to
outputs/models/ so other scripts (e.g. DEEP-VOICE scoring) can reuse them.

Usage:
    python experiments/run_final_experiments.py
    python experiments/run_final_experiments.py --windows 1.0 --regimes clean \
        --models logistic_regression --skip-cnn --seeds 42 --n-boot 50
"""

import argparse
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
MODELS = OUT / "models"

CONDITIONS = ["clean", "telephone", "telephone_codec"]
REGIME_CONDS = {"clean": ["clean"], "aug": ["clean", "telephone", "telephone_codec"],
                "aug_tel": ["clean", "telephone"], "aug_codec": ["clean", "telephone_codec"]}
VG_WINDOW = 1.0                                   # VISHGUARD windows are 1 s
DETERMINISTIC = {"logistic_regression", "svm"}    # seed has no effect; run once


# ============================================================
# METRICS  (label 1 = spoof, score high = more likely spoof)
# ============================================================

def eer_and_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    eer, _ = eer_and_threshold(y, s)
    return {
        "n": int(len(y)),
        "eer": eer,
        "auc": float(roc_auc_score(y, s)),
        "acc": float((pred == y).mean()),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def bootstrap_ci(y, s, n_boot, seed):
    """95% bootstrap CI for EER and AUC, resampling eval samples with replacement."""
    nan = {"eer_lo": np.nan, "eer_hi": np.nan, "auc_lo": np.nan, "auc_hi": np.nan}
    if n_boot <= 0:
        return nan
    rng = np.random.default_rng(seed)
    n, eers, aucs = len(y), [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yb = y[i]
        if yb.min() == yb.max():
            continue
        eers.append(eer_and_threshold(yb, s[i])[0])
        aucs.append(roc_auc_score(yb, s[i]))
    if not eers:
        return nan
    return {"eer_lo": float(np.percentile(eers, 2.5)),
            "eer_hi": float(np.percentile(eers, 97.5)),
            "auc_lo": float(np.percentile(aucs, 2.5)),
            "auc_hi": float(np.percentile(aucs, 97.5))}


# ============================================================
# DATA
# ============================================================

def load_asv(split, cond, window):
    p = PROC / "asvspoof" / f"{split}_{cond}_w{window:g}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {"mfcc": d["mfcc"], "mel": d["mel"], "y": d["y"],
            "system_id": d["system_id"]}


def concat(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load_vg(cond):
    p = PROC / "vishguard" / f"{cond}_w1.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


# ============================================================
# MODELS
# ============================================================

class SkModel:
    def __init__(self, kind, seed):
        self.kind = kind
        if kind == "logistic_regression":
            self.m = LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=seed)
        elif kind == "random_forest":
            self.m = RandomForestClassifier(n_estimators=300, max_depth=20,
                                            class_weight="balanced", n_jobs=-1,
                                            random_state=seed)
        elif kind == "svm":
            self.m = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
        else:
            raise ValueError(kind)

    def fit(self, train, dev=None):
        self.sc = StandardScaler().fit(train["mfcc"])
        self.m.fit(self.sc.transform(train["mfcc"]), train["y"])
        return self

    def score(self, d):
        X = self.sc.transform(d["mfcc"])
        if self.kind == "random_forest":
            return self.m.predict_proba(X)[:, 1]
        return self.m.decision_function(X)


class LightweightCNN(nn.Module):
    """Same architecture as src/models/cnn.py."""

    def __init__(self, num_classes=2):
        super().__init__()

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*block(1, 32), *block(32, 64),
                                      *block(64, 128), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 64),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3),
                                        nn.Linear(64, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class CnnModel:
    def __init__(self, seed, epochs=30, patience=7, batch=64, lr=1e-3):
        self.seed, self.epochs, self.patience = seed, epochs, patience
        self.batch, self.lr = batch, lr
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _x(self, mel, idx=None):
        t = torch.from_numpy(mel if idx is None else mel[idx])
        return t.float().unsqueeze(1).to(self.dev)

    def fit(self, train, dev):
        torch.manual_seed(self.seed)
        self.net = LightweightCNN().to(self.dev)
        y = train["y"]
        counts = np.bincount(y, minlength=2).astype(float)
        w = torch.tensor(len(y) / (2 * np.maximum(counts, 1)),
                         dtype=torch.float32, device=self.dev)
        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        yt = torch.from_numpy(y).long().to(self.dev)
        yd = torch.from_numpy(dev["y"]).long().to(self.dev)

        best, best_state, bad = float("inf"), None, 0
        rng = np.random.default_rng(self.seed)
        for ep in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(y))
            for s in range(0, len(y), self.batch):
                idx = order[s:s + self.batch]
                if len(idx) < 2:
                    continue                       # BatchNorm needs >1 sample
                opt.zero_grad()
                loss = crit(self.net(self._x(train["mel"], idx)), yt[idx])
                loss.backward()
                opt.step()
            vloss = self._loss(dev, yd, crit)
            sched.step(vloss)
            if vloss < best:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _loss(self, d, yt, crit):
        self.net.eval()
        tot = 0.0
        for s in range(0, len(yt), 256):
            idx = np.arange(s, min(s + 256, len(yt)))
            tot += crit(self.net(self._x(d["mel"], idx)), yt[idx]).item() * len(idx)
        return tot / len(yt)

    @torch.no_grad()
    def score(self, d):
        self.net.eval()
        out = []
        for s in range(0, len(d["y"]), 256):
            idx = np.arange(s, min(s + 256, len(d["y"])))
            logits = self.net(self._x(d["mel"], idx))
            out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())  # logit margin
        return np.concatenate(out)


def build(kind, seed, epochs):
    return CnnModel(seed, epochs=epochs) if kind == "cnn" else SkModel(kind, seed)


def save_frozen(kind, model, thr, tag):
    """Save a trained model + its frozen dev threshold for reuse by other scripts."""
    MODELS.mkdir(parents=True, exist_ok=True)
    if kind == "cnn":
        torch.save({"kind": kind, "thr": thr, "state": model.net.state_dict()},
                   MODELS / f"{tag}.pt")
    else:
        joblib.dump({"kind": kind, "thr": thr, "sc": model.sc, "m": model.m},
                    MODELS / f"{tag}.joblib", compress=3)


# ============================================================
# EXPERIMENT
# ============================================================

def per_attack_eer(d, s):
    rows = {}
    bona = d["y"] == 0
    for a in sorted(set(d["system_id"]) - {"-"}):
        m = bona | (d["system_id"] == a)
        if len(set(d["y"][m])) == 2:
            rows[a] = eer_and_threshold(d["y"][m], s[m])[0]
    return rows


def run(args):
    global OUT, MODELS
    if args.out_suffix:
        OUT = OUT / args.out_suffix
        MODELS = OUT / "models"
    OUT.mkdir(exist_ok=True, parents=True)
    rows, attack_rows, vg_rows = [], [], []
    models = args.models + ([] if args.skip_cnn else ["cnn"])
    first_seed = args.seeds[0]
    vgs = {c: load_vg(c) for c in CONDITIONS}

    for window in args.windows:
        evs = {c: load_asv("eval", c, window) for c in CONDITIONS}
        for regime in args.regimes:
            conds = REGIME_CONDS[regime]
            train = concat([load_asv("train", c, window) for c in conds])
            dev = concat([load_asv("dev", c, window) for c in conds])
            if train is None or dev is None:
                print(f"[w{window:g}/{regime}] missing train/dev cache -- skipped")
                continue
            print(f"\n=== window {window:g}s | regime {regime} | "
                  f"train {len(train['y'])} dev {len(dev['y'])} ===", flush=True)

            for seed in args.seeds:
                for kind in models:
                    if kind in DETERMINISTIC and seed != first_seed:
                        continue
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    model = build(kind, seed, args.epochs).fit(train, dev)

                    dev_eer, thr = eer_and_threshold(dev["y"], model.score(dev))  # FROZEN
                    print(f"  seed {seed} {kind:20s} dev_eer={dev_eer:.4f}", flush=True)
                    is_vg_window = abs(window - VG_WINDOW) < 1e-9
                    if is_vg_window and kind == "cnn" and args.save_all_seeds:
                        save_frozen(kind, model, thr, f"{regime}_w1_{kind}_s{seed}")
                    elif is_vg_window and seed == first_seed:
                        save_frozen(kind, model, thr, f"{regime}_w1_{kind}")

                    for cond, ev in evs.items():
                        if ev is None:
                            continue
                        s = model.score(ev)                                       # eval: once
                        r = metrics_at(ev["y"], s, thr)
                        r.update(bootstrap_ci(ev["y"], s, args.n_boot, seed))
                        rows.append({"regime": regime, "window": window, "model": kind,
                                     "seed": seed, "eval_condition": cond,
                                     "dev_eer": dev_eer, "threshold": thr, **r})
                        print(f"    eval/{cond:16s} eer={r['eer']:.4f} auc={r['auc']:.4f} "
                              f"[{r['eer_lo']:.3f},{r['eer_hi']:.3f}] "
                              f"bal_acc={r['bal_acc']:.4f}", flush=True)
                        if cond == "clean":
                            for a, e in per_attack_eer(ev, s).items():
                                attack_rows.append({"regime": regime, "window": window,
                                                    "model": kind, "seed": seed,
                                                    "attack": a, "eer": e})

                    if is_vg_window:                  # external VISHGUARD (1 s only)
                        for cond, vg in vgs.items():
                            if vg is None:
                                continue
                            s = model.score(vg)
                            w = metrics_at(vg["y"], s, thr)
                            calls = pd.DataFrame({"call": vg["call_id"], "y": vg["y"], "s": s})
                            cg = calls.groupby("call").agg(y=("y", "first"), s=("s", "mean"))
                            c = metrics_at(cg["y"].to_numpy(), cg["s"].to_numpy(), thr)
                            for level, m in (("window", w), ("call", c)):
                                vg_rows.append({"regime": regime, "model": kind,
                                                "seed": seed, "condition": cond,
                                                "level": level, **m})
                            print(f"    VISHGUARD/{cond:16s} window_eer={w['eer']:.3f} "
                                  f"call_auc={c['auc']:.3f} call_acc={c['acc']:.3f}",
                                  flush=True)

    if not rows:
        sys.exit("Nothing was evaluated: no cached features found in "
                 f"{PROC}. Run features/extract_features.py first.")

    # ---- raw per-seed rows ----
    rs = pd.DataFrame(rows)
    rs.to_csv(OUT / "final_results_per_seed.csv", index=False)
    pd.DataFrame(attack_rows).to_csv(OUT / "final_per_attack_eer_per_seed.csv", index=False)
    pd.DataFrame(vg_rows).to_csv(OUT / "final_vishguard_results_per_seed.csv", index=False)

    # ---- seed-averaged tables (old schema + CI / std columns) ----
    keys = ["regime", "window", "model", "eval_condition"]
    num = ["dev_eer", "threshold", "n", "eer", "auc", "acc", "bal_acc", "f1",
           "eer_lo", "eer_hi", "auc_lo", "auc_hi"]
    g = rs.groupby(keys)
    res = g[num].mean()
    res["eer_seed_std"] = g["eer"].std().fillna(0.0)
    res["auc_seed_std"] = g["auc"].std().fillna(0.0)
    res["n_seeds"] = g["seed"].nunique()
    res = res.reset_index()
    res.to_csv(OUT / "final_results.csv", index=False)

    if attack_rows:
        (pd.DataFrame(attack_rows).groupby(["regime", "window", "model", "attack"])["eer"]
         .mean().reset_index().to_csv(OUT / "final_per_attack_eer.csv", index=False))
    else:
        pd.DataFrame(attack_rows).to_csv(OUT / "final_per_attack_eer.csv", index=False)
    if vg_rows:
        vnum = ["n", "eer", "auc", "acc", "bal_acc", "f1"]
        (pd.DataFrame(vg_rows).groupby(["regime", "model", "condition", "level"])[vnum]
         .mean().reset_index().to_csv(OUT / "final_vishguard_results.csv", index=False))
    else:
        pd.DataFrame(vg_rows).to_csv(OUT / "final_vishguard_results.csv", index=False)

    for regime in res["regime"].unique():
        print(f"\n--- EER (%) by window, eval=clean, regime={regime} ---")
        t = res[(res.regime == regime) & (res.eval_condition == "clean")]
        print((t.pivot_table(index="window", columns="model", values="eer") * 100)
              .round(1).sort_index(ascending=False).to_string())
        print(f"\n--- EER (%) by eval condition, window=1.0, regime={regime} ---")
        t = res[(res.regime == regime) & (res.window == 1.0)]
        if not t.empty:
            print((t.pivot_table(index="eval_condition", columns="model",
                                 values="eer") * 100).round(1).to_string())
    print(f"\nSaved results to {OUT}/final_*.csv  (frozen 1 s models in {MODELS})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", nargs="+", type=float,
                   default=[3.0, 2.0, 1.5, 1.0, 0.75, 0.5])
    p.add_argument("--regimes", nargs="+", default=["clean", "aug"],
                   choices=list(REGIME_CONDS))
    p.add_argument("--models", nargs="+",
                   default=["logistic_regression", "random_forest", "svm"])
    p.add_argument("--skip-cnn", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--out-suffix", default="",
                   help="write outputs to outputs/<suffix>/ instead of overwriting")
    p.add_argument("--save-all-seeds", action="store_true",
                   help="save every seed's CNN at 1 s (for external evaluation)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
