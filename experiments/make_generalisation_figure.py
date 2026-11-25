#!/usr/bin/env python3
"""Generalisation-gap figure: the same frozen models on three datasets.

    ASVspoof 2019 LA eval  (in-domain)       <- outputs/final_results.csv + outputs/seeds5/final_results_per_seed.csv
    VISHGUARD              (external)        <- outputs/final_vishguard_results.csv  (window level)
    DEEP-VOICE             (external, RVC)   <- outputs/deepvoice/deepvoice_results_per_model.csv

Metric: window-level EER (%), 1 s windows, frozen ASVspoof-dev thresholds irrelevant for EER.
CNN bars are the mean over seeds with +/- 1 std error bars; LR/RF/SVM are single models.
Chance is 50% EER.

    python3 experiments/make_generalisation_figure.py                 # models trained on clean audio
    python3 experiments/make_generalisation_figure.py --regime aug    # channel-augmented training
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"

MODELS = ["logistic_regression", "random_forest", "svm", "cnn"]
COLORS = {"logistic_regression": "#2196F3", "random_forest": "#4CAF50",
          "svm": "#FF5722", "cnn": "#9C27B0"}
LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
          "svm": "SVM (RBF)", "cnn": "CNN (mean of seeds)"}
DS_ASV = "ASVspoof 2019 LA eval\n(in-domain)"
DS_VG = "VISHGUARD\n(external)"
DS_DV = "DEEP-VOICE\n(external)"
DATASETS = [DS_ASV, DS_VG, DS_DV]
COND_TITLE = {"clean": "Clean audio", "telephone": "Telephone audio (300-3400 Hz, 8 kHz)",
              "telephone_codec": "Telephone + G.711 A-law"}
KEEP = ["dataset", "regime", "model", "condition", "eer", "seed"]


def asv(window):
    b = pd.read_csv(OUT / "final_results.csv")
    if "window" in b.columns:
        b = b[b["window"] == window]
    b = b.rename(columns={"eval_condition": "condition"}).copy()
    b["seed"] = -1
    f = OUT / "seeds5" / "final_results_per_seed.csv"
    if f.exists():
        s = pd.read_csv(f)
        if "window" in s.columns:
            s = s[s["window"] == window]
        s = s[s["model"] == "cnn"].rename(columns={"eval_condition": "condition"})
        b = pd.concat([b[b["model"] != "cnn"], s], ignore_index=True)
    b["dataset"] = DS_ASV
    return b[KEEP]


def vishguard():
    v = pd.read_csv(OUT / "final_vishguard_results.csv")
    v = v[v["level"] == "window"].copy()
    v["seed"] = -1
    for p in glob.glob(str(OUT / "seeds5" / "*vishguard*.csv")):
        s = pd.read_csv(p)
        need = {"model", "regime", "condition", "eer", "seed"}
        if need <= set(s.columns):
            if "level" in s.columns:
                s = s[s["level"] == "window"]
            s = s[s["model"] == "cnn"]
            if len(s):
                v = pd.concat([v[v["model"] != "cnn"], s], ignore_index=True)
                break
    v["dataset"] = DS_VG
    return v[KEEP]


def deepvoice():
    d = pd.read_csv(OUT / "deepvoice" / "deepvoice_results_per_model.csv").copy()
    d["dataset"] = DS_DV
    return d[KEEP]


def aggregate(df):
    g = df.groupby(["dataset", "regime", "model", "condition"])["eer"]
    t = g.agg(["mean", "std", "count"]).reset_index()
    t["std"] = t["std"].fillna(0.0)
    return t


def draw(t, regime, conds, path):
    fig, axes = plt.subplots(1, len(conds), figsize=(6.4 * len(conds), 5.2),
                             sharey=True, squeeze=False)
    w = 0.8 / len(MODELS)
    x = np.arange(len(DATASETS))
    for ax, cond in zip(axes[0], conds):
        for i, m in enumerate(MODELS):
            means, errs = [], []
            for ds in DATASETS:
                r = t[(t.dataset == ds) & (t.regime == regime) &
                      (t.model == m) & (t.condition == cond)]
                means.append(float(r["mean"].iloc[0]) * 100 if len(r) else np.nan)
                errs.append(float(r["std"].iloc[0]) * 100 if len(r) else 0.0)
            pos = x + (i - len(MODELS) / 2 + 0.5) * w
            ax.bar(pos, np.nan_to_num(np.array(means)), w, color=COLORS[m], label=LABELS[m],
                   hatch="//" if m == "cnn" else "", edgecolor="white",
                   yerr=np.array(errs) if m == "cnn" else None,
                   error_kw={"capsize": 3, "lw": 1.2})
            for xi, v, e in zip(pos, means, errs):
                if not np.isnan(v):
                    ax.text(xi, v + e + 0.8, f"{v:.0f}", ha="center", fontsize=8)
        ax.axhline(50, color="k", ls=":", lw=1)
        ax.text(-0.48, 51, "chance (50%)", ha="left", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, fontsize=9)
        ax.set_title(COND_TITLE.get(cond, cond))
        ax.set_ylim(0, 70)
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("Window-level EER (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(MODELS), fontsize=9,
               bbox_to_anchor=(0.5, 0.945), frameon=False)
    reg = {"clean": "trained on clean ASVspoof audio",
           "aug": "trained on clean + telephone + codec ASVspoof audio"}[regime]
    fig.suptitle(f"Generalisation gap: frozen models {reg}, 1 s windows")
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    fig.text(0.5, 0.02,
             "VISHGUARD: 40 calls, labels are call intent (fraud vs legitimate), not confirmed "
             "voice authenticity.\n"
             "DEEP-VOICE: real vs RVC voice conversion (8 real / 56 fake files). "
             "CNN error bars = std over seeds; LR/RF/SVM are single models.",
             ha="center", va="bottom", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regime", choices=["clean", "aug"], default="clean")
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--conditions", nargs="+", default=["clean", "telephone"])
    a = p.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    parts = [asv(a.window)]
    for name, fn in [("VISHGUARD", vishguard), ("DEEP-VOICE", deepvoice)]:
        try:
            parts.append(fn())
        except FileNotFoundError as e:
            print(f"skipping {name}: {e}")
    t = aggregate(pd.concat(parts, ignore_index=True))

    path = FIG / f"final_generalisation_gap_{a.regime}.png"
    draw(t, a.regime, a.conditions, path)
    print("saved", path.name)

    tab = t[t.regime == a.regime].copy()
    tab["dataset"] = tab["dataset"].str.replace("\n", " ")
    tab["EER_%"] = (tab["mean"] * 100).round(1)
    tab["std_%"] = (tab["std"] * 100).round(1)
    tab = tab.rename(columns={"count": "n_runs"})[
        ["dataset", "condition", "model", "EER_%", "std_%", "n_runs"]]
    tab.to_csv(OUT / f"final_generalisation_table_{a.regime}.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
