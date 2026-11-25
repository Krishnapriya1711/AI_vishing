#!/usr/bin/env python3
"""Figures and a summary table from outputs/final_*.csv (run_final_experiments.py).

    python experiments/make_final_figures.py                # 1 s window for the bar charts
    python experiments/make_final_figures.py --window 0.5

Writes outputs/figures/final_*.png and outputs/final_summary_table.csv.
All y-axes start at 0. Figures for data that is not in the CSVs are skipped.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"

COLORS = {"logistic_regression": "#2196F3", "random_forest": "#4CAF50",
          "svm": "#FF5722", "cnn": "#9C27B0"}
LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
          "svm": "SVM (RBF)", "cnn": "CNN"}
MARK = {"logistic_regression": "o", "random_forest": "s", "svm": "D", "cnn": "^"}
REGIME = {"clean": "Trained on clean audio only",
          "aug": "Trained on clean + telephone + codec"}
COND = {"clean": "Clean\n(16 kHz)", "telephone": "Telephone\n(300-3400 Hz, 8 kHz)",
        "telephone_codec": "Telephone + G.711\nA-law"}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150,
                     "savefig.dpi": 150, "savefig.bbox": "tight"})


def order(models):
    return [m for m in ["logistic_regression", "random_forest", "svm", "cnn"] if m in models]


def fig_eer_vs_window(res):
    d = res[res.eval_condition == "clean"]
    regimes = [r for r in ["clean", "aug"] if r in set(d.regime)]
    if d.empty or d.window.nunique() < 2:
        return
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.6 * len(regimes), 4.6),
                             sharey=True, squeeze=False)
    for ax, reg in zip(axes[0], regimes):
        for m in order(set(d.model)):
            s = d[(d.regime == reg) & (d.model == m)].sort_values("window", ascending=False)
            ax.plot(s.window.to_numpy(), (s.eer * 100).to_numpy(), marker=MARK[m], color=COLORS[m],
                    label=LABELS[m], lw=2.5 if m == "cnn" else 1.8,
                    ls="--" if m == "cnn" else "-")
        ax.set_title(REGIME[reg]); ax.set_xlabel("Window length (s)")
        ax.set_xticks(sorted(d.window.unique(), reverse=True))
        ax.invert_xaxis(); ax.grid(alpha=0.3); ax.set_ylim(0, 60)
    axes[0][0].set_ylabel("EER (%) on ASVspoof 2019 LA eval (clean)")
    axes[0][-1].legend(loc="lower left", fontsize=9)
    fig.suptitle("EER vs window length (frozen dev threshold; chance = 50%)")
    fig.savefig(FIG / "final_eer_vs_window.png"); plt.close(fig)
    print("  final_eer_vs_window.png")


def fig_phone(res, window):
    d = res[res.window == window]
    regimes = [r for r in ["clean", "aug"] if r in set(d.regime)]
    conds = [c for c in COND if c in set(d.eval_condition)]
    if d.empty or len(conds) < 2:
        return
    models = order(set(d.model)); w = 0.8 / len(models)
    fig, axes = plt.subplots(1, len(regimes), figsize=(6.2 * len(regimes), 4.8),
                             sharey=True, squeeze=False)
    for ax, reg in zip(axes[0], regimes):
        x = np.arange(len(conds))
        for i, m in enumerate(models):
            vals = [d[(d.regime == reg) & (d.model == m) & (d.eval_condition == c)].eer.mean() * 100
                    for c in conds]
            ax.bar(x + (i - len(models) / 2 + 0.5) * w, np.asarray(vals, dtype=float), w, color=COLORS[m],
                   label=LABELS[m], hatch="//" if m == "cnn" else "", edgecolor="white")
            for xi, v in zip(x + (i - len(models) / 2 + 0.5) * w, vals):
                ax.text(xi, v + 0.6, f"{v:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([COND[c] for c in conds], fontsize=9)
        ax.set_title(REGIME[reg]); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 60)
    axes[0][0].set_ylabel("EER (%)"); axes[0][-1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Telephone / codec robustness, {window:g} s windows (ASVspoof 2019 LA eval)")
    fig.savefig(FIG / "final_phone_robustness.png"); plt.close(fig)
    print("  final_phone_robustness.png")


def fig_vishguard(vg):
    d = vg[vg.level == "call"]
    if d.empty:
        return
    regimes = [r for r in ["clean", "aug"] if r in set(d.regime)]
    conds = [c for c in COND if c in set(d.condition)]
    models = order(set(d.model)); w = 0.8 / len(models)
    fig, axes = plt.subplots(1, len(regimes), figsize=(6.2 * len(regimes), 4.6),
                             sharey=True, squeeze=False)
    for ax, reg in zip(axes[0], regimes):
        x = np.arange(len(conds))
        for i, m in enumerate(models):
            vals = [d[(d.regime == reg) & (d.model == m) & (d.condition == c)].auc.mean()
                    for c in conds]
            ax.bar(x + (i - len(models) / 2 + 0.5) * w, np.asarray(vals, dtype=float), w, color=COLORS[m],
                   label=LABELS[m], hatch="//" if m == "cnn" else "", edgecolor="white")
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([COND[c] for c in conds], fontsize=9)
        ax.set_title(REGIME[reg]); ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("Call-level ROC-AUC"); axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("External evaluation on VISHGUARD (40 calls, mean window score per call; "
                 "dotted line = chance)")
    fig.savefig(FIG / "final_vishguard_call_auc.png"); plt.close(fig)
    print("  final_vishguard_call_auc.png")


def fig_attacks(att, window):
    d = att[(att.window == window) & (att.regime == "clean")]
    if d.empty:
        return
    piv = d.pivot_table(index="attack", columns="model", values="eer") * 100
    models = order(set(piv.columns)); w = 0.8 / len(models); x = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(piv)), 4.6))
    for i, m in enumerate(models):
        ax.bar(x + (i - len(models) / 2 + 0.5) * w, piv[m].to_numpy(), w, color=COLORS[m],
               label=LABELS[m], hatch="//" if m == "cnn" else "", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(list(piv.index), rotation=45)
    ax.set_ylabel("EER (%) vs bonafide"); ax.set_ylim(0, 60); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"Per-attack EER, {window:g} s windows, clean eval "
                 "(A07-A19 are unseen in training)")
    ax.legend(fontsize=8); fig.savefig(FIG / "final_per_attack_eer.png"); plt.close(fig)
    print("  final_per_attack_eer.png")


def summary(res):
    d = res.copy()
    d["EER_%"] = (d.eer * 100).round(1)
    t = d.pivot_table(index=["regime", "window", "eval_condition"], columns="model",
                      values="EER_%")
    t = t[order(set(t.columns))]
    t.to_csv(OUT / "final_summary_table.csv")
    print("\nEER (%) summary, saved to final_summary_table.csv:")
    print(t.to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=float, default=1.0)
    a = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(OUT / "final_results.csv")
    print("Figures:")
    fig_eer_vs_window(res)
    fig_phone(res, a.window)
    for name, fn in [("final_vishguard_results.csv", fig_vishguard)]:
        f = OUT / name
        if f.exists() and f.stat().st_size > 5:
            fn(pd.read_csv(f))
    f = OUT / "final_per_attack_eer.csv"
    if f.exists() and f.stat().st_size > 5:
        fig_attacks(pd.read_csv(f), a.window)
    summary(res)


if __name__ == "__main__":
    main()
