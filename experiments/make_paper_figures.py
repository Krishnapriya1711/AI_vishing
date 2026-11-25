#!/usr/bin/env python3
"""Paper figures with error bars (seed std) from the multi-seed result files."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["logistic_regression", "random_forest", "svm", "cnn"]
NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
         "svm": "SVM (RBF)", "cnn": "Mel-CNN"}
COL = {"logistic_regression": "#2196F3", "random_forest": "#4CAF50",
       "svm": "#FF5722", "cnn": "#9C27B0"}
MARK = {"logistic_regression": "o", "random_forest": "s", "svm": "D", "cnn": "^"}
CONDS = ["clean", "telephone", "telephone_codec"]
CLAB = {"clean": "Clean\n(16 kHz)", "telephone": "Telephone\n(300-3400 Hz, 8 kHz)",
        "telephone_codec": "Telephone + G.711\nA-law"}
REG = {"clean": "Trained on clean audio only", "aug": "Trained on clean + telephone + codec"}


def stats(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return np.nan, 0.0
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def read(p):
    p = OUT / p
    if not p.exists():
        print(f"  (missing {p.relative_to(ROOT)})")
        return None
    return pd.read_csv(p)


first = read("final_results.csv")                       # baselines, first multi-seed run
s1 = read("seeds5/final_results_per_seed.csv")          # CNN, 1 s, 5 seeds
s05 = read("seeds5_w05/final_results_per_seed.csv")     # CNN, 0.5 s, 5 seeds
abl = read("ablation/final_results_per_seed.csv")


def eer_stat(model, regime, window, cond):
    if model == "cnn":
        d = {1.0: s1, 0.5: s05}[window]
        v = d[(d.model == "cnn") & (d.regime == regime) & (d.eval_condition == cond)]["eer"]
        return stats(v.to_numpy())
    d = first[(first.model == model) & (first.regime == regime) &
              (first.window == window) & (first.eval_condition == cond)]
    if d.empty:
        return np.nan, 0.0
    sd = float(d.eer_seed_std.iloc[0]) if d.n_seeds.iloc[0] > 1 else 0.0
    return float(d.eer.iloc[0]), sd


def grouped_bars(ax, groups, series, value, colors, names, scale=100.0,
                 fmt="{:.1f}", labels=True):
    n, width = len(series), 0.8 / len(series)
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        vals = [value(s, g) for g in groups]
        m = np.array([v[0] for v in vals], dtype=float) * scale
        e = np.nan_to_num(np.array([v[1] for v in vals], dtype=float) * scale)
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, np.nan_to_num(m), width * 0.95, yerr=e, capsize=2,
               color=colors[s], label=names[s], edgecolor="white",
               hatch="//" if s == "cnn" else None, error_kw=dict(lw=0.8))
        if labels:
            for px, mv, ev in zip(pos, m, e):
                if np.isfinite(mv):
                    ax.text(px, mv + ev + 0.5 * (scale / 100.0), fmt.format(mv),
                            ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)


def fig_window():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    xs = np.array([0.5, 1.0])
    for ax, reg in zip(axes, ["clean", "aug"]):
        for m in MODELS:
            st = [eer_stat(m, reg, float(w), "clean") for w in xs]
            ax.errorbar(xs, np.array([a * 100 for a, _ in st]),
                        yerr=np.array([b * 100 for _, b in st]), marker=MARK[m],
                        color=COL[m], linestyle="--" if m == "cnn" else "-", lw=2,
                        capsize=3, label=NAMES[m])
        ax.set_xticks(xs); ax.set_xlim(0.4, 1.1); ax.set_ylim(0, 60)
        ax.set_xlabel("Window length (s)"); ax.set_title(REG[reg]); ax.grid(alpha=0.3)
    axes[0].set_ylabel("EER (%) on ASVspoof 2019 LA eval (clean)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("EER vs window length (frozen dev threshold; chance = 50%; bars = seed std)")
    fig.tight_layout(); fig.savefig(FIG / "final_eer_vs_window.png", dpi=200); plt.close(fig)


def fig_robust():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, reg in zip(axes, ["clean", "aug"]):
        grouped_bars(ax, CONDS, MODELS, lambda m, c, r=reg: eer_stat(m, r, 1.0, c), COL, NAMES)
        ax.set_xticklabels([CLAB[c] for c in CONDS], fontsize=8)
        ax.set_ylim(0, 60); ax.set_title(REG[reg])
    axes[0].set_ylabel("EER (%)"); axes[0].legend(fontsize=8)
    fig.suptitle("Telephone / codec robustness, 1 s windows, ASVspoof 2019 LA eval (bars = seed std)")
    fig.tight_layout(); fig.savefig(FIG / "final_phone_robustness.png", dpi=200); plt.close(fig)


def fig_ablation():
    if abl is None or s1 is None:
        return
    series = ["clean", "aug_tel", "aug_codec", "aug"]
    names = {"clean": "clean only", "aug_tel": "clean + telephone",
             "aug_codec": "clean + A-law", "aug": "clean + tel. + A-law"}
    colors = {"clean": "#9E9E9E", "aug_tel": "#64B5F6", "aug_codec": "#FFB74D", "aug": "#9C27B0"}

    def val(sr, c):
        d = s1 if sr in ("clean", "aug") else abl
        v = d[(d.model == "cnn") & (d.regime == sr) & (d.eval_condition == c)]["eer"]
        return stats(v.to_numpy())
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    grouped_bars(ax, CONDS, series, val, colors, names)
    ax.set_xticklabels([CLAB[c] for c in CONDS], fontsize=8)
    ax.set_ylim(0, 45); ax.set_ylabel("Mel-CNN EER (%)"); ax.legend(title="training data", fontsize=8)
    ax.set_title("Training-condition ablation, 1 s windows (mean over 5 seeds, bars = std)")
    fig.tight_layout(); fig.savefig(FIG / "final_ablation.png", dpi=200); plt.close(fig)


def fig_attack():
    pa_s = read("seeds5/final_per_attack_eer_per_seed.csv")
    pa_f = read("final_per_attack_eer.csv")
    if pa_s is None or pa_f is None:
        return
    pa_s = pa_s[(pa_s.regime == "clean") & (pa_s.window == 1.0) & (pa_s.model == "cnn")]
    pa_f = pa_f[(pa_f.regime == "clean") & (pa_f.window == 1.0)]
    attacks = sorted(pa_f.attack.unique())

    def val(m, a):
        if m == "cnn":
            return stats(pa_s[pa_s.attack == a]["eer"].to_numpy())
        d = pa_f[(pa_f.model == m) & (pa_f.attack == a)]
        return (float(d.eer.iloc[0]) if len(d) else np.nan), 0.0
    fig, ax = plt.subplots(figsize=(12, 4.4))
    grouped_bars(ax, attacks, MODELS, val, COL, NAMES, labels=False)
    ax.set_xticklabels(attacks, rotation=45); ax.set_ylim(0, 60)
    ax.set_ylabel("EER (%) vs bona fide"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("Per-attack EER, 1 s windows, clean eval, clean-trained (A07-A19 unseen in training)")
    fig.tight_layout(); fig.savefig(FIG / "final_per_attack_eer.png", dpi=200); plt.close(fig)


def fig_vishguard():
    vf = read("final_vishguard_results.csv")
    vs = read("seeds5/final_vishguard_results_per_seed.csv")
    if vf is None or vs is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, reg in zip(axes, ["clean", "aug"]):
        def val(m, c, r=reg):
            if m == "cnn":
                d = vs[(vs.model == "cnn") & (vs.regime == r) & (vs.condition == c) & (vs.level == "call")]
                return stats(d["auc"].to_numpy())
            d = vf[(vf.model == m) & (vf.regime == r) & (vf.condition == c) & (vf.level == "call")]
            return (float(d.auc.iloc[0]) if len(d) else np.nan), 0.0
        grouped_bars(ax, CONDS, MODELS, val, COL, NAMES, scale=1.0, fmt="{:.2f}")
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xticklabels([CLAB[c] for c in CONDS], fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(REG[reg])
    axes[0].set_ylabel("Call-level ROC-AUC"); axes[0].legend(fontsize=8)
    fig.suptitle("Exploratory: 40 VISHGUARD calls (labels = call intent; dotted = chance; bars = seed std)")
    fig.tight_layout(); fig.savefig(FIG / "final_vishguard_call_auc.png", dpi=200); plt.close(fig)


def fig_deepvoice():
    r = read("deepvoice/deepvoice_results_per_model.csv")
    if r is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, reg in zip(axes, ["clean", "aug"]):
        def val(m, c, rg=reg):
            d = r[(r.model == m) & (r.regime == rg) & (r.condition == c)]
            if d.empty:
                return np.nan, 0.0
            return float(d.auc.mean()), float(((d.auc_hi - d.auc_lo) / 2).mean())
        grouped_bars(ax, CONDS, MODELS, val, COL, NAMES, scale=1.0, fmt="{:.2f}")
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xticklabels([CLAB[c] for c in CONDS], fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(REG[reg])
    axes[0].set_ylabel("Window-level ROC-AUC"); axes[0].legend(fontsize=8)
    fig.suptitle("DEEP-VOICE (RVC), frozen ASVspoof-trained models (bars = half-width of "
                 "file-cluster 95% bootstrap interval)")
    fig.tight_layout(); fig.savefig(FIG / "final_deepvoice.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    if first is None or s1 is None or s05 is None:
        raise SystemExit("Need outputs/final_results.csv, outputs/seeds5/ and outputs/seeds5_w05/")
    for f in (fig_window, fig_robust, fig_ablation, fig_attack, fig_vishguard, fig_deepvoice):
        f()
    print("figures in", FIG)
