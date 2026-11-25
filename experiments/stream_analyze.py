#!/usr/bin/env python3
"""Window-to-call aggregation, early detection, detection latency and classification metrics.

Reads outputs/streaming/window_scores.npz (from stream_score.py) and, for every saved model
(regime x model x seed), builds per-utterance risk scores R_N after observing the first N
windows (N = 1..nmax, i.e. 1.0, 1.5, 2.0 ... s of audio) under five aggregation rules:

    mean         running mean of the window scores
    moving_avg   highest moving average (last 3 windows) so far
    ema          exponentially weighted average (alpha = 0.5, recent windows weigh more)
    vote         fraction of windows above the frozen window threshold
    consecutive  longest run of consecutive windows above the frozen window threshold

Thresholds tau are chosen on DEV only (EER point of R_N) and then frozen:
    clean-trained models -> dev clean;  augmented models -> dev clean + telephone + codec.
Eval is applied with those frozen values. An alarm is raised at the first window where
R_N >= tau; detection latency is the audio duration at that window.

Outputs (outputs/streaming/): rules_by_N.csv, latency_table.csv, window_metrics.csv,
and figures in outputs/figures/stream_*.png.

    python3 experiments/stream_analyze.py
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from stream_common import CONDS, duration_at  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

RULES = ["mean", "moving_avg", "ema", "vote", "consecutive"]
RULE_LABEL = {"mean": "Running mean", "moving_avg": "Moving average (3 win.)",
              "ema": "Exp. weighted (a=0.5)", "vote": "Majority-vote fraction",
              "consecutive": "Consecutive high windows"}
RULE_COLOR = dict(zip(RULES, ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]))
MODEL_ORDER = ["logistic_regression", "random_forest", "svm", "cnn"]
MODEL_LABEL = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
               "svm": "SVM (RBF)", "cnn": "CNN"}
MODEL_COLOR = {"logistic_regression": "#2196F3", "random_forest": "#4CAF50",
               "svm": "#FF5722", "cnn": "#9C27B0"}
COND_TITLE = {"clean": "Clean", "telephone": "Telephone", "telephone_codec": "Telephone + G.711"}
REGIME_TITLE = {"clean": "trained on clean audio", "aug": "trained on clean + telephone + codec"}


# ----------------------------------------------------------------- helpers
def eer_thr(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def utt_matrix(d, mask, score, nmax):
    mask = mask & (d["widx"] < nmax)
    u, inv = np.unique(d["utt"][mask], return_inverse=True)
    S = np.full((len(u), nmax), np.nan)
    S[inv, d["widx"][mask]] = score[mask]
    y = np.zeros(len(u), dtype=int)
    y[inv] = d["y"][mask]
    return S, y


def rule_matrices(S, theta, alpha=0.5, M=3):
    """R[rule][:, N-1] = risk score after observing the first N windows (held after the end)."""
    n, T = S.shape
    valid = ~np.isnan(S)
    cnt = np.cumsum(valid, axis=1)
    R = {}
    R["mean"] = np.nancumsum(S, axis=1) / np.maximum(cnt, 1)

    ma = np.full_like(S, np.nan)
    for t in range(T):
        blk = S[:, max(0, t - M + 1):t + 1]
        c = (~np.isnan(blk)).sum(1)
        v = np.nansum(blk, axis=1) / np.where(c > 0, c, np.nan)
        v[~valid[:, t]] = np.nan
        ma[:, t] = v
    R["moving_avg"] = np.fmax.accumulate(ma, axis=1)

    e = S[:, 0].copy()
    ema = np.empty_like(S)
    ema[:, 0] = e
    for t in range(1, T):
        e = np.where(valid[:, t], alpha * np.nan_to_num(S[:, t]) + (1 - alpha) * e, e)
        ema[:, t] = e
    R["ema"] = ema

    hit = (S >= theta) & valid
    R["vote"] = np.cumsum(hit, axis=1) / np.maximum(cnt, 1)

    run = np.zeros(n)
    best = np.zeros(n)
    cons = np.empty_like(S)
    for t in range(T):
        run = np.where(valid[:, t], np.where(hit[:, t], run + 1, 0), run)
        best = np.maximum(best, run)
        cons[:, t] = best
    R["consecutive"] = cons
    return R


def at_threshold(y, s, tau):
    pred = s >= tau
    tpr = float(pred[y == 1].mean())
    fpr = float(pred[y == 0].mean())
    return tpr, fpr, (tpr + (1 - fpr)) / 2


# -------------------------------------------------------------------- core
def analyse(d, thr, nmax, regimes):
    keys = sorted(k[len("score__"):] for k in d if k.startswith("score__"))
    rules_rows, lat_rows, win_rows, roc_store = [], [], [], {}

    for key in keys:
        regime, kind, seed = key.split("|")
        seed = int(seed)
        if regime not in regimes:
            continue
        theta = thr[key]
        score = d["score__" + key]

        # ---- DEV: frozen tau for every rule and N
        dev_conds = ["clean"] if regime == "clean" else CONDS
        Rdev, ydev = {r: [] for r in RULES}, []
        for c in dev_conds:
            m = (d["split"] == "dev") & (d["cond"] == c)
            if not m.any():
                continue
            S, y = utt_matrix(d, m, score, nmax)
            R = rule_matrices(S, theta)
            for r in RULES:
                Rdev[r].append(R[r])
            ydev.append(y)
        if not ydev:
            continue
        ydev = np.concatenate(ydev)
        if len(set(ydev)) < 2:
            continue
        Rdev = {r: np.concatenate(v) for r, v in Rdev.items()}
        tau = {r: [eer_thr(ydev, Rdev[r][:, n])[1] for n in range(nmax)] for r in RULES}

        # ---- EVAL per condition
        for c in CONDS:
            m = (d["split"] == "eval") & (d["cond"] == c)
            if not m.any():
                continue
            S, y = utt_matrix(d, m, score, nmax)
            if len(set(y)) < 2:
                continue
            R = rule_matrices(S, theta)
            for r in RULES:
                for n in range(nmax):
                    s = R[r][:, n]
                    eer, _ = eer_thr(y, s)
                    tpr, fpr, bal = at_threshold(y, s, tau[r][n])
                    rules_rows.append(dict(regime=regime, kind=kind, seed=seed, cond=c, rule=r,
                                           N=n + 1, duration_s=duration_at(n + 1), eer=eer,
                                           auc=float(roc_auc_score(y, s)), tpr=tpr, fpr=fpr,
                                           bal_acc=bal, tau=tau[r][n]))
                # alarm / latency with the final-N tau, applied at every step
                alarm = R[r] >= tau[r][-1]
                ever = alarm.any(axis=1)
                first = np.argmax(alarm, axis=1) + 1
                sp = y == 1
                lat = np.array([duration_at(n) for n in first[sp & ever]])
                lat_rows.append(dict(
                    regime=regime, kind=kind, seed=seed, cond=c, rule=r,
                    alarm_tpr=float(ever[sp].mean()), alarm_fpr=float(ever[~sp].mean()),
                    mean_latency_s=float(lat.mean()) if len(lat) else np.nan,
                    median_latency_s=float(np.median(lat)) if len(lat) else np.nan,
                    n_spoof=int(sp.sum()), n_bonafide=int((~sp).sum())))

            # window-level metrics at the frozen window threshold
            sw, yw = score[m], d["y"][m]
            pred = (sw >= theta).astype(int)
            tn = int(((pred == 0) & (yw == 0)).sum()); fp = int(((pred == 1) & (yw == 0)).sum())
            fn = int(((pred == 0) & (yw == 1)).sum()); tp = int(((pred == 1) & (yw == 1)).sum())
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            fpr_w = fp / max(fp + tn, 1)
            win_rows.append(dict(regime=regime, kind=kind, seed=seed, cond=c, n_windows=len(yw),
                                 precision=prec, recall=rec,
                                 f1=2 * prec * rec / max(prec + rec, 1e-12),
                                 accuracy=(tp + tn) / len(yw),
                                 bal_acc=(rec + (1 - fpr_w)) / 2, auc=float(roc_auc_score(yw, sw)),
                                 eer=eer_thr(yw, sw)[0], tn=tn, fp=fp, fn=fn, tp=tp, threshold=theta))
            fpr_c, tpr_c, _ = roc_curve(yw, sw)
            roc_store[(regime, kind, seed, c)] = (fpr_c, tpr_c, float(roc_auc_score(yw, sw)))
        print(f"  analysed {key}", flush=True)

    return (pd.DataFrame(rules_rows), pd.DataFrame(lat_rows), pd.DataFrame(win_rows), roc_store)


def agg(df, by, cols):
    g = df.groupby(by)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().fillna(0.0).add_suffix("_std"))
    out["n_runs"] = df.groupby(by).size()
    return out.reset_index()


# ----------------------------------------------------------------- figures
def fig_early(ra, regime, kind, path):
    d = ra[(ra.regime == regime) & (ra.kind == kind)]
    conds = [c for c in CONDS if c in set(d.cond)]
    if d.empty:
        return
    fig, axes = plt.subplots(2, len(conds), figsize=(5.2 * len(conds), 7.2),
                             sharex=True, squeeze=False)
    for j, c in enumerate(conds):
        for r in RULES:
            s = d[(d.cond == c) & (d.rule == r)].sort_values("duration_s")
            x = s.duration_s.to_numpy()
            for row, col in enumerate(["eer", "bal_acc"]):
                m = s[col + "_mean"].to_numpy() * 100
                sd = s[col + "_std"].to_numpy() * 100
                axes[row][j].plot(x, m, marker="o", ms=4, color=RULE_COLOR[r], label=RULE_LABEL[r])
                if sd.max() > 0:
                    axes[row][j].fill_between(x, m - sd, m + sd, color=RULE_COLOR[r], alpha=0.12)
        axes[0][j].set_title(COND_TITLE[c])
        axes[0][j].set_ylim(0, 60); axes[1][j].set_ylim(40, 100)
        axes[1][j].set_xlabel("Observed audio (s)")
        for row in (0, 1):
            axes[row][j].grid(alpha=0.3)
        axes[0][j].axhline(50, color="k", ls=":", lw=0.8)
    axes[0][0].set_ylabel("Utterance-level EER (%)")
    axes[1][0].set_ylabel("Balanced accuracy at frozen tau (%)")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Early detection: {MODEL_LABEL[kind]} {REGIME_TITLE[regime]} "
                 "(ASVspoof 2019 LA eval; mean +/- std over seeds where available)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150); plt.close(fig)
    print("  ", path.name)


def fig_roc(roc, regime, path):
    conds = [c for c in CONDS if any(k[0] == regime and k[3] == c for k in roc)]
    if not conds:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4.8), squeeze=False)
    for ax, c in zip(axes[0], conds):
        for (reg, kind, seed, cc), (f, t, a) in sorted(roc.items()):
            if reg != regime or cc != c:
                continue
            lab = f"{MODEL_LABEL[kind]} (AUC {a:.2f})" if kind != "cnn" or seed == min(
                k[2] for k in roc if k[0] == regime and k[1] == "cnn" and k[3] == c) else None
            ax.plot(f, t, color=MODEL_COLOR[kind], lw=1.0 if kind == "cnn" else 1.6,
                    alpha=0.55 if kind == "cnn" else 0.9, label=lab)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(COND_TITLE[c]); ax.set_xlabel("False positive rate")
        ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("True positive rate")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Window-level ROC, models {REGIME_TITLE[regime]} (CNN: one line per seed)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150); plt.close(fig)
    print("  ", path.name)


def fig_confusion(win, regime, kind, path):
    d = win[(win.regime == regime) & (win.kind == kind)]
    conds = [c for c in CONDS if c in set(d.cond)]
    if d.empty:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(4.6 * len(conds), 4.4), squeeze=False)
    for ax, c in zip(axes[0], conds):
        s = d[d.cond == c]
        cm = np.array([[s.tn.mean(), s.fp.mean()], [s.fn.mean(), s.tp.mean()]])
        rowpct = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(rowpct, cmap="Purples", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.0f}\n({rowpct[i, j]:.0f}%)", ha="center", va="center",
                        color="white" if rowpct[i, j] > 55 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred. bona fide", "pred. spoof"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true bona fide", "true spoof"])
        ax.set_title(f"{COND_TITLE[c]}  (bal. acc {s.bal_acc.mean() * 100:.1f}%)")
    fig.suptitle(f"{MODEL_LABEL[kind]} {REGIME_TITLE[regime]}: window-level confusion "
                 "(mean counts over seeds, frozen dev threshold)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150); plt.close(fig)
    print("  ", path.name)


def fig_latency(la, regime, kind, path):
    d = la[(la.regime == regime) & (la.kind == kind)]
    conds = [c for c in CONDS if c in set(d.cond)]
    if d.empty:
        return
    fig, axes = plt.subplots(2, len(conds), figsize=(5.2 * len(conds), 7.0),
                             squeeze=False, sharey="row")
    x = np.arange(len(RULES))
    for j, c in enumerate(conds):
        s = d[d.cond == c].set_index("rule").reindex(RULES)
        axes[0][j].bar(x - 0.2, s.alarm_tpr_mean.to_numpy() * 100, 0.4, color="#2e7d32",
                       label="spoof utterances flagged")
        axes[0][j].bar(x + 0.2, s.alarm_fpr_mean.to_numpy() * 100, 0.4, color="#c62828",
                       label="bona fide flagged (false alarm)")
        axes[1][j].bar(x, np.nan_to_num(s.mean_latency_s_mean.to_numpy()), 0.6, color="#5c6bc0")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xticks(x); ax.set_xticklabels(["mean", "mov.avg", "EMA", "vote", "consec."],
                                                  fontsize=9)
            ax.grid(alpha=0.3, axis="y")
        axes[0][j].set_title(COND_TITLE[c]); axes[0][j].set_ylim(0, 105)
    axes[0][0].set_ylabel("Share of utterances (%)")
    axes[1][0].set_ylabel("Mean time to first alarm (s)")
    axes[0][-1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Alarm behaviour and detection latency: {MODEL_LABEL[kind]} {REGIME_TITLE[regime]}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150); plt.close(fig)
    print("  ", path.name)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="outputs/streaming")
    ap.add_argument("--nmax", type=int, default=6, help="max windows observed (6 -> 3.5 s)")
    ap.add_argument("--regimes", nargs="+", default=["clean", "aug"])
    a = ap.parse_args()

    sdir = ROOT / a.dir
    fdir = ROOT / "outputs" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    z = np.load(sdir / "window_scores.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    thr = json.loads((sdir / "thresholds.json").read_text())

    print("analysing ...")
    rules, lat, win, roc = analyse(d, thr, a.nmax, a.regimes)
    if rules.empty:
        sys.exit("Nothing to analyse (need dev and eval windows with both classes).")

    ra = agg(rules, ["regime", "kind", "cond", "rule", "N", "duration_s"],
             ["eer", "auc", "tpr", "fpr", "bal_acc"])
    la = agg(lat, ["regime", "kind", "cond", "rule"],
             ["alarm_tpr", "alarm_fpr", "mean_latency_s", "median_latency_s"])
    wa = agg(win, ["regime", "kind", "cond"],
             ["precision", "recall", "f1", "accuracy", "bal_acc", "auc", "eer", "tn", "fp", "fn", "tp"])
    ra.to_csv(sdir / "rules_by_N.csv", index=False)
    la.to_csv(sdir / "latency_table.csv", index=False)
    win.to_csv(sdir / "window_metrics_per_seed.csv", index=False)
    wa.to_csv(sdir / "window_metrics.csv", index=False)

    print("\nfigures:")
    for regime in a.regimes:
        fig_early(ra, regime, "cnn", fdir / f"stream_early_detection_{regime}.png")
        fig_roc(roc, regime, fdir / f"stream_roc_{regime}.png")
        fig_confusion(win, regime, "cnn", fdir / f"stream_confusion_cnn_{regime}.png")
        fig_latency(la, regime, "cnn", fdir / f"stream_latency_{regime}.png")

    pd.set_option("display.width", 220)
    nmax = ra.N.max()
    print(f"\n=== Utterance-level EER (%) after {nmax} windows "
          f"({duration_at(nmax):.1f} s), by aggregation rule ===")
    t = ra[ra.N == nmax].copy()
    t["EER"] = (t.eer_mean * 100).round(1)
    print(t.pivot_table(index=["regime", "kind", "cond"], columns="rule", values="EER")
          [RULES].to_string())
    print("\n=== Alarm TPR / FPR and mean latency, CNN ===")
    c = la[la.kind == "cnn"].copy()
    for col in ("alarm_tpr_mean", "alarm_fpr_mean"):
        c[col] = (c[col] * 100).round(0)
    c["lat_s"] = c.mean_latency_s_mean.round(2)
    print(c[["regime", "cond", "rule", "alarm_tpr_mean", "alarm_fpr_mean", "lat_s"]]
          .to_string(index=False))
    print("\n=== Window-level precision / recall / F1 (frozen threshold) ===")
    w = wa.copy()
    for col in ("precision", "recall", "f1", "bal_acc"):
        w[col] = (w[col + "_mean"] * 100).round(1)
    print(w[["regime", "kind", "cond", "precision", "recall", "f1", "bal_acc"]]
          .to_string(index=False))
    print(f"\nSaved tables to {sdir}")


if __name__ == "__main__":
    main()
