#!/usr/bin/env python3
"""ROC curves + confusion matrices at the frozen dev threshold (CNN, 1 s, ASVspoof eval)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from run_final_experiments import CnnModel, LightweightCNN       # noqa: E402

SEED = 42                                    # fixed in advance, not picked from results
CONDS = ["clean", "telephone", "telephone_codec"]
TITLE = {"clean": "Clean", "telephone": "Telephone", "telephone_codec": "Telephone + G.711 A-law"}
REG = {"clean": "trained on clean", "aug": "trained on clean+tel+codec"}


def load(regime):
    m = CnnModel(seed=0)
    ck = torch.load(ROOT / f"outputs/seeds5/models/{regime}_w1_cnn_s{SEED}.pt", map_location=m.dev)
    m.net = LightweightCNN().to(m.dev)
    m.net.load_state_dict(ck["state"])
    m.net.eval()
    return m, float(ck["thr"])


def main():
    fig_dir = ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = {r: load(r) for r in REG}
    scores, ys = {}, {}
    for c in CONDS:
        d = np.load(ROOT / "data" / "processed" / "asvspoof" / f"eval_{c}_w1.npz")
        y, mel = d["y"], d["mel"]
        ys[c] = y
        for r, (m, thr) in models.items():
            scores[(r, c)] = m.score({"mel": mel, "y": y})

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, c in zip(axes, CONDS):
        for r in REG:
            fpr, tpr, _ = roc_curve(ys[c], scores[(r, c)])
            ax.plot(fpr, tpr, label=REG[r])
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_title(TITLE[c]); ax.set_xlabel("False positive rate"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("True positive rate (spoof)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"ROC, Mel-CNN, 1 s windows, ASVspoof 2019 LA eval (seed {SEED})")
    fig.tight_layout()
    fig.savefig(fig_dir / "final_roc.png", dpi=200)

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for i, r in enumerate(REG):
        thr = models[r][1]
        for j, c in enumerate(CONDS):
            pred = (scores[(r, c)] >= thr).astype(int)
            cm = confusion_matrix(ys[c], pred, labels=[0, 1])
            rows.append(dict(regime=r, condition=c, tn=cm[0, 0], fp=cm[0, 1], fn=cm[1, 0], tp=cm[1, 1]))
            norm = cm / cm.sum(1, keepdims=True)
            ax = axes[i, j]
            ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{cm[a, b]}\n({norm[a, b]*100:.0f}%)", ha="center", va="center",
                            color="white" if norm[a, b] > 0.5 else "black", fontsize=9)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["pred bona fide", "pred spoof"], fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["bona fide", "spoof"], fontsize=8)
            ax.set_title(f"{TITLE[c]}\n{REG[r]}", fontsize=9)
    fig.suptitle(f"Confusion matrices at the frozen dev threshold (seed {SEED})")
    fig.tight_layout()
    fig.savefig(fig_dir / "final_confusion.png", dpi=200)
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "confusion_matrices.csv", index=False)
    print("saved final_roc.png, final_confusion.png, outputs/confusion_matrices.csv")


if __name__ == "__main__":
    main()
