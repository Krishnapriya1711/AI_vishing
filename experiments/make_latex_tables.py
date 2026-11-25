#!/usr/bin/env python3
"""Write LaTeX tables from the result CSVs -> outputs/tables/*.tex (use \\input{...})."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
         "svm": "SVM (RBF)", "cnn": "Mel-CNN"}
CONDS = ["clean", "telephone", "telephone_codec"]


def fmt(m, sd=None, scale=100, nd=1):
    if m is None or not np.isfinite(m):
        return "--"
    s = f"{m * scale:.{nd}f}"
    return s if sd is None or not np.isfinite(sd) else f"{s}$\\pm${sd * scale:.{nd}f}"


def from_seeds(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return None, None
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else None)


def wrap(caption, label, colspec, header, rows):
    return "\n".join([r"\begin{table}[htbp]", rf"\caption{{{caption}}}", r"\begin{center}",
                      r"\footnotesize\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{colspec}}}",
                      r"\toprule", header, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
                      rf"\label{{{label}}}", r"\end{center}", r"\end{table}", ""])


def main():
    first = pd.read_csv(OUT / "final_results.csv")
    s1 = pd.read_csv(OUT / "seeds5" / "final_results_per_seed.csv")
    s05 = pd.read_csv(OUT / "seeds5_w05" / "final_results_per_seed.csv")
    cnn = {1.0: s1[s1.model == "cnn"], 0.5: s05[s05.model == "cnn"]}

    # ---- Table 1: EER by model / training regime / eval condition (1 s) ----
    rows = []
    for model in NAMES:
        cells = []
        for regime in ("clean", "aug"):
            for cond in CONDS:
                if model == "cnn":
                    d = cnn[1.0]
                    m, sd = from_seeds(d[(d.regime == regime) & (d.eval_condition == cond)]["eer"])
                else:
                    d = first[(first.model == model) & (first.regime == regime) &
                              (first.eval_condition == cond) & (first.window == 1.0)]
                    m = d.eer.iloc[0] if len(d) else None
                    sd = d.eer_seed_std.iloc[0] if len(d) and d.n_seeds.iloc[0] > 1 else None
                cells.append(fmt(m, sd))
        rows.append(NAMES[model] + " & " + " & ".join(cells) + r" \\")
    hdr = (r"& \multicolumn{3}{c}{Trained on clean} & \multicolumn{3}{c}{Trained on clean+tel.+codec} \\" "\n"
           r"Model & Clean & Tel. & Tel.+A-law & Clean & Tel. & Tel.+A-law \\")
    (TAB / "table_main.tex").write_text(wrap(
        "EER (\\%) on ASVspoof 2019 LA eval, 1\\,s windows, threshold frozen on dev. Mel-CNN: mean$\\pm$std over 5 "
        "training seeds; RF: 3 seeds; Logistic Regression and SVM are deterministic.",
        "tab:main", "lcccccc", hdr, rows))

    # ---- Table 2: window length (CNN only) ----
    rows = []
    for regime, lab in (("clean", "Clean-trained"), ("aug", "Channel-aware")):
        cells = []
        for w in (1.0, 0.5):
            for cond in ("clean", "telephone"):
                d = cnn[w]
                m, sd = from_seeds(d[(d.regime == regime) & (d.eval_condition == cond)]["eer"])
                cells.append(fmt(m, sd))
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    hdr = (r"& \multicolumn{2}{c}{1.0\,s} & \multicolumn{2}{c}{0.5\,s} \\" "\n"
           r"Mel-CNN & Clean & Tel. & Clean & Tel. \\")
    (TAB / "table_window.tex").write_text(wrap(
        "Mel-CNN EER (\\%) for 1.0\\,s and 0.5\\,s windows (mean$\\pm$std, 5 seeds).",
        "tab:window", "lcccc", hdr, rows))

    # ---- Table 3: VISHGUARD call-level AUC (exploratory) ----
    vf = pd.read_csv(OUT / "final_vishguard_results.csv")
    vs = pd.read_csv(OUT / "seeds5" / "final_vishguard_results_per_seed.csv")
    rows = []
    for regime in ("clean", "aug"):
        for model in NAMES:
            cells = []
            for cond in CONDS:
                if model == "cnn":
                    d = vs[(vs.model == "cnn") & (vs.regime == regime) &
                           (vs.condition == cond) & (vs.level == "call")]
                    m, sd = from_seeds(d["auc"])
                else:
                    d = vf[(vf.model == model) & (vf.regime == regime) &
                           (vf.condition == cond) & (vf.level == "call")]
                    m, sd = (d.auc.iloc[0] if len(d) else None), None
                cells.append(fmt(m, sd, scale=1, nd=2))
            rows.append(f"{regime} & {NAMES[model]} & " + " & ".join(cells) + r" \\")
    hdr = r"Training & Model & Clean & Tel. & Tel.+A-law \\"
    (TAB / "table_vishguard.tex").write_text(wrap(
        "Exploratory external evaluation on 40 VISHGUARD calls: call-level ROC-AUC (chance = 0.50). "
        "Labels are call intent, not voice authenticity.",
        "tab:vish", "llccc", hdr, rows))
    print("wrote", ", ".join(p.name for p in sorted(TAB.glob("*.tex"))))


if __name__ == "__main__":
    main()
