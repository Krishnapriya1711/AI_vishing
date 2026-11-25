#!/usr/bin/env python3
"""Streaming / early-detection analysis on ASVspoof dev + eval utterances (frozen CNNs).

Window scores come from outputs/seeds5/models/*_w1_cnn_s*.pt. Aggregation rules:
  mean     running mean of window scores
  ma3      mean of the last 3 windows
  vote     fraction of windows above the model's frozen window threshold
  consec2  best score reached by two consecutive windows, max_i min(s_i, s_{i+1})
tau for every (model, rule, observed duration) is the EER point on DEV, frozen for EVAL.
Only utterances with >= dmax seconds after silence trimming are used, so every
duration is measured on the same utterances (counts are printed).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "features"):
    sys.path.insert(0, str(p))

import extract_features as fx                                   # noqa: E402  (chdirs to repo root)
from run_final_experiments import (CnnModel, LightweightCNN,     # noqa: E402
                                   eer_and_threshold, metrics_at)

RULES = ["mean", "ma3", "vote", "consec2"]
LABEL = {"bonafide": 0, "spoof": 1}


def _utt_mels(file, split, cond, kmax, root):
    import sys as _s
    for q in (root, root + "/features"):
        if q not in _s.path:
            _s.path.insert(0, q)
    import librosa
    import extract_features as fx_
    from preprocessing.audio_loader import load_audio
    try:
        a, sr = load_audio(file, split, condition=cond)
    except Exception:
        return file, None
    a = fx_.peak_norm(fx_.to_16k(a, sr))
    a, _ = librosa.effects.trim(a, top_db=fx_.TRIM_DB)
    W, H = 16000, 8000                                           # 1 s window, 0.5 s hop
    n = 0 if len(a) < W else 1 + (len(a) - W) // H
    n = min(n, kmax)
    if n == 0:
        return file, None
    mels = np.stack([fx_.log_mel(a[i * H:i * H + W], fx_.frames_for(1.0)) for i in range(n)])
    return file, mels


def load_cnns(mdir):
    models = []
    for p in sorted(Path(mdir).glob("*_w1_cnn_s*.pt")):
        regime, rest = p.stem.split("_w1_", 1)
        seed = int(rest.rsplit("_s", 1)[1])
        m = CnnModel(seed=0)
        ck = torch.load(p, map_location=m.dev)
        m.net = LightweightCNN().to(m.dev)
        m.net.load_state_dict(ck["state"])
        m.net.eval()
        models.append(dict(regime=regime, seed=seed, model=m, thr=float(ck["thr"])))
    if not models:
        sys.exit(f"No CNN checkpoints in {mdir}")
    return models


def score_split(split, cond, sub, models, kmax, args, out):
    tag = f"_n{args.limit}" if args.limit else ""
    path = out / f"win_scores_{split}_{cond}{tag}.npz"
    if path.exists() and not args.recompute:
        d = np.load(path)
        return d["S"], d["y"]
    files = sub["file"].tolist()
    y = sub["label"].map(LABEL).to_numpy()
    S = np.full((len(models), len(files), kmax), np.nan, np.float32)
    for c0 in range(0, len(files), args.chunk):
        res = Parallel(n_jobs=args.jobs)(
            delayed(_utt_mels)(f, split, cond, kmax, str(ROOT)) for f in files[c0:c0 + args.chunk])
        ui, ki, mels = [], [], []
        for j, (_, m) in enumerate(res):
            if m is None:
                continue
            for k in range(len(m)):
                ui.append(c0 + j); ki.append(k); mels.append(m[k])
        if mels:
            X = np.stack(mels)
            dummy = {"mel": X, "y": np.zeros(len(X), dtype=np.int64)}
            ui, ki = np.array(ui), np.array(ki)
            for mi, mm in enumerate(models):
                S[mi, ui, ki] = mm["model"].score(dummy)
        print(f"  [{split}/{cond}] {min(c0 + args.chunk, len(files))}/{len(files)} utterances", flush=True)
    np.savez_compressed(path, S=S, y=y)
    return S, y


def agg(x, rule, thr_w):
    n = x.shape[1]
    if rule == "mean":
        return x.mean(1)
    if rule == "ma3":
        return x[:, max(0, n - 3):].mean(1)
    if rule == "vote":
        return (x >= thr_w).mean(1)
    if rule == "consec2":
        return x[:, 0] if n == 1 else np.minimum(x[:, :-1], x[:, 1:]).max(1)
    raise ValueError(rule)


def plot(summ, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    s = summ[summ.condition == "clean"]
    regs = sorted(s.regime.unique())
    fig, axes = plt.subplots(1, len(regs), figsize=(5.2 * len(regs), 4), sharey=True, squeeze=False)
    for ax, reg in zip(axes[0], regs):
        for rule in RULES:
            t = s[(s.regime == reg) & (s.rule == rule)].sort_values("duration")
            ax.errorbar(t["duration"].to_numpy(), (t["eer"] * 100).to_numpy(),
                        yerr=(t["eer_std"] * 100).to_numpy(), marker="o", capsize=3, label=rule)
        ax.set_title("Trained on clean audio only" if reg == "clean"
                     else "Trained on clean + telephone + codec")
        ax.set_xlabel("Observed audio (s)")
        ax.set_ylim(0, 60)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Utterance-level EER (%), clean eval")
    axes[0][-1].legend(title="aggregation rule", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "final_early_detection.png", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="outputs/seeds5/models")
    ap.add_argument("--out", default="outputs/streaming")
    ap.add_argument("--conditions", nargs="+", default=["clean", "telephone"],
                    choices=["clean", "telephone", "telephone_codec"])
    ap.add_argument("--dmax", type=float, default=3.0)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="utterances per split (smoke test)")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    kmax = int(round((args.dmax - 1) / 0.5)) + 1
    durs = [1.0 + 0.5 * i for i in range(kmax)]
    models = load_cnns(ROOT / args.models_dir)
    print(f"{len(models)} CNNs | durations {durs} s | {kmax} windows max")

    from preprocessing.audio_loader import find_audio_file
    df = pd.read_csv(ROOT / "data" / "metadata" / "asvspoof2019_la_audio_stats.csv")
    subs = {}
    for split in ("dev", "eval"):
        sub = df[df["split"] == split]
        keep = []
        for r in sub.itertuples(index=False):
            try:
                find_audio_file(r.file, r.split); keep.append(True)
            except FileNotFoundError:
                keep.append(False)
        sub = sub[np.array(keep)]
        if args.limit:
            sub = sub.sample(n=min(args.limit, len(sub)), random_state=0)
        subs[split] = sub.reset_index(drop=True)
        print(f"{split}: {len(sub)} utterances on disk")

    rows, lat_rows = [], []
    for cond in args.conditions:
        Sd, yd = score_split("dev", cond, subs["dev"], models, kmax, args, out)
        Se, ye = score_split("eval", cond, subs["eval"], models, kmax, args, out)
        kd = ~np.isnan(Sd[0]).any(1)
        ke = ~np.isnan(Se[0]).any(1)
        print(f"[{cond}] utterances with >= {args.dmax:g} s: dev {kd.sum()}/{len(kd)} "
              f"(spoof {int(yd[kd].sum())}), eval {ke.sum()}/{len(ke)} (spoof {int(ye[ke].sum())})")
        for mi, mm in enumerate(models):
            xd, ydd, xe, yee = Sd[mi][kd], yd[kd], Se[mi][ke], ye[ke]
            base = dict(regime=mm["regime"], seed=mm["seed"], condition=cond)
            for rule in RULES:
                for d, n in zip(durs, range(1, kmax + 1)):
                    sd = agg(xd[:, :n], rule, mm["thr"])
                    se = agg(xe[:, :n], rule, mm["thr"])
                    _, tau = eer_and_threshold(ydd, sd)              # tau chosen on DEV only
                    rows.append({**base, "rule": rule, "duration": d, "n_windows": n,
                                 **metrics_at(yee, se, tau)})
            # running-mean alarm: first duration at which the mean crosses tau
            tau = eer_and_threshold(ydd, xd.mean(1))[1]
            flag = (np.cumsum(xe, axis=1) / np.arange(1, kmax + 1)) >= tau
            hit, first, fake = flag.any(1), flag.argmax(1), yee == 1
            lat = 1.0 + 0.5 * first
            lat_rows.append({**base, "tau": tau,
                             "det_rate_fake": float(hit[fake].mean()),
                             "det_at_full_fake": float(flag[fake, -1].mean()),
                             "fa_rate_real": float(hit[~fake].mean()),
                             "fa_at_full_real": float(flag[~fake, -1].mean()),
                             "mean_latency_fake_s": float(lat[fake & hit].mean()) if (fake & hit).any() else np.nan})

    res = pd.DataFrame(rows)
    res.to_csv(out / "streaming_rules_per_model.csv", index=False)
    g = res.groupby(["regime", "condition", "rule", "duration"])
    summ = g[["eer", "auc", "bal_acc"]].mean()
    summ["eer_std"] = g["eer"].std().fillna(0.0)
    summ["n_models"] = g.size()
    summ = summ.reset_index()
    summ.to_csv(out / "streaming_rules_summary.csv", index=False)

    lat = pd.DataFrame(lat_rows)
    lat.to_csv(out / "streaming_latency_per_model.csv", index=False)
    lsum = lat.groupby(["regime", "condition"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    lsum.to_csv(out / "streaming_latency_summary.csv", index=False)

    pd.set_option("display.width", 200)
    for cond in args.conditions:
        t = summ[summ.condition == cond].pivot_table(index="duration", columns=["regime", "rule"], values="eer")
        print(f"\n--- utterance-level EER (%) vs observed audio, eval={cond} ---")
        print((t * 100).round(1).to_string())
    print("\n--- running-mean alarm (rates are fractions; latency in s) ---")
    print(lsum.round(3).to_string(index=False))
    plot(summ, ROOT / "outputs" / "figures")
    print(f"\nSaved to {out} and outputs/figures/final_early_detection.png")


if __name__ == "__main__":
    main()
