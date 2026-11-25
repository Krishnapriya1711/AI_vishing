#!/usr/bin/env python3
"""Score DEEP-VOICE (RVC voice conversion) windows with FROZEN ASVspoof-trained models.

Reuses the repo's feature code (features/extract_features.py) and channel simulation
(preprocessing/audio_loader.py). No retraining, no threshold tuning on this data.

Reports per model / condition:
  eer, auc     window level, with 95% CI from a FILE-CLUSTER bootstrap
               (real files and fake files resampled separately)
  tpr_fake     share of fake windows flagged at the frozen dev threshold
  fpr_real     share of real windows wrongly flagged at that threshold
  bal_acc      balanced accuracy at that threshold
and per-target-speaker flag rates (fakes) / per-speaker false-alarm rates (real).

Usage (from repo root):
    python3 experiments/eval_deepvoice.py --limit 400 --n-boot 20     # smoke test
    python3 experiments/eval_deepvoice.py                             # full run
"""
import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "features"):
    sys.path.insert(0, str(p))

from run_final_experiments import (CnnModel, LightweightCNN, SkModel,   # noqa: E402
                                   eer_and_threshold, metrics_at)

CONDS = ["clean", "telephone", "telephone_codec"]
KINDS = {"cnn", "logistic_regression", "random_forest", "svm"}


# ---------------------------------------------------------------- features
def _features(wav_i16, cond, root):
    """Same chain as preprocessing/audio_loader.load_audio + features/extract_features."""
    import sys as _sys
    for q in (root, root + "/features"):
        if q not in _sys.path:
            _sys.path.insert(0, q)
    import librosa
    import extract_features as fx
    from preprocessing.audio_loader import apply_alaw, normalize, telephone_filter

    a = wav_i16.astype(np.float32) / 32767.0
    if cond != "clean":
        a = telephone_filter(a, 16000)
        a = librosa.resample(a, orig_sr=16000, target_sr=8000)
        a = normalize(a)
        if cond == "telephone_codec":
            a = apply_alaw(a)
        a = fx.to_16k(a, 8000)
    a = fx.peak_norm(a)
    return fx.mfcc_stats(a), fx.log_mel(a, fx.frames_for(1.0))


def wav_signature(wav):
    """Cheap fingerprint of the audio, so a stale feature cache is never reused."""
    return int(np.sum(wav[:, ::97], dtype=np.int64))


def get_feats(cond, wav, y, sig, args, cache_dir):
    tag = f"_n{args.limit}" if args.limit else ""
    p = cache_dir / f"deepvoice_{cond}_w1_feats{tag}.npz"
    if p.exists() and not args.recompute:
        d = np.load(p)
        if len(d["y"]) == len(y) and int(d["sig"]) == sig:
            return {"mfcc": d["mfcc"], "mel": d["mel"], "y": d["y"]}
        print(f"  cache for {cond} does not match the audio -> recomputing", flush=True)
    print(f"  extracting {cond} features for {len(wav)} windows ...", flush=True)
    out = Parallel(n_jobs=args.jobs, batch_size=32, verbose=5)(
        delayed(_features)(w, cond, str(ROOT)) for w in wav)
    d = {"mfcc": np.stack([o[0] for o in out]),
         "mel": np.stack([o[1] for o in out]), "y": y}
    np.savez_compressed(p, sig=np.array(sig), **d)
    return d


# ------------------------------------------------------------------ models
def _torch_load(path, dev):
    try:
        return torch.load(path, map_location=dev, weights_only=False)
    except TypeError:                      # older torch has no weights_only argument
        return torch.load(path, map_location=dev)


def discover(dirs):
    found = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*_w1_*")):
            if p.suffix not in (".pt", ".joblib"):
                continue
            regime, rest = p.stem.split("_w1_", 1)
            m = re.search(r"_s(\d+)$", rest)
            seed = int(m.group(1)) if m else -1
            kind = re.sub(r"_s\d+$", "", rest)
            if kind not in KINDS or (kind == "cnn" and seed == -1):
                continue
            found[(regime, kind, seed)] = p
    return found


def load_model(path):
    if path.suffix == ".pt":
        m = CnnModel(seed=0)
        ck = _torch_load(path, m.dev)
        m.net = LightweightCNN().to(m.dev)
        m.net.load_state_dict(ck["state"])
        m.net.eval()
        return m, float(ck["thr"])
    ck = joblib.load(path)
    m = SkModel(ck["kind"], 0)
    m.sc, m.m = ck["sc"], ck["m"]
    return m, float(ck["thr"])


# --------------------------------------------------------------- bootstrap
def make_boots(files, y, n_boot, seed):
    """File-cluster bootstrap: resample real files and fake files separately."""
    rng = np.random.default_rng(seed)
    uniq = list(pd.unique(files))
    idx_of = {f: np.flatnonzero(files == f) for f in uniq}
    real = [f for f in uniq if y[idx_of[f][0]] == 0]
    fake = [f for f in uniq if y[idx_of[f][0]] == 1]
    if not real or not fake:
        sys.exit("Need windows from both real and fake files (try a larger --limit).")
    boots = []
    for _ in range(n_boot):
        pick = list(rng.choice(real, len(real))) + list(rng.choice(fake, len(fake)))
        boots.append(np.concatenate([idx_of[f] for f in pick]))
    return boots


def evaluate(model, thr, d, boots):
    s = model.score(d)
    y = d["y"]
    r = metrics_at(y, s, thr)
    pred = s >= thr
    r["tpr_fake"] = float(pred[y == 1].mean())
    r["fpr_real"] = float(pred[y == 0].mean())
    eers = [eer_and_threshold(y[b], s[b])[0] for b in boots]
    aucs = [roc_auc_score(y[b], s[b]) for b in boots]
    r.update(eer_lo=float(np.percentile(eers, 2.5)), eer_hi=float(np.percentile(eers, 97.5)),
             auc_lo=float(np.percentile(aucs, 2.5)), auc_hi=float(np.percentile(aucs, 97.5)))
    return s, pred, r


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/deepvoice/deepvoice_w1.npz")
    ap.add_argument("--cnn-dir", default="outputs/seeds5/models")
    ap.add_argument("--base-dir", default="outputs/models")
    ap.add_argument("--out", default="outputs/deepvoice")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="random subset of windows (smoke test)")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "data" / "processed" / "deepvoice"

    z = dict(np.load(ROOT / args.data, allow_pickle=True))      # read the .npz once
    sel = np.arange(len(z["y"]))
    if args.limit:
        sel = np.sort(np.random.default_rng(0).choice(sel, args.limit, replace=False))
    wav, y = z["wav"][sel], z["y"][sel].astype(np.int64)
    fname = np.asarray(z["fname"][sel]).astype(str)
    src = np.asarray(z["src"][sel]).astype(str)
    tgt = np.asarray(z["tgt"][sel]).astype(str)
    sig = wav_signature(wav)
    print(f"windows: {len(y)} | real {int((y == 0).sum())} | fake {int((y == 1).sum())}")

    feats = {c: get_feats(c, wav, y, sig, args, cache_dir) for c in CONDS}
    boots = make_boots(fname, y, args.n_boot, seed=42)

    found = discover([ROOT / args.cnn_dir, ROOT / args.base_dir])
    if not found:
        sys.exit("No saved models found; run experiments/run_final_experiments.py first.")
    print(f"models found: {len(found)}")

    rows, spk_rows = [], []
    for (regime, kind, seed), path in sorted(found.items()):
        model, thr = load_model(path)
        for cond in CONDS:
            s, pred, r = evaluate(model, thr, feats[cond], boots)
            rows.append({"regime": regime, "model": kind, "seed": seed,
                         "condition": cond, "threshold": thr, **r})
            for t in np.unique(tgt[y == 1]):
                m = (y == 1) & (tgt == t)
                spk_rows.append({"regime": regime, "model": kind, "seed": seed, "condition": cond,
                                 "role": "fake_target", "speaker": t, "n": int(m.sum()),
                                 "flag_rate": float(pred[m].mean())})
            for sp in np.unique(src[y == 0]):
                m = (y == 0) & (src == sp)
                spk_rows.append({"regime": regime, "model": kind, "seed": seed, "condition": cond,
                                 "role": "real_speaker", "speaker": sp, "n": int(m.sum()),
                                 "flag_rate": float(pred[m].mean())})
        print(f"  done {regime:5s} {kind:20s} seed {seed}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "deepvoice_results_per_model.csv", index=False)
    spk = pd.DataFrame(spk_rows)
    spk.to_csv(out / "deepvoice_per_speaker.csv", index=False)

    g = res.groupby(["regime", "model", "condition"])
    summ = g[["eer", "auc", "bal_acc", "tpr_fake", "fpr_real", "eer_lo", "eer_hi"]].mean()
    summ["eer_seed_std"] = g["eer"].std().fillna(0.0)
    summ["auc_seed_std"] = g["auc"].std().fillna(0.0)
    summ["n_models"] = g.size()
    summ = summ.reset_index()
    summ.to_csv(out / "deepvoice_summary.csv", index=False)

    pd.set_option("display.width", 200)
    for regime in summ["regime"].unique():
        t = summ[summ.regime == regime].copy()
        for c in ("eer", "auc", "bal_acc", "tpr_fake", "fpr_real", "eer_seed_std"):
            t[c] = t[c].round(3)
        print(f"\n=== DEEP-VOICE, regime={regime} "
              "(EER/AUC window level; flags at the frozen dev threshold) ===")
        print(t[["model", "condition", "eer", "eer_seed_std", "auc", "tpr_fake",
                 "fpr_real", "bal_acc", "n_models"]].to_string(index=False))

    c = spk[(spk.model == "cnn") & (spk.condition == "clean")]
    if len(c):
        print("\n=== CNN, clean audio: share of windows flagged as spoof (mean over seeds) ===")
        print("fake windows, by TARGET speaker (higher = clone caught):")
        print(c[c.role == "fake_target"].pivot_table(index="speaker", columns="regime",
              values="flag_rate", aggfunc="mean").round(2).to_string())
        print("real windows, by speaker (higher = more false alarms):")
        print(c[c.role == "real_speaker"].pivot_table(index="speaker", columns="regime",
              values="flag_rate", aggfunc="mean").round(2).to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
