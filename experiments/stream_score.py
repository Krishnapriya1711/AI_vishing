#!/usr/bin/env python3
"""Score sliding windows (1.0 s window, 0.5 s hop) of ASVspoof utterances with the FROZEN models.

For every utterance in the dev and eval files on disk, under each channel condition,
windows start at t_k = k * 0.5 s and only complete windows are kept (first prediction
after 1 s of audio). Features use the same code as features/extract_features.py. No
silence trimming (a live call contains silence) and no retraining.

Writes (in outputs/streaming/):
    window_scores.npz   per-window metadata + one score array per saved model
    thresholds.json     the frozen window-level threshold of every model
    latency.json        measured feature / model cost per window (milliseconds)

    python3 experiments/stream_score.py                       # all dev + eval files on disk
    python3 experiments/stream_score.py --max-files 300       # smoke test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "features"))

from stream_common import (CONDS, HOP_SEC, SR, WIN_SEC, Scorer, discover)  # noqa: E402


def window_features(file, split, cond, root, max_win, norm):
    """Runs in a worker: load, apply channel, cut windows, extract features."""
    import sys as _sys
    for q in (root, root + "/features"):
        if q not in _sys.path:
            _sys.path.insert(0, q)
    import extract_features as fx
    from preprocessing.audio_loader import load_audio

    try:
        a, sr = load_audio(file, split, condition=cond)
    except Exception:
        return None
    a = fx.to_16k(a, sr)
    if norm == "utterance":
        a = fx.peak_norm(a)
    n, hop = int(WIN_SEC * SR), int(HOP_SEC * SR)
    starts = list(range(0, len(a) - n + 1, hop))[:max_win]
    if not starts:
        return None
    t0 = time.perf_counter()
    segs = [a[s:s + n] for s in starts]
    if norm == "window":                      # strictly causal normalisation
        segs = [fx.peak_norm(x) for x in segs]
    mf = np.stack([fx.mfcc_stats(x) for x in segs])
    mel = np.stack([fx.log_mel(x, fx.frames_for(WIN_SEC)) for x in segs])
    feat_ms = (time.perf_counter() - t0) * 1000.0 / len(starts)
    return mf, mel, feat_ms


def present_files(df, split):
    from preprocessing.audio_loader import find_audio_file
    sub = df[df["split"] == split]
    keep = []
    for r in sub.itertuples(index=False):
        try:
            find_audio_file(r.file, split)
            keep.append(True)
        except FileNotFoundError:
            keep.append(False)
    return sub[np.array(keep)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["dev", "eval"])
    ap.add_argument("--conditions", nargs="+", default=CONDS, choices=CONDS)
    ap.add_argument("--cnn-dir", default="outputs/seeds5/models")
    ap.add_argument("--base-dir", default="outputs/models")
    ap.add_argument("--out", default="outputs/streaming")
    ap.add_argument("--max-files", type=int, default=0, help="cap files per split (smoke test)")
    ap.add_argument("--max-windows", type=int, default=8,
                    help="windows kept per utterance (8 -> first 4.5 s)")
    ap.add_argument("--norm", choices=["utterance", "window"], default="utterance",
                    help="peak normalisation: whole utterance (as in training) or per window (causal)")
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    found = discover([ROOT / args.cnn_dir, ROOT / args.base_dir])
    if not found:
        sys.exit("No saved models found. Expected <regime>_w1_cnn_s<seed>.pt / <regime>_w1_<kind>.joblib")
    scorers = {k: Scorer(p) for k, p in found.items()}
    keys = sorted(scorers)
    print(f"models: {len(keys)}")

    df = pd.read_csv(ROOT / "data" / "metadata" / "asvspoof2019_la_audio_stats.csv")
    meta = {k: [] for k in ["split", "cond", "utt", "widx", "y", "system_id", "fname"]}
    score = {k: [] for k in keys}
    feat_ms, score_ms = [], {k: [] for k in keys}
    utt_counter = 0

    for split in args.splits:
        sub = present_files(df, split)
        if args.max_files:
            sub = sub.sample(n=min(args.max_files, len(sub)), random_state=42)
        print(f"[{split}] {len(sub)} files on disk", flush=True)
        if len(sub) == 0:
            continue
        for cond in args.conditions:
            rows = list(sub.itertuples(index=False))
            for c0 in range(0, len(rows), args.chunk):
                chunk = rows[c0:c0 + args.chunk]
                res = Parallel(n_jobs=args.jobs)(
                    delayed(window_features)(r.file, split, cond, str(ROOT),
                                             args.max_windows, args.norm) for r in chunk)
                mfs, mels, ys, sy, fl, ut, wi = [], [], [], [], [], [], []
                for r, o in zip(chunk, res):
                    if o is None:
                        continue
                    mf, mel, fm = o
                    feat_ms.append(fm)
                    k = len(mf)
                    mfs.append(mf); mels.append(mel)
                    ys += [1 if r.label == "spoof" else 0] * k
                    sy += [r.system_id] * k
                    fl += [r.file] * k
                    ut += [utt_counter] * k
                    wi += list(range(k))
                    utt_counter += 1
                if not mfs:
                    continue
                mf_all, mel_all = np.concatenate(mfs), np.concatenate(mels)
                n = len(mf_all)
                meta["split"] += [split] * n
                meta["cond"] += [cond] * n
                meta["utt"] += ut
                meta["widx"] += wi
                meta["y"] += ys
                meta["system_id"] += sy
                meta["fname"] += fl
                for k_ in keys:
                    t0 = time.perf_counter()
                    score[k_].append(scorers[k_].score(mf_all, mel_all))
                    score_ms[k_].append((time.perf_counter() - t0) * 1000.0 / n)
                print(f"  {split}/{cond}: {min(c0 + args.chunk, len(rows))}/{len(rows)} files",
                      flush=True)

    if not meta["y"]:
        sys.exit("No windows scored: no audio found for the requested splits.")

    # single-window latency (what a live system would see: one window at a time)
    sample_mf, sample_mel = mf_all[:1], mel_all[:1]
    single = {}
    for k_ in keys:
        scorers[k_].score(sample_mf, sample_mel)                 # warm-up
        t0 = time.perf_counter()
        for _ in range(30):
            scorers[k_].score(sample_mf, sample_mel)
        single["|".join(map(str, k_))] = (time.perf_counter() - t0) * 1000.0 / 30
    lat = {"feature_ms_per_window_mean": float(np.mean(feat_ms)),
           "feature_ms_per_window_p95": float(np.percentile(feat_ms, 95)),
           "model_ms_per_window_batch": {"|".join(map(str, k_)): float(np.mean(v))
                                         for k_, v in score_ms.items()},
           "model_ms_single_window": single,
           "algorithmic_latency_s": WIN_SEC, "hop_s": HOP_SEC}
    (out / "latency.json").write_text(json.dumps(lat, indent=2))
    (out / "thresholds.json").write_text(json.dumps(
        {"|".join(map(str, k_)): scorers[k_].thr for k_ in keys}, indent=2))

    arrays = {k: np.array(v) for k, v in meta.items()}
    for k_ in keys:
        arrays["score__" + "|".join(map(str, k_))] = np.concatenate(score[k_])
    np.savez_compressed(out / "window_scores.npz", **arrays)
    print(f"\nSaved {len(arrays['y'])} windows x {len(keys)} models to {out}")
    print(f"feature extraction: {lat['feature_ms_per_window_mean']:.1f} ms/window")


if __name__ == "__main__":
    main()
