#!/usr/bin/env python3
"""One-time feature extraction on top of the repo's own loaders.

Reads the repo metadata, loads audio with ``preprocessing.audio_loader``
(ASVspoof) and ``preprocessing.vishguard_audio_loader`` (VISHGUARD), and
caches features to ``data/processed/`` so every later experiment trains on
the cached arrays only (no audio decoding, no telephone filtering).

Per cropped window it stores:
    mfcc : (N, 120)        MFCC(20) + delta + delta-delta, mean+std over time
    mel  : (N, 64, T)      per-window z-scored log-Mel (float16), T depends on window
    y    : (N,)            0 = bonafide, 1 = spoof

All conditions are brought to the SAME representation: telephone audio is
resampled back to 16 kHz after the 300-3400 Hz band-limit / A-law step, so
features have identical shape and mel-bin meaning across conditions.
All audio is peak-normalised to 0.95 so both datasets get identical levels.

Run from anywhere (the script chdirs to the repo root):

    python features/extract_features.py asvspoof            # train/dev/eval subsets
    python features/extract_features.py vishguard           # 40 calls, 1 s windows
"""

import argparse
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                       # audio_loader uses a relative data/raw path
sys.path.insert(0, str(ROOT))

META = ROOT / "data" / "metadata"
OUT = ROOT / "data" / "processed"

# ---- feature settings (same for every dataset / condition) ----
SR = 16000
N_FFT, HOP, WIN = 512, 160, 400
N_MELS, N_MFCC, FMAX = 64, 20, 8000
TRIM_DB = 30
LABEL = {"bonafide": 0, "spoof": 1}


# ============================================================
# FEATURES
# ============================================================

def peak_norm(a):
    p = np.max(np.abs(a))
    return (a / p * 0.95).astype(np.float32) if p > 0 else a.astype(np.float32)


def to_16k(a, sr):
    return a if sr == SR else librosa.resample(a, orig_sr=sr, target_sr=SR)


def frames_for(window_sec):
    return 1 + int(round(window_sec * SR)) // HOP


def mfcc_stats(a):
    m = librosa.feature.mfcc(y=a, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT,
                             hop_length=HOP, win_length=WIN,
                             n_mels=N_MELS, fmax=FMAX)
    feats = np.vstack([m, librosa.feature.delta(m),
                       librosa.feature.delta(m, order=2)]).T
    return np.concatenate([feats.mean(0), feats.std(0)]).astype(np.float32)


def log_mel(a, n_frames):
    m = librosa.feature.melspectrogram(y=a, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       win_length=WIN, n_mels=N_MELS, fmax=FMAX)
    m = librosa.power_to_db(m, ref=np.max)
    m = (m - m.mean()) / (m.std() + 1e-8)
    if m.shape[1] >= n_frames:
        m = m[:, :n_frames]
    else:
        m = np.pad(m, ((0, 0), (0, n_frames - m.shape[1])))
    return m.astype(np.float16)


def crop(a, window_sec, mode):
    n = int(round(window_sec * SR))
    if len(a) < n:
        return None
    s = 0 if mode == "start" else (len(a) - n) // 2
    return a[s:s + n]


def window_name(w):
    return f"w{w:g}"


# ============================================================
# ASVSPOOF
# ============================================================

def sample_split(df, cap, balanced, seed=42):
    """Subsample one split. balanced=True -> equal bonafide/spoof."""
    if cap is None or cap >= len(df):
        return df
    if balanced:
        per = cap // 2
        parts = [g.sample(n=min(per, len(g)), random_state=seed)
                 for _, g in df.groupby("label")]
    else:
        parts = [g.sample(n=max(1, min(len(g), round(cap * len(g) / len(df)))),
                          random_state=seed)
                 for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def process_asv_file(file, split, label, system_id,
                     conditions, windows, crop_mode, trim):
    from preprocessing.audio_loader import load_audio
    out = {}
    for cond in conditions:
        try:
            a, sr = load_audio(file, split, condition=cond)
        except Exception:
            continue
        a = peak_norm(to_16k(a, sr))
        if trim:
            a, _ = librosa.effects.trim(a, top_db=TRIM_DB)
        for w in windows:
            seg = crop(a, w, crop_mode)
            if seg is None:
                continue
            out[(cond, w)] = (mfcc_stats(seg), log_mel(seg, frames_for(w)))
    return file, label, system_id, out


def run_asvspoof(args):
    df = pd.read_csv(META / "asvspoof2019_la_audio_stats.csv")
    caps = {"train": args.max_train, "dev": args.max_dev, "eval": args.max_eval}
    out_dir = OUT / "asvspoof"
    out_dir.mkdir(parents=True, exist_ok=True)

    from preprocessing.audio_loader import find_audio_file

    def has_audio(r):
        try:
            find_audio_file(r.file, r.split)
            return True
        except FileNotFoundError:
            return False

    for split in args.splits:
        sub = df[df["split"] == split]
        present = np.array([has_audio(r) for r in sub.itertuples(index=False)])
        print(f"\n[{split}] audio found for {present.sum()}/{len(sub)} files",
              flush=True)
        if present.sum() == 0:
            sys.exit(f"No {split} audio found. Put the ASVspoof 2019 LA FLACs "
                     f"under {ROOT / 'data/raw/LA/LA'}/ASVspoof2019_LA_{split}/flac/")
        sub = sub[present]          # work with whatever is actually on disk
        if not args.full:
            sub = sample_split(sub, caps[split], balanced=(split == "train"))
        print(f"[{split}] using {len(sub)} files  "
              f"{sub['label'].value_counts().to_dict()}", flush=True)

        results = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(process_asv_file)(r.file, r.split, r.label, r.system_id,
                                      args.conditions, args.windows,
                                      args.crop, args.trim)
            for r in sub.itertuples(index=False))

        for cond in args.conditions:
            for w in args.windows:
                items = [(f, l, s, o[(cond, w)]) for f, l, s, o in results
                         if (cond, w) in o]
                if not items:
                    continue
                path = out_dir / f"{split}_{cond}_{window_name(w)}.npz"
                np.savez_compressed(
                    path,
                    ids=np.array([i[0] for i in items]),
                    y=np.array([LABEL[i[1]] for i in items], dtype=np.int64),
                    system_id=np.array([i[2] for i in items]),
                    mfcc=np.stack([i[3][0] for i in items]),
                    mel=np.stack([i[3][1] for i in items]),
                )
                print(f"  saved {path.name}  N={len(items)}", flush=True)


# ============================================================
# VISHGUARD
# ============================================================

def run_vishguard(args):
    from preprocessing.vishguard_audio_loader import load_vishguard_audio
    from preprocessing.audio_loader import apply_alaw   # real G.711 A-law

    from preprocessing.vishguard_audio_loader import find_audio_file

    meta = pd.read_csv(META / "vishguard_short_window_metadata.csv")
    calls = meta[["file", "label"]].drop_duplicates()
    missing = []
    for r in calls.itertuples(index=False):
        try:
            find_audio_file(r.file, r.label)
        except FileNotFoundError:
            missing.append(f"{r.label}/{r.file}")
    print(f"VISHGUARD audio found for {len(calls) - len(missing)}/{len(calls)} calls")
    if missing:
        sys.exit("Missing VISHGUARD WAVs, e.g. " + ", ".join(missing[:3]) +
                 f"\nExpected under {ROOT / 'data/raw/banking_vishing/audio'}/{{spoof,bonafide}}/. "
                 "preprocessing/extract_vishguard_selected.py fetches them.")
    out_dir = OUT / "vishguard"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions:
        mfcc, mel, y, call_id, start = [], [], [], [], []
        for (file, label), g in meta.groupby(["file", "label"]):
            # the VISHGUARD loader's codec branch is not real A-law, so load
            # the telephone signal and apply the ASVspoof A-law ourselves
            loader_cond = "telephone" if cond == "telephone_codec" else cond
            a, sr = load_vishguard_audio(file, label, condition=loader_cond)
            if cond == "telephone_codec":
                a = apply_alaw(a)
            a = peak_norm(to_16k(a, sr))
            for r in g.itertuples(index=False):
                s = int(round(r.start_sec * SR))
                e = s + int(round(r.window_sec * SR))
                seg = a[s:e]
                if len(seg) < e - s:
                    continue
                mfcc.append(mfcc_stats(seg))
                mel.append(log_mel(seg, frames_for(r.window_sec)))
                y.append(LABEL[label])
                call_id.append(r.call_id)
                start.append(r.start_sec)
        path = out_dir / f"{cond}_w1.npz"
        np.savez_compressed(path, mfcc=np.stack(mfcc), mel=np.stack(mel),
                            y=np.array(y), call_id=np.array(call_id),
                            start=np.array(start))
        print(f"saved {path.name}  windows={len(y)}  "
              f"calls={len(set(call_id))}", flush=True)


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    conds = ["clean", "telephone", "telephone_codec"]

    a = sub.add_parser("asvspoof")
    a.add_argument("--splits", nargs="+", default=["train", "dev", "eval"])
    a.add_argument("--conditions", nargs="+", default=conds, choices=conds)
    a.add_argument("--windows", nargs="+", type=float,
                   default=[3.0, 2.0, 1.5, 1.0, 0.75, 0.5])
    a.add_argument("--crop", choices=["center", "start"], default="center",
                   help="'start' matches the repo's start_sec=0 metadata but "
                        "leaks leading-silence length")
    a.add_argument("--trim", action="store_true",
                   help="trim leading/trailing silence before cropping")
    a.add_argument("--max-train", type=int, default=4000,
                   help="balanced bonafide/spoof subsample")
    a.add_argument("--max-dev", type=int, default=3000)
    a.add_argument("--max-eval", type=int, default=8000)
    a.add_argument("--full", action="store_true", help="ignore caps")
    a.add_argument("--jobs", type=int, default=-1)

    v = sub.add_parser("vishguard")
    v.add_argument("--conditions", nargs="+", default=conds, choices=conds)

    args = p.parse_args()
    run_asvspoof(args) if args.cmd == "asvspoof" else run_vishguard(args)


if __name__ == "__main__":
    main()
