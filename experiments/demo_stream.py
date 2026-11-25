#!/usr/bin/env python3
"""Streaming demo: feed a recording through the detector window by window.

Reads any audio file (WAV/FLAC, any sample rate, stereo is mixed to mono), optionally
simulates the telephone channel, then slides a 1.0 s window with a 0.5 s hop and scores
each window with the FROZEN models (the 5 CNN seeds are averaged). For every window it
prints the margin above the model's frozen threshold (positive = looks spoofed), the
state of the alarm rule, and the compute time. When the rule fires it prints the
warning text from the paper. Research demonstration only: no bank integration and no
automatic call termination.

Single call:
    python3 experiments/demo_stream.py data/raw/banking_vishing/audio/spoof/lot11_en_fraud_1040.wav \
        --regime aug --condition telephone_codec
    python3 experiments/demo_stream.py call.wav --rule mean --realtime

Whole folder(s), models loaded once, one summary line per call and per label:
    python3 experiments/demo_stream.py --batch data/raw/banking_vishing/audio \
        --regime aug --condition telephone_codec --out-csv outputs/vishguard_alarms.csv
  (the label comes from the parent folder name: 'spoof' -> 1, 'bonafide' -> 0)

Rules (all on the per-window margin):
    consecutive   alarm after --k consecutive windows above threshold   (default k = 3)
    mean          alarm when the running mean margin >= --tau            (default 0)
    vote          alarm when >= 50% of the windows so far are above threshold (after --k windows)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "features"):
    sys.path.insert(0, str(p))

from stream_common import (HOP_SEC, SR, WARNING, WIN_SEC, Scorer, discover,  # noqa: E402
                           duration_at)

AUDIO_EXT = {".wav", ".flac"}
LABELS = {"spoof": 1, "bonafide": 0}


def load_call(path, condition):
    import librosa
    a, sr = librosa.load(path, sr=None, mono=True)
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    if condition != "clean":
        from preprocessing.audio_loader import apply_alaw, normalize, telephone_filter
        a = telephone_filter(a, SR)
        a = librosa.resample(a, orig_sr=SR, target_sr=8000)
        a = normalize(a)
        if condition == "telephone_codec":
            a = apply_alaw(a)
        a = librosa.resample(a, orig_sr=8000, target_sr=SR)
    return a.astype(np.float32)


def run_call(audio, scorers, fx, rule, kreq, tau, verbose=True,
             realtime=False, stop_at_alarm=False):
    """Slide the window over one recording. Returns a dict, or None if it is too short."""
    n, hop = int(WIN_SEC * SR), int(HOP_SEC * SR)
    starts = list(range(0, len(audio) - n + 1, hop))
    if not starts:
        return None
    if verbose:
        print(" window  time   margin  above  state           compute")

    margins, run, alarm_at, comp = [], 0, None, []
    for k, s in enumerate(starts):
        t0 = time.perf_counter()
        w = fx.peak_norm(audio[s:s + n])                 # causal: only this window
        mf = fx.mfcc_stats(w)[None]
        mel = fx.log_mel(w, fx.frames_for(WIN_SEC))[None]
        m = float(np.mean([sc.score(mf, mel)[0] - sc.thr for sc in scorers]))
        ms = (time.perf_counter() - t0) * 1000.0
        comp.append(ms)
        margins.append(m)
        above = m >= 0
        run = run + 1 if above else 0
        if rule == "consecutive":
            fired, state = run >= kreq, f"run {run}/{kreq}"
        elif rule == "mean":
            rm = float(np.mean(margins))
            fired, state = rm >= tau, f"mean {rm:+.2f}"
        else:
            frac = float(np.mean(np.array(margins) >= 0))
            fired, state = (len(margins) >= kreq and frac >= 0.5), f"{frac * 100:.0f}% above"
        t_end = duration_at(k + 1)
        flag = "ALARM" if fired and alarm_at is None else ""
        if verbose:
            print(f" {k + 1:5d} {t_end:5.1f}s {m:+7.2f}  {'yes' if above else ' no'}   "
                  f"{state:<14s}  {ms:5.1f} ms  {flag}")
        if fired and alarm_at is None:
            alarm_at = t_end
            if verbose:
                print(f"\n>>> {WARNING}\n    (raised after {alarm_at:.1f} s of audio)\n")
            if stop_at_alarm:
                break
        if realtime:
            time.sleep(HOP_SEC)

    return {"margins": margins, "alarm_at": alarm_at, "comp": comp,
            "seconds": len(audio) / SR}


def find_files(roots):
    files = []
    for r in roots:
        r = Path(r)
        if r.is_file():
            files.append(r)
        elif r.is_dir():
            files += sorted(p for p in r.rglob("*") if p.suffix.lower() in AUDIO_EXT)
        else:
            sys.exit(f"Not found: {r}")
    return files


def batch(files, scorers, fx, args):
    import pandas as pd
    rows, all_comp = [], []
    for i, f in enumerate(files, 1):
        audio = load_call(str(f), args.condition)
        res = run_call(audio, scorers, fx, args.rule, args.k, args.tau, verbose=False)
        label = LABELS.get(f.parent.name.lower(), -1)
        if res is None:
            print(f"[{i}/{len(files)}] {f.name}: shorter than one window, skipped")
            continue
        all_comp += res["comp"]
        rows.append({"file": f.name, "folder": f.parent.name, "label": label,
                     "seconds": round(res["seconds"], 1), "windows": len(res["margins"]),
                     "alarm": res["alarm_at"] is not None, "alarm_at_s": res["alarm_at"],
                     "mean_margin": float(np.mean(res["margins"])),
                     "frac_above": float(np.mean(np.array(res["margins"]) >= 0))})
        r = rows[-1]
        at = f"{r['alarm_at_s']:.1f}s" if r["alarm"] else "-"
        print(f"[{i}/{len(files)}] {f.parent.name}/{f.name}: alarm={'yes' if r['alarm'] else 'no '} "
              f"at={at:>6s} mean_margin={r['mean_margin']:+.2f}", flush=True)

    if not rows:
        sys.exit("No calls were scored.")
    df = pd.DataFrame(rows)
    print(f"\n=== {args.kind} ({args.regime}-trained), condition={args.condition}, rule={args.rule} ===")
    for folder, g in df.groupby("folder"):
        at = g.alarm_at_s.dropna()
        print(f"{folder:10s} alarmed {int(g.alarm.sum())}/{len(g)}"
              f" | median time to alarm {at.median():.1f} s" if len(at) else
              f"{folder:10s} alarmed {int(g.alarm.sum())}/{len(g)} | no alarms",
              f"| mean margin {g.mean_margin.mean():+.2f} | windows above threshold {g.frac_above.mean() * 100:.0f}%")
    lab = df[df.label >= 0]
    if lab.label.nunique() == 2:
        from sklearn.metrics import roc_auc_score
        print(f"\ncall-level AUC (mean window margin, label from folder name): "
              f"{roc_auc_score(lab.label, lab.mean_margin):.3f}  "
              f"(n = {len(lab)}; 0.5 = chance)")
    c = np.array(all_comp)
    print(f"compute per window: median {np.median(c):.1f} ms, 95th pct {np.percentile(c, 95):.1f} ms "
          f"(hop {HOP_SEC * 1000:.0f} ms)")
    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", help="a single recording (omit when using --batch)")
    ap.add_argument("--batch", nargs="+", metavar="PATH",
                    help="files or folders to score in one go (models are loaded once)")
    ap.add_argument("--out-csv", default="", help="write the batch results to this CSV")
    ap.add_argument("--regime", choices=["clean", "aug"], default="aug")
    ap.add_argument("--kind", default="cnn",
                    choices=["cnn", "logistic_regression", "random_forest", "svm"])
    ap.add_argument("--seed", type=int, default=None, help="use one CNN seed instead of the average")
    ap.add_argument("--condition", choices=["clean", "telephone", "telephone_codec"],
                    default="clean")
    ap.add_argument("--rule", choices=["consecutive", "mean", "vote"], default="consecutive")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--cnn-dir", default="outputs/seeds5/models")
    ap.add_argument("--base-dir", default="outputs/models")
    ap.add_argument("--realtime", action="store_true", help="sleep one hop between windows")
    ap.add_argument("--stop-at-alarm", action="store_true")
    a = ap.parse_args()

    if bool(a.audio) == bool(a.batch):
        sys.exit("Give either one audio file or --batch PATH [PATH ...].")

    found = {k: p for k, p in discover([ROOT / a.cnn_dir, ROOT / a.base_dir]).items()
             if k[0] == a.regime and k[1] == a.kind and (a.seed is None or k[2] == a.seed)}
    if not found:
        sys.exit(f"No saved '{a.kind}' models for regime '{a.regime}'"
                 + (f" and seed {a.seed}" if a.seed is not None else ""))
    scorers = [Scorer(p) for p in found.values()]
    import extract_features as fx

    if a.batch:
        files = find_files(a.batch)
        if not files:
            sys.exit("No .wav/.flac files found.")
        print(f"{len(files)} recordings | condition={a.condition} | "
              f"{len(scorers)} x {a.kind} ({a.regime}-trained) | rule={a.rule}\n")
        batch(files, scorers, fx, a)
        return

    audio = load_call(a.audio, a.condition)
    print(f"{Path(a.audio).name}: {len(audio) / SR:.1f} s | condition={a.condition} | "
          f"{len(scorers)} x {a.kind} ({a.regime}-trained) | rule={a.rule}\n")
    res = run_call(audio, scorers, fx, a.rule, a.k, a.tau, verbose=True,
                   realtime=a.realtime, stop_at_alarm=a.stop_at_alarm)
    if res is None:
        sys.exit(f"Recording is shorter than one {WIN_SEC:.0f} s window.")

    comp, margins = res["comp"], res["margins"]
    print(f"\nwindows scored: {len(margins)} | mean compute {np.mean(comp):.1f} ms/window "
          f"(max {np.max(comp):.1f} ms) vs hop {HOP_SEC * 1000:.0f} ms")
    print("result:", f"WARNING raised at {res['alarm_at']:.1f} s" if res["alarm_at"] is not None
          else "no warning raised")


if __name__ == "__main__":
    main()
