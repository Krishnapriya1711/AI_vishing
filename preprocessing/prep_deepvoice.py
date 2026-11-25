#!/usr/bin/env python3
"""Sample 1 s windows from DEEP-VOICE (RVC) recordings -> 16 kHz mono int16 cache."""
import argparse, math, pathlib
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly

SR = 16000


def read_win(f, start, win, pad, sr_in):
    f.seek(start - pad)
    x = f.read(win + 2 * pad, dtype="float32", always_2d=True).mean(axis=1)
    g = math.gcd(SR, sr_in)
    y = resample_poly(x, SR // g, sr_in // g)
    lp = round(pad * SR / sr_in)
    y = y[lp:lp + SR]
    return y if len(y) == SR else None


def parse(stem):
    s = stem.lower().replace("-original", "")
    if "-to-" in s:
        a, b = s.split("-to-", 1)
        return a, b
    return s, s  # real recording: source = target = speaker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/deepvoice/AUDIO")
    ap.add_argument("--out", default="data/processed/deepvoice")
    ap.add_argument("--k-real", type=int, default=400, help="windows per real file")
    ap.add_argument("--k-fake", type=int, default=57, help="windows per fake file")
    ap.add_argument("--seg-sec", type=float, default=10.0)
    ap.add_argument("--min-rms", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    root, out = pathlib.Path(a.root), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [(p, 0) for p in sorted((root / "REAL").glob("*.wav"))] + \
            [(p, 1) for p in sorted((root / "FAKE").glob("*.wav"))]
    if not files:
        raise SystemExit(f"No .wav files found under {root.resolve()} "
                         "(run from the repo root or pass --root)")

    wavs, rows = [], []
    for p, label in files:
        k = a.k_real if label == 0 else a.k_fake
        src, tgt = parse(p.stem)
        with sf.SoundFile(p) as f:
            sr = f.samplerate
            win, pad = sr, int(0.1 * sr)
            n = f.frames // win
            cand = rng.permutation(np.arange(1, n - 1))   # skip first/last second
            got = []
            for c in cand:
                y = read_win(f, int(c) * win, win, pad, sr)
                if y is None or np.sqrt(np.mean(y ** 2)) < a.min_rms:
                    continue
                got.append((int(c), y))
                if len(got) == k:
                    break
        for c, y in sorted(got, key=lambda t: t[0]):
            wavs.append((np.clip(y, -1, 1) * 32767).astype(np.int16))
            rows.append(dict(file=p.stem, y=label, src=src, tgt=tgt, start_s=c,
                             seg=f"{p.stem}_{int(c // a.seg_sec)}"))
        print(f"{p.name:28s} label={label} windows={len(got)}")

    meta = pd.DataFrame(rows)
    np.savez_compressed(out / "deepvoice_w1.npz",
                        wav=np.stack(wavs), y=meta.y.to_numpy(),
                        fname=meta.file.to_numpy(), tgt=meta.tgt.to_numpy(),
                        src=meta.src.to_numpy(), seg=meta.seg.to_numpy(),
                        start_s=meta.start_s.to_numpy())
    meta.to_csv(out / "deepvoice_w1_meta.csv", index=False)
    print("\nwindows:", len(meta), "| real:", int((meta.y == 0).sum()),
          "| fake:", int((meta.y == 1).sum()))


if __name__ == "__main__":
    main()
