#!/usr/bin/env python3
"""Shared helpers: ASVspoof audio index, cached 1 s windows, channel conditions, feature cache.

Reads labels from data/metadata/asvspoof2019_la_audio_stats.csv (columns: file, split, label).
Label values: 'bonafide' -> 0, 'spoof' -> 1.
Audio lives under <asv_root>/ASVspoof2019_LA_{split}/flac/<file_id>.flac.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "features"):
    sys.path.insert(0, str(p))

SR = 16000
CACHE = ROOT / "outputs" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
_IDX = {}

# ---- metadata CSV (labels) ----
_META_CSV = ROOT / "data" / "metadata" / "asvspoof2019_la_audio_stats.csv"


def read_protocol(asv_root, split):
    """Return list of (file_id, is_spoof) for the given split from the metadata CSV.

    asv_root is accepted for API compatibility but not used for label reading;
    it is used only by index_audio() for locating .flac files.
    """
    import pandas as pd
    df = pd.read_csv(_META_CSV)
    sub = df[df["split"] == split]
    ids = sub["file"].astype(str)
    # Strip extension if present
    ids = ids.map(lambda s: Path(s).stem)
    lab = sub["label"].astype(str).str.lower().isin(["spoof", "1", "true"]).astype(int)
    return list(zip(ids.tolist(), lab.tolist()))


def index_audio(asv_root):
    """Build {file_id: Path} for all .flac files under asv_root.

    Uses the known folder structure: <asv_root>/ASVspoof2019_LA_{split}/flac/*.flac.
    Falls back to a full recursive search if that structure is not found.
    """
    k = str(asv_root)
    if k not in _IDX:
        root = Path(asv_root)
        idx = {}
        # Try the known structure first (much faster than a full rglob on 120k files)
        split_dirs = list(root.glob("ASVspoof2019_LA_*/flac"))
        if split_dirs:
            for d in split_dirs:
                for p in d.glob("*.flac"):
                    idx[p.stem] = p
        else:
            # Fallback: recursive search
            for p in root.rglob("*.flac"):
                idx[p.stem] = p
        _IDX[k] = idx
        print(f"index_audio: found {len(idx)} .flac files under {root}", flush=True)
    return _IDX[k]


def load_trimmed(path):
    import librosa
    a, _ = librosa.load(str(path), sr=SR, mono=True)
    a, _ = librosa.effects.trim(a, top_db=30)
    return a.astype(np.float32)


def center_window(a, n=SR):
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    s = (len(a) - n) // 2
    return a[s:s + n]


def _load_center(path):
    return center_window(load_trimmed(path))


def get_windows(asv_root, split, n_per_class, seed=0):
    """Balanced random subset, one center-cropped 1 s window per file.

    Returns (x[N,16000], y[N]) where y: 0=bonafide, 1=spoof.
    Cached in outputs/cache/ so subsequent calls are instant.
    """
    fx_ = CACHE / f"winx_{split}_{n_per_class}_{seed}.npy"
    fy_ = CACHE / f"winy_{split}_{n_per_class}_{seed}.npy"
    if fx_.exists() and fy_.exists():
        return np.load(fx_, mmap_mode="r"), np.load(fy_)

    from joblib import Parallel, delayed
    rows = read_protocol(asv_root, split)
    idx = index_audio(asv_root)

    rng = np.random.default_rng(seed)
    sel = []
    for lab in (0, 1):
        ids = [r for r, l in rows if l == lab and r in idx]
        if len(ids) == 0:
            raise RuntimeError(
                f"No audio found for label={lab} split={split}. "
                f"Check that {asv_root}/ASVspoof2019_LA_{split}/flac/*.flac exists."
            )
        n = min(n_per_class, len(ids))
        sel += [(str(i), lab) for i in rng.choice(ids, n, replace=False)]

    sel = [sel[i] for i in rng.permutation(len(sel))]
    xs = Parallel(n_jobs=-1, backend="multiprocessing", batch_size=16)(
        delayed(_load_center)(idx[fid]) for fid, _ in sel
    )
    x = np.stack(xs).astype(np.float32)
    y = np.array([l for _, l in sel], dtype=np.int64)
    np.save(fx_, x)
    np.save(fy_, y)
    print(f"  get_windows [{split}] n={len(y)}, bonafide={int((y==0).sum())}, spoof={int((y==1).sum())}", flush=True)
    return x, y


# ---------- channel conditions ----------
def apply_condition(a, cond, seed=0):
    """Apply a channel condition to a 1 s waveform at SR=16000.

    cond = base[+n<SNR>]
    base options:
        clean            : no degradation
        telephone        : 300-3400 Hz bandpass -> 8 kHz -> normalise -> 16 kHz
        telephone_codec  : telephone + G.711 A-law
        codec_only       : 8 kHz resample + A-law, no bandpass
        telephone_mulaw  : telephone + G.711 mu-law
    +n10 suffix: add white noise at 10 dB SNR after the channel step.
    """
    import librosa
    from preprocessing.audio_loader import apply_alaw, normalize, telephone_filter

    base, _, noise = cond.partition("+n")
    n0 = len(a)
    x = np.asarray(a, dtype=np.float32)

    if base in ("telephone", "telephone_codec", "telephone_mulaw"):
        x = telephone_filter(x, SR)
    if base != "clean":
        x = librosa.resample(x, orig_sr=SR, target_sr=8000)
        x = normalize(x)
        if base in ("telephone_codec", "codec_only"):
            x = apply_alaw(x)
        elif base == "telephone_mulaw":
            x = np.clip(x, -1, 1)
            x = librosa.mu_expand(
                librosa.mu_compress(x, mu=255, quantize=True),
                mu=255, quantize=True
            )
        x = librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=8000, target_sr=SR)

    if noise:
        rng = np.random.default_rng(seed)
        p = float(np.mean(x ** 2)) + 1e-12
        snr_linear = 10 ** (float(noise) / 10)
        x = x + rng.normal(0, np.sqrt(p / snr_linear), len(x)).astype(np.float32)

    x = np.asarray(x, dtype=np.float32)
    # Restore to original length (channel ops may change length slightly)
    if len(x) < n0:
        x = np.pad(x, (0, n0 - len(x)))
    return x[:n0]


# ---------- hand-crafted features ----------
def spectral_stats(a):
    """24-dim spectral feature vector: mean+std of 12 spectral descriptors."""
    import librosa
    S = np.abs(librosa.stft(a, n_fft=512, hop_length=160))
    rows = [
        librosa.feature.spectral_centroid(S=S, sr=SR),         # 1
        librosa.feature.spectral_bandwidth(S=S, sr=SR),        # 1
        librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85),  # 1
        librosa.feature.spectral_flatness(S=S),                # 1
        librosa.feature.zero_crossing_rate(a, frame_length=512, hop_length=160),  # 1
        librosa.feature.rms(S=S, frame_length=512),            # 1
        librosa.feature.spectral_contrast(S=S, sr=SR, n_bands=5),  # 6
    ]
    M = np.vstack(rows)   # (12, T)
    return np.concatenate([M.mean(1), M.std(1)]).astype(np.float32)   # (24,)


def f0_stats(a):
    """8-dim F0 feature vector from pyin."""
    import librosa
    f0, _vflag, vprob = librosa.pyin(
        a, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=160
    )
    v = f0[~np.isnan(f0)]
    out = np.zeros(8, dtype=np.float32)
    out[4] = len(v) / max(1, len(f0))          # voiced ratio
    out[7] = float(np.nanmean(vprob))
    if len(v) >= 3:
        d = np.diff(np.log(np.maximum(v, 1e-6)))
        out[0] = float(v.mean())
        out[1] = float(v.std())
        out[2] = float(np.median(v))
        out[3] = float(v.max() - v.min())
        out[5] = float(np.abs(d).mean())
        out[6] = float(d.std())
    return np.nan_to_num(out)


def _feat_one(args):
    """Worker function: apply channel + extract one feature kind for one window."""
    w, cond, i, kind = args
    import extract_features as fx
    a = fx.peak_norm(apply_condition(np.asarray(w, dtype=np.float32), cond, i))
    if kind == "mfcc":
        return fx.mfcc_stats(a)
    if kind == "mel":
        return fx.log_mel(a, fx.frames_for(1.0))
    if kind == "spectral":
        return spectral_stats(a)
    # kind == "f0"
    return f0_stats(a)


def get_feats(asv_root, split, n_per_class, cond, kind, seed=0):
    """Extract or load cached features for a balanced subset.

    Returns (X: float32 array, y: int64 array).
    X shape depends on kind:
        mfcc     -> (N, 120)
        mel      -> (N, 64, 101)   float16 stored as float32 on return
        spectral -> (N, 24)
        f0       -> (N, 8)
    """
    fy_ = CACHE / f"winy_{split}_{n_per_class}_{seed}.npy"
    # Include asv_root hash in cache key so different roots don't collide
    root_tag = abs(hash(str(asv_root))) % (10**6)
    f = CACHE / f"feat_{split}_{n_per_class}_{cond}_{kind}_{root_tag}.npy"

    if f.exists() and fy_.exists():
        X = np.load(f)
        y = np.load(fy_)
        return X.astype(np.float32), y

    from joblib import Parallel, delayed
    x, y = get_windows(asv_root, split, n_per_class, seed)
    out = Parallel(n_jobs=-1, backend="multiprocessing", batch_size=32)(
        delayed(_feat_one)((x[i], cond, i, kind)) for i in range(len(x))
    )
    X = np.stack(out).astype(np.float32)
    np.save(f, X)
    print(f"  cached {f.name}  shape={X.shape}", flush=True)
    return X, y


# ---------- metrics ----------
def eer_thr(y, s):
    """Equal error rate and the threshold that achieves it."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2), float(thr[i])


def bal_acc(y, s, thr):
    """Balanced accuracy at a fixed threshold."""
    p = s >= thr
    return float((p[y == 1].mean() + (~p[y == 0]).mean()) / 2)
