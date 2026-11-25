#!/usr/bin/env python3
"""Download ONLY the ASVspoof 2019 LA files the experiments will use.

Reads the remote LA.zip with HTTP range requests (like extract_vishguard_selected.py
does for VISHGUARD), so the 7 GB archive is never downloaded or unzipped:
roughly 15k files (~1-2 GB) with the default caps.

It samples with the same rule and seed as features/extract_features.py, then
writes the FLACs to data/raw/LA/LA/<ASVspoof2019_LA_split>/flac/ where the
repo's loaders expect them. Already-downloaded files are skipped, so it is
safe to stop and rerun.

    pip install remotezip
    python preprocessing/fetch_asvspoof_subset.py --check      # server + layout test
    python preprocessing/fetch_asvspoof_subset.py              # default caps
    python preprocessing/fetch_asvspoof_subset.py --max-train 2000 --max-dev 1000 --max-eval 3000
"""

import argparse
import concurrent.futures as cf
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata" / "asvspoof2019_la_audio_stats.csv"
DEST = ROOT / "data" / "raw" / "LA" / "LA"

# Official Edinburgh DataShare record 10283/3336. If this 404s, open the record
# page in a browser, copy the LA.zip download link and pass it with --url.
DEFAULT_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"

FOLDER = {"train": "ASVspoof2019_LA_train",
          "dev": "ASVspoof2019_LA_dev",
          "eval": "ASVspoof2019_LA_eval"}


def sample_split(df, cap, balanced, seed=42):
    """Same rule as features/extract_features.py."""
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


def split_of(name):
    for split, folder in FOLDER.items():
        if f"/{folder}/" in "/" + name:
            return split
    return None


def build_index(url):
    t = time.time()
    with RemoteZip(url) as rz:
        names = rz.namelist()
    idx = {}
    for n in names:
        if n.lower().endswith(".flac"):
            sp = split_of(n)
            if sp:
                idx[(sp, Path(n).stem)] = n
    print(f"Remote zip: {len(names)} entries, {len(idx)} FLACs indexed "
          f"({time.time() - t:.0f}s)")
    for sp in FOLDER:
        print(f"  {sp}: {sum(1 for k in idx if k[0] == sp)}")
    return idx


def worker(url, jobs, retries=3):
    done, failed = 0, []
    rz = RemoteZip(url)
    for sp, stem, entry in jobs:
        out = DEST / FOLDER[sp] / "flac" / f"{stem}.flac"
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries):
            try:
                with rz.open(entry) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                done += 1
                break
            except Exception:
                out.unlink(missing_ok=True)
                if attempt == retries - 1:
                    failed.append(stem)
                else:
                    time.sleep(1 + attempt)
                    rz = RemoteZip(url)
    rz.close()
    return done, failed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--check", action="store_true",
                   help="only test the server and show the zip layout")
    p.add_argument("--splits", nargs="+", default=["train", "dev", "eval"])
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--max-dev", type=int, default=3000)
    p.add_argument("--max-eval", type=int, default=8000)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    idx = build_index(args.url)
    if not idx:
        sys.exit("No FLACs found in the zip. Check --url and the zip layout.")
    if args.check:
        print("Example entries:", list(idx.values())[:2])
        print("OK: range requests work. Run again without --check.")
        return

    df = pd.read_csv(META)
    caps = {"train": args.max_train, "dev": args.max_dev, "eval": args.max_eval}
    jobs, have = [], 0
    for sp in args.splits:
        sub = df[df["split"] == sp]
        sub = sample_split(sub, caps[sp], balanced=(sp == "train"))
        for f in sub["file"]:
            if (DEST / FOLDER[sp] / "flac" / f"{f}.flac").exists():
                have += 1
            elif (sp, f) in idx:
                jobs.append((sp, f, idx[(sp, f)]))
        print(f"[{sp}] selected {len(sub)} files")
    print(f"Already on disk: {have} | to download: {len(jobs)}")
    if not jobs:
        return

    chunks = [jobs[i::args.workers] for i in range(args.workers)]
    t, total, failed = time.time(), 0, []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(worker, args.url, c) for c in chunks if c]
        for fut in cf.as_completed(futs):
            d, f = fut.result()
            total += d
            failed += f
            print(f"  ... {total}/{len(jobs)} files "
                  f"({time.time() - t:.0f}s)", flush=True)
    size = sum(f.stat().st_size for f in DEST.rglob("*.flac")) / 1e9
    print(f"\nDownloaded {total}, failed {len(failed)}. FLACs on disk: {size:.2f} GB")
    if failed:
        print("Rerun to retry the failed files:", failed[:5], "...")


if __name__ == "__main__":
    main()
