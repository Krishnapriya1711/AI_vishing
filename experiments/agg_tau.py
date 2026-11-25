#!/usr/bin/env python3
import argparse, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from window_cache import (read_protocol, index_audio, load_trimmed, apply_condition, eer_thr, ROOT, CACHE, SR)
from stream_common import HOP_SEC, WIN_SEC, Scorer, discover
import extract_features as fx

ap = argparse.ArgumentParser()
ap.add_argument("--asv-root", required=True)
ap.add_argument("--regime", choices=["clean", "aug"], default="aug")
ap.add_argument("--kind", default="cnn")
ap.add_argument("--condition", default="clean", help="clean | telephone | telephone_codec")
ap.add_argument("--min-sec", type=float, default=4.0, help="keep utterances at least this long after trimming")
ap.add_argument("--n-dev", type=int, default=300, help="max per class")
ap.add_argument("--n-eval", type=int, default=500, help="max per class")
ap.add_argument("--max-fpr", type=float, default=0.10)
ap.add_argument("--cnn-dir", default="outputs/seeds5/models")
ap.add_argument("--base-dir", default="outputs/models")
a = ap.parse_args()

found = {k: p for k, p in discover([ROOT / a.cnn_dir, ROOT / a.base_dir]).items()
         if k[0] == a.regime and k[1] == a.kind}
if not found:
    raise SystemExit("no saved models")
scorers = [Scorer(p) for p in found.values()]
print(f"{len(scorers)} models | regime={a.regime} condition={a.condition}")


def window_margins(x):
    n, hop = int(WIN_SEC * SR), int(HOP_SEC * SR)
    out = []
    for s in range(0, len(x) - n + 1, hop):
        w = fx.peak_norm(x[s:s + n])
        mf, mel = fx.mfcc_stats(w)[None], fx.log_mel(w, fx.frames_for(WIN_SEC))[None]
        out.append(float(np.mean([sc.score(mf, mel)[0] - sc.thr for sc in scorers])))
    return np.array(out)


def collect(split, n_per_class):
    cf = CACHE / f"agg_{split}_{a.regime}_{a.kind}_{a.condition}_{a.min_sec}_{n_per_class}.pkl"
    if cf.exists():
        return pickle.load(open(cf, "rb"))
    rows, idx = read_protocol(a.asv_root, split), index_audio(a.asv_root)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(rows))
    got = {0: [], 1: []}
    for j, i in enumerate(order):
        fid, lab = rows[i]
        if len(got[lab]) >= n_per_class or fid not in idx:
            continue
        x = load_trimmed(idx[fid])
        if len(x) < a.min_sec * SR:
            continue
        got[lab].append(window_margins(apply_condition(x, a.condition, seed=len(got[lab]))))
        if all(len(v) >= n_per_class for v in got.values()):
            break
    M = got[0] + got[1]
    y = np.array([0] * len(got[0]) + [1] * len(got[1]))
    print(f"{split}: {len(got[0])} bona fide, {len(got[1])} spoof utterances >= {a.min_sec}s")
    pickle.dump((M, y), open(cf, "wb"))
    return M, y


def series(m, rule, k):
    n, i = len(m), np.arange(len(m))
    if rule == "mean":
        return np.cumsum(m) / (i + 1)
    if rule == "movavg":
        return np.array([m[max(0, t - k + 1):t + 1].mean() for t in i])
    if rule == "consec":                      # min of last k margins >= tau  (tau = 0 -> "k in a row above threshold")
        return np.array([m[t - k + 1:t + 1].min() if t >= k - 1 else -1e9 for t in i])
    frac = np.cumsum(m >= 0) / (i + 1)        # vote: fraction of windows so far above threshold
    return np.where(i >= k - 1, frac, -1e9)


def utt_scores(M, rule, k, nwin=None):
    return np.array([series(m, rule, k)[:nwin].max() for m in M])


def pick_tau(y, s, max_fpr):
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(fpr <= max_fpr)[0]
    i = ok[np.argmax(tpr[ok])]
    return float(thr[i]), float(fpr[i]), float(tpr[i])


def first_alarm(m, rule, k, tau):
    hit = np.where(series(m, rule, k) >= tau)[0]
    return WIN_SEC + HOP_SEC * hit[0] if len(hit) else np.nan


Md, yd = collect("dev", a.n_dev)
Me, ye = collect("eval", a.n_eval)
CONFIG = {"mean": [1], "movavg": [2, 3, 4], "consec": [2, 3, 4], "vote": [2, 3, 4]}

chosen, rows = {}, []
for rule, ks in CONFIG.items():
    best = None
    for k in ks:                                                        # k and tau chosen on DEV only
        tau, fpr, tpr = pick_tau(yd, utt_scores(Md, rule, k), a.max_fpr)
        if best is None or tpr > best[3] + 1e-9:
            best = (k, tau, fpr, tpr)
    k, tau, dfpr, dtpr = best
    chosen[rule] = (k, tau)
    s = utt_scores(Me, rule, k)
    p = s >= tau
    tpr, fpr = p[ye == 1].mean(), p[ye == 0].mean()
    t_sp = [first_alarm(m, rule, k, tau) for m, l in zip(Me, ye) if l == 1]
    t_bf = [first_alarm(m, rule, k, tau) for m, l in zip(Me, ye) if l == 0]
    rows.append(dict(rule=rule, k=k, tau=round(tau, 3), dev_TPR=dtpr, dev_FPR=dfpr, eval_TPR=tpr, eval_FPR=fpr,
                     eval_bal_acc=(tpr + 1 - fpr) / 2, eval_EER=eer_thr(ye, s)[0], eval_AUC=roc_auc_score(ye, s),
                     median_alarm_s_spoof=np.nanmedian(t_sp), median_alarm_s_bonafide_FA=np.nanmedian(t_bf)))
res = pd.DataFrame(rows)
out = ROOT / "outputs" / "aggregation"; out.mkdir(parents=True, exist_ok=True)
tag = f"{a.regime}_{a.condition}"
res.to_csv(out / f"rules_{tag}.csv", index=False)
print(res.round(3).to_string(index=False))

# early detection: frozen (k, tau); only the first nwin windows are observed
grid = np.arange(WIN_SEC, a.min_sec + 1e-9, HOP_SEC)
cur = []
for rule, (k, tau) in chosen.items():
    for D in grid:
        nw = int((D - WIN_SEC) / HOP_SEC) + 1
        s = utt_scores(Me, rule, k, nw)
        p = s >= tau
        tpr, fpr = p[ye == 1].mean(), p[ye == 0].mean()
        cur.append(dict(rule=rule, observed_s=D, TPR=tpr, FPR=fpr, bal_acc=(tpr + 1 - fpr) / 2,
                        EER=eer_thr(ye, s)[0]))
cur = pd.DataFrame(cur)
cur.to_csv(out / f"early_{tag}.csv", index=False)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
for rule, g in cur.groupby("rule"):
    xs = g.observed_s.to_numpy()
    ax[0].plot(xs, 100 * g.bal_acc.to_numpy(), marker="o", label=rule)
    ax[1].plot(xs, 100 * g.EER.to_numpy(), marker="o", label=rule)
ax[0].set_ylabel("balanced accuracy at frozen τ (%)")
ax[1].set_ylabel("utterance EER (%)")
for x_ in ax:
    x_.set_xlabel("observed audio (s)")
    x_.grid(alpha=.3)
ax[0].legend()
plt.tight_layout()
plt.savefig(out / f"early_{tag}.png", dpi=200)
print("saved to", out)
