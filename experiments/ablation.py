#!/usr/bin/env python3
import argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from window_cache import get_feats, eer_thr, bal_acc, ROOT

REGIMES = {"clean": ["clean"], "tel_only": ["telephone"], "codec_only": ["codec_only"],
           "mixed": ["clean", "telephone", "telephone_codec"],
           "mixed_plus": ["clean", "telephone", "telephone_codec", "telephone_mulaw", "telephone_codec+n10"]}
TESTS = ["clean", "telephone", "telephone_codec", "codec_only", "telephone_mulaw", "clean+n10", "telephone_codec+n10"]

ap = argparse.ArgumentParser()
ap.add_argument("--asv-root", required=True)
ap.add_argument("--n-train", type=int, default=2000)
ap.add_argument("--n-dev", type=int, default=1500)
ap.add_argument("--n-eval", type=int, default=4000)
ap.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123, 2024, 1337])
ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
ap.add_argument("--epochs", type=int, default=30)
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = ROOT / "outputs" / "ablation"; (OUT / "models").mkdir(parents=True, exist_ok=True)
RAW = OUT / "raw.csv"


class MelCNN(nn.Module):
    def __init__(s):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.c = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(s, x):
        return s.c(s.f(x))


def stack(split, n, conds):
    Xs, ys = zip(*[get_feats(a.asv_root, split, n, c, "mel") for c in conds])
    return torch.tensor(np.concatenate(Xs))[:, None], torch.tensor(np.concatenate(ys))


@torch.no_grad()
def margins(model, X):
    model.eval()
    out = [model(X[i:i + 512].to(dev)).cpu() for i in range(0, len(X), 512)]
    o = torch.cat(out)
    return (o[:, 1] - o[:, 0]).numpy()


def loss_on(model, X, y, lossf):
    model.eval(); tot = 0.0
    with torch.no_grad():
        for i in range(0, len(X), 512):
            tot += lossf(model(X[i:i + 512].to(dev)), y[i:i + 512].to(dev)).item() * len(X[i:i + 512])
    return tot / len(X)


def train(Xtr, ytr, Xdv, ydv, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MelCNN().to(dev)
    w = torch.tensor(len(ytr) / (2 * np.bincount(ytr.numpy())), dtype=torch.float32, device=dev)
    lossf_train = nn.CrossEntropyLoss(weight=w)
    lossf_dev = nn.CrossEntropyLoss(weight=w, reduction="sum")
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, best_state, bad = 1e9, None, 0
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            if len(b) < 2:
                continue
            opt.zero_grad()
            lossf_train(m(Xtr[b].to(dev)), ytr[b].to(dev)).backward()
            opt.step()
        dl = loss_on(m, Xdv, ydv, lossf_dev) / float(w.mean() * len(ydv))
        sch.step(dl)
        if dl < best - 1e-4:
            best, bad = dl, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 7:
                break
    m.load_state_dict(best_state)
    return m, ep + 1


done = set()
if RAW.exists():
    r = pd.read_csv(RAW); done = set(zip(r.regime, r.seed))
print("trainable params:", sum(p.numel() for p in MelCNN().parameters() if p.requires_grad))

for reg in a.regimes:
    Xtr, ytr = stack("train", a.n_train, REGIMES[reg])
    Xdv, ydv = stack("dev", a.n_dev, REGIMES[reg])
    for seed in a.seeds:
        if (reg, seed) in done:
            continue
        m, ep = train(Xtr, ytr, Xdv, ydv, seed)
        _, thr = eer_thr(ydv.numpy(), margins(m, Xdv))          # frozen on the regime's own dev set
        torch.save(m.state_dict(), OUT / "models" / f"{reg}_s{seed}.pt")
        rows = []
        for t in TESTS:
            Xe, ye = get_feats(a.asv_root, "eval", a.n_eval, t, "mel")
            s = margins(m, torch.tensor(Xe)[:, None])
            e, _ = eer_thr(ye, s)
            rows.append(dict(regime=reg, seed=seed, test=t, epochs=ep, EER=100 * e,
                             AUC=roc_auc_score(ye, s), bal_acc_frozen=100 * bal_acc(ye, s, thr)))
        pd.DataFrame(rows).to_csv(RAW, mode="a", header=not RAW.exists(), index=False)
        print(f"{reg} seed {seed}: " + " ".join(f"{r['test']}={r['EER']:.1f}" for r in rows), flush=True)

df = pd.read_csv(RAW)
g = df.groupby(["regime", "test"]).EER.agg(["mean", "std"]).reset_index()
g["cell"] = g.apply(lambda r: f"{r['mean']:.1f} ± {r['std']:.1f}", axis=1)
tab = g.pivot(index="regime", columns="test", values="cell").reindex(
    [r for r in list(REGIMES) if r in g["regime"].values]
)
# Keep only test columns that were actually evaluated
present_tests = [t for t in TESTS if t in tab.columns]
tab = tab[present_tests]
tab.to_csv(OUT / "ablation_eer_table.csv")

# Write LaTeX table manually (avoids jinja2 version requirement)
def _write_latex(tab, path):
    cols = tab.columns.tolist()
    col_fmt = "l" + "r" * len(cols)
    short = {"clean": "Clean", "telephone": "Tel", "telephone_codec": "Tel+Codec",
             "codec_only": "Codec-only", "telephone_mulaw": "Tel+μ-law",
             "clean+n10": "Clean+Noise", "telephone_codec+n10": "Tel+Codec+Noise"}
    lines = [
        "\\begin{tabular}{" + col_fmt + "}",
        "\\toprule",
        "Regime & " + " & ".join(short.get(c, c) for c in cols) + " \\\\",
        "\\midrule",
    ]
    for regime, row in tab.iterrows():
        lines.append(regime + " & " + " & ".join(str(row[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines))

_write_latex(tab, OUT / "ablation_table.tex")
print(tab.to_string())
