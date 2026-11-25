#!/usr/bin/env python3
"""Speaker-disjoint training/eval on DEEP-VOICE (RVC). 4 folds of 2 held-out speakers."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from run_final_experiments import (CnnModel, LightweightCNN,        # noqa: E402
                                   eer_and_threshold, metrics_at)

FOLDS = [("biden", "linus"), ("margot", "musk"), ("obama", "ryan"), ("taylor", "trump")]


class TunableCnn(CnnModel):
    """CnnModel whose training can start from given weights."""

    def __init__(self, seed, init_state=None, **kw):
        super().__init__(seed, **kw)
        self.init_state = init_state

    def fit(self, train, dev):
        torch.manual_seed(self.seed)
        self.net = LightweightCNN().to(self.dev)
        if self.init_state is not None:
            self.net.load_state_dict(self.init_state)
        y = train["y"]
        counts = np.bincount(y, minlength=2).astype(float)
        w = torch.tensor(len(y) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=self.dev)
        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        yt = torch.from_numpy(y).long().to(self.dev)
        yd = torch.from_numpy(dev["y"]).long().to(self.dev)
        best, best_state, bad = float("inf"), None, 0
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(y))
            for s in range(0, len(y), self.batch):
                idx = order[s:s + self.batch]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = crit(self.net(self._x(train["mel"], idx)), yt[idx])
                loss.backward()
                opt.step()
            vloss = self._loss(dev, yd, crit)
            sched.step(vloss)
            if vloss < best:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="clean", choices=["clean", "telephone", "telephone_codec"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--init-from", default="",
                    help='ASVspoof checkpoint template, e.g. "outputs/seeds5/models/clean_w1_cnn_s{seed}.pt"')
    ap.add_argument("--lr", type=float, default=0.0, help="default 1e-3 (scratch) or 1e-4 (fine-tune)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default="scratch")
    args = ap.parse_args()
    lr = args.lr or (1e-4 if args.init_from else 1e-3)

    base = ROOT / "data" / "processed" / "deepvoice"
    meta = pd.read_csv(base / "deepvoice_w1_meta.csv")
    f = np.load(base / f"deepvoice_{args.cond}_w1_feats.npz")
    mel, y = f["mel"], f["y"].astype(np.int64)
    assert len(y) == len(meta) and (y == meta.y.to_numpy()).all(), "features and metadata disagree"
    real, fake = (meta.y == 0), (meta.y == 1)

    rows = []
    for fi, pair in enumerate(FOLDS):
        pl = list(pair)
        in_test = (real & meta.src.isin(pl)) | (fake & meta.src.isin(pl) & meta.tgt.isin(pl))
        in_pool = (real & ~meta.src.isin(pl)) | (fake & ~meta.src.isin(pl) & ~meta.tgt.isin(pl))
        te = in_test.to_numpy()
        pool_files = meta[in_pool].drop_duplicates("file")[["file", "y"]]
        for seed in args.seeds:
            rng = np.random.default_rng(1000 * fi + seed)
            dev_real = rng.choice(pool_files[pool_files.y == 0].file.to_numpy(), 1, replace=False)
            fk = pool_files[pool_files.y == 1].file.to_numpy()
            dev_fake = rng.choice(fk, max(1, int(round(0.15 * len(fk)))), replace=False)
            is_dev = (in_pool & meta.file.isin(set(dev_real) | set(dev_fake))).to_numpy()
            is_tr = (in_pool.to_numpy()) & ~is_dev

            spk_tr = set(meta.src[is_tr]) | set(meta.tgt[is_tr])
            assert not (spk_tr & set(pair)), "speaker leakage into training!"
            assert not (set(meta.file[is_tr]) & set(meta.file[te])), "file leakage!"

            init = None
            if args.init_from:
                init = torch.load(args.init_from.format(seed=seed), map_location="cpu")["state"]
            model = TunableCnn(seed, init_state=init, epochs=args.epochs, lr=lr)
            model.fit({"mel": mel[is_tr], "y": y[is_tr]}, {"mel": mel[is_dev], "y": y[is_dev]})

            _, thr = eer_and_threshold(y[is_dev], model.score({"mel": mel[is_dev], "y": y[is_dev]}))
            s = model.score({"mel": mel[te], "y": y[te]})
            yt = y[te]
            r = metrics_at(yt, s, thr)
            r.update(tpr_fake=float((s[yt == 1] >= thr).mean()), fpr_real=float((s[yt == 0] >= thr).mean()),
                     fold=fi, pair="+".join(pair), seed=seed, cond=args.cond, tag=args.tag,
                     n_train=int(is_tr.sum()), n_dev=int(is_dev.sum()), n_test=int(te.sum()),
                     n_test_real=int((yt == 0).sum()), n_test_fake=int((yt == 1).sum()))
            rows.append(r)
            print(f"fold {fi} {pair} seed {seed}: train {r['n_train']} dev {r['n_dev']} test {r['n_test']} | "
                  f"EER {r['eer']:.3f} AUC {r['auc']:.3f} flag fake {r['tpr_fake']:.2f} flag real {r['fpr_real']:.2f}",
                  flush=True)

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "deepvoice"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"deepvoice_speaker_disjoint_{args.tag}.csv", index=False)
    print(f"\n=== {args.tag}: speaker-disjoint, {args.cond} audio (mean over seeds, per held-out pair) ===")
    print(res.groupby("pair")[["eer", "auc", "bal_acc", "tpr_fake", "fpr_real"]].mean().round(3).to_string())
    print("\noverall mean +/- std over folds x seeds:")
    for c in ("eer", "auc", "bal_acc", "tpr_fake", "fpr_real"):
        print(f"  {c:9s} {res[c].mean():.3f} +/- {res[c].std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
