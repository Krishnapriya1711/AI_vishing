"""Shared helpers for the streaming experiments (stream_score / stream_analyze / demo_stream).

Loads the models that run_final_experiments.py saved:
    CNN checkpoints   <dir>/<regime>_w1_cnn_s<seed>.pt          {"state": ..., "thr": ...}
    sklearn models    <dir>/<regime>_w1_<kind>.joblib           {"kind", "sc", "m", "thr"}
`thr` is the window-level EER threshold chosen on ASVspoof dev and then frozen.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
WIN_SEC, HOP_SEC = 1.0, 0.5
CONDS = ["clean", "telephone", "telephone_codec"]
KINDS = {"cnn", "logistic_regression", "random_forest", "svm"}
WARNING = ("Potential AI-generated voice detected. Verify the caller before "
           "sharing OTP, PIN or banking information.")


def duration_at(n_windows):
    """Seconds of audio needed before the n-th window is complete."""
    return WIN_SEC + HOP_SEC * (n_windows - 1)


class LightweightCNN(nn.Module):
    """Same architecture as src/models/cnn.py."""

    def __init__(self, num_classes=2):
        super().__init__()

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*block(1, 32), *block(32, 64),
                                      *block(64, 128), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 64),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3),
                                        nn.Linear(64, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def discover(dirs):
    """{(regime, kind, seed): path}; seed = -1 for the single sklearn models."""
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


class Scorer:
    """Loads one saved model; score(mfcc, mel) -> higher = more likely spoof."""

    def __init__(self, path, device=None):
        self.path = Path(path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.path.suffix == ".pt":
            try:
                ck = torch.load(self.path, map_location=self.device)
            except Exception:
                ck = torch.load(self.path, map_location=self.device, weights_only=False)
            self.kind = "cnn"
            self.net = LightweightCNN().to(self.device)
            self.net.load_state_dict(ck["state"])
            self.net.eval()
        else:
            ck = joblib.load(self.path)
            self.kind = ck["kind"]
            self.sc, self.m = ck["sc"], ck["m"]
        self.thr = float(ck["thr"])

    @torch.no_grad()
    def score(self, mfcc, mel):
        if self.kind == "cnn":
            out = []
            for s in range(0, len(mel), 512):
                x = torch.from_numpy(mel[s:s + 512]).float().unsqueeze(1).to(self.device)
                lg = self.net(x)
                out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
            return np.concatenate(out).astype(np.float32)
        X = self.sc.transform(mfcc)
        if self.kind == "random_forest":
            return self.m.predict_proba(X)[:, 1].astype(np.float32)
        return self.m.decision_function(X).astype(np.float32)
