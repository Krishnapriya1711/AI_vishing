"""Detection metrics: EER, min-tDCF, accuracy, precision, recall, F1.

Score convention throughout this module:

* ``labels`` — 0 = bonafide, 1 = spoof.
* ``scores`` — higher value → more likely to be spoof.

For a threshold τ:
    predict spoof  if score >= τ
    predict bona   if score < τ

So:
    FPR(τ) = P(score >= τ | bonafide)   — bonafide flagged as spoof
    FNR(τ) = P(score <  τ | spoof)      — spoof accepted as bona fide
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# LABEL CONVENTION
# ============================================================

BONAFIDE = 0
SPOOF = 1


# ============================================================
# EQUAL ERROR RATE
# ============================================================

def compute_eer(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """Compute the Equal Error Rate and its operating threshold.

    Returns
    -------
    eer : float
        Equal error rate in the range [0, 1].
    threshold : float
        Score threshold at which FPR ≈ FNR.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    # Sort by score descending (higher = more likely spoof)
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    n_total = len(labels)
    n_spoof = np.sum(labels == SPOOF)
    n_bona = np.sum(labels == BONAFIDE)

    if n_spoof == 0 or n_bona == 0:
        raise ValueError("Need both bonafide and spoof samples for EER.")

    # Cumulative counts as we lower the threshold from +inf
    # At each position, everything at or above this score is "spoof"
    cum_spoof = np.cumsum(sorted_labels == SPOOF)   # true spoofs above threshold
    cum_bona = np.cumsum(sorted_labels == BONAFIDE)  # bona fides above threshold

    # FNR = spoof below threshold / total spoof
    fnr = (n_spoof - cum_spoof) / n_spoof
    # FPR = bona above threshold / total bona
    fpr = cum_bona / n_bona

    # Find the crossing point
    diff = fpr - fnr
    # Find where diff changes sign
    idx = np.argmin(np.abs(diff))

    # Linear interpolation for a more precise EER
    if idx > 0:
        # Interpolate between idx-1 and idx
        a = diff[idx - 1]
        b = diff[idx]
        if abs(a - b) > 1e-10:
            t = a / (a - b)
            eer = fpr[idx - 1] + t * (fpr[idx] - fpr[idx - 1])
            thr = sorted_scores[idx - 1] + t * (
                sorted_scores[idx] - sorted_scores[idx - 1]
            )
        else:
            eer = (fpr[idx] + fnr[idx]) / 2.0
            thr = sorted_scores[idx]
    else:
        eer = (fpr[idx] + fnr[idx]) / 2.0
        thr = sorted_scores[idx]

    return float(eer), float(thr)


# ============================================================
# MINIMUM t-DCF (ASVspoof 2019)
# ============================================================

def compute_min_tdcf(
    labels: np.ndarray,
    scores: np.ndarray,
    p_spoof: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> float:
    """Compute the minimum tandem Detection Cost Function (min-tDCF).

    Uses the ASVspoof 2019 cost model:

        tDCF(τ) = [C_miss · Pmiss(τ) · P_bona + C_fa · Pfa(τ) · P_spoof]
                   / min(C_miss · P_bona, C_fa · P_spoof)

    where (in our score convention):
        Pmiss(τ) = P(score >= τ | bonafide)  — bona fide rejected
        Pfa(τ)   = P(score <  τ | spoof)     — spoof accepted

    Parameters
    ----------
    labels : array-like  (0 = bonafide, 1 = spoof)
    scores : array-like  (higher = more likely spoof)
    p_spoof : float
        Prior probability of a spoof trial.
    c_miss : float
        Cost of missing a bonafide (rejecting bona fide).
    c_fa : float
        Cost of a false alarm (accepting a spoof).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    p_bona = 1.0 - p_spoof

    n_spoof = np.sum(labels == SPOOF)
    n_bona = np.sum(labels == BONAFIDE)

    if n_spoof == 0 or n_bona == 0:
        raise ValueError("Need both bonafide and spoof samples for tDCF.")

    # Sort by score descending
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    cum_spoof = np.cumsum(sorted_labels == SPOOF)
    cum_bona = np.cumsum(sorted_labels == BONAFIDE)

    # Pmiss = fraction of bonafide with score >= threshold (rejected)
    p_miss = cum_bona / n_bona
    # Pfa = fraction of spoof with score < threshold (accepted)
    p_fa = (n_spoof - cum_spoof) / n_spoof

    # tDCF for each threshold
    c_norm = 1.0 / min(c_miss * p_bona, c_fa * p_spoof)
    tdcf = c_norm * (c_miss * p_miss * p_bona + c_fa * p_fa * p_spoof)

    return float(np.min(tdcf))


# ============================================================
# COMPREHENSIVE EVALUATION
# ============================================================

def evaluate_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> Dict[str, float]:
    """Compute a full evaluation report from raw scores.

    Parameters
    ----------
    labels : array-like (0 = bonafide, 1 = spoof)
    scores : array-like (higher = more likely spoof)
    threshold : float or None
        If None, uses the EER-optimal threshold.

    Returns
    -------
    dict with keys: eer, eer_threshold, min_tdcf, accuracy,
    precision, recall, f1, auc, confusion_matrix
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    # EER
    eer, eer_threshold = compute_eer(labels, scores)

    # Threshold for hard predictions
    if threshold is None:
        threshold = eer_threshold

    preds = (scores >= threshold).astype(int)

    # min-tDCF
    try:
        min_tdcf = compute_min_tdcf(labels, scores)
    except ValueError:
        min_tdcf = float("nan")

    # Standard metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    # AUC
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(labels, preds)

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "min_tdcf": min_tdcf,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
    }


# ============================================================
# FORMAT RESULTS FOR PRINTING
# ============================================================

def format_results(results: Dict[str, float]) -> str:
    """Pretty-print an evaluation result dict."""
    lines = [
        f"  EER          : {results['eer']:.4f}",
        f"  EER threshold : {results['eer_threshold']:.4f}",
        f"  min-tDCF      : {results['min_tdcf']:.4f}",
        f"  Accuracy      : {results['accuracy']:.4f}",
        f"  Precision     : {results['precision']:.4f}",
        f"  Recall        : {results['recall']:.4f}",
        f"  F1-score      : {results['f1']:.4f}",
        f"  AUC           : {results['auc']:.4f}",
    ]

    cm = results.get("confusion_matrix")
    if cm is not None:
        lines.append(f"  Confusion     : TN={cm[0,0]} FP={cm[0,1]} "
                      f"FN={cm[1,0]} TP={cm[1,1]}")

    return "\n".join(lines)
