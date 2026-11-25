#!/usr/bin/env python3
"""Corrected experiments: no noise injection, proper train/val/eval protocol.

Key corrections from the review:
  1. NO calibrated noise injection — features are used as extracted.
  2. Training data split into train (80%) + validation (20%).
  3. CNN early stopping uses validation loss, NOT evaluation loss.
  4. EER threshold found on validation set, frozen, then applied to eval.
  5. Evaluation set touched exactly once per model per condition.
  6. Same train/val/eval splits for all four models.
  7. Deterministic seeds (random, numpy, torch).
  8. Graph axes start at 0, not clipped to hide poor results.

Usage:
    python experiments/run_corrected_experiments.py
"""

import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import Config
from src.utils.audio import load_audio
from src.data.dataset import (
    load_metadata,
    filter_metadata,
    split_train_val,
    prepare_flat_features,
)
from src.features.extractors import extract_features, extract_spectrogram_for_cnn
from src.models.baselines import BaselineModel
from src.models.cnn import LightweightCNN
from src.evaluation.metrics import (
    compute_eer,
    evaluate_with_frozen_threshold,
    evaluate_predictions,
    format_results,
)


# ============================================================
# DETERMINISTIC SEEDS
# ============================================================

SEED = 42

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SPECTROGRAM PRE-EXTRACTION
# ============================================================

def extract_all_spectrograms(df, condition, window_sec, config,
                              n_mels=64, max_frames=300):
    """Pre-extract log-Mel spectrograms into GPU tensors."""
    specs, labels = [], []
    for _, row in df.iterrows():
        try:
            audio, sr = load_audio(
                file_name=row["file"], split=row["split"],
                condition=condition, duration=window_sec,
                base_dir=config.raw_audio_base,
                target_sr_clean=config.target_sr_clean,
                target_sr_phone=config.target_sr_phone,
            )
            mel = extract_spectrogram_for_cnn(
                audio, sr, n_mels=n_mels,
                hop_length=config.hop_length, n_fft=config.n_fft,
                win_length=config.win_length,
                fmax=min(sr / 2, config.fmax_clean),
                max_frames=max_frames,
            )
            specs.append(mel)
            labels.append(config.label_map.get(row.get("label", "spoof"), 1))
        except (FileNotFoundError, ValueError):
            continue

    if not specs:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs_t = torch.from_numpy(np.stack(specs)).float().unsqueeze(1).to(device)
    labels_t = torch.tensor(labels, dtype=torch.long).to(device)
    return specs_t, labels_t


# ============================================================
# CNN TRAIN + EVALUATE (proper protocol)
# ============================================================

def train_eval_cnn_corrected(
    train_specs, train_labels,
    val_specs, val_labels,
    eval_specs, eval_labels,
    n_mels=64,
    epochs=50,
    lr=1e-3,
    patience=7,
    batch_size=32,
):
    """Train CNN with proper train/val/eval separation.

    - Train on train_specs/train_labels (weight updates)
    - Validate on val_specs/val_labels (early stopping, checkpoint)
    - Evaluate on eval_specs/eval_labels (final metrics, once)

    Returns: (eval_results_dict, training_history_dict, n_params)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightCNN(n_mels=n_mels).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        idx = torch.randperm(len(train_specs), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_specs), batch_size):
            batch_idx = idx[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(train_specs[batch_idx])
            loss = criterion(logits, train_labels[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # --- Validation (early stopping + checkpoint selection) ---
        model.eval()
        with torch.no_grad():
            val_logits = model(val_specs)
            val_loss = criterion(val_logits, val_labels).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1:2d}/{epochs}  "
              f"train={avg_train:.4f}  val={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best={best_epoch}, val_loss={best_val_loss:.4f})")
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Find EER threshold on VALIDATION set ---
    model.eval()
    with torch.no_grad():
        val_scores = torch.softmax(model(val_specs), dim=1)[:, 1].cpu().numpy()
    val_labels_np = val_labels.cpu().numpy()
    val_eer, val_threshold = compute_eer(val_labels_np, val_scores)
    print(f"  Validation EER={val_eer:.4f}, threshold={val_threshold:.4f}")

    # --- Evaluate on EVAL set with FROZEN threshold ---
    with torch.no_grad():
        eval_scores = torch.softmax(model(eval_specs), dim=1)[:, 1].cpu().numpy()
    eval_labels_np = eval_labels.cpu().numpy()

    eval_results = evaluate_with_frozen_threshold(
        eval_labels_np, eval_scores, val_threshold
    )

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    return eval_results, history, model.num_parameters()


# ============================================================
# EXPERIMENT 1: SHORT-DURATION DETECTION
# ============================================================

def run_short_duration_corrected(config, max_train=600, max_eval=400):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: SHORT-DURATION DETECTION (Corrected Protocol)")
    print("  No noise injection | Train/Val/Eval separation | Frozen threshold")
    print("=" * 70)

    windows = [3.0, 2.0, 1.5, 1.0, 0.75, 0.5]
    rows = []

    for window in windows:
        print(f"\n--- Window {window}s ---")
        set_seeds(SEED)

        sw_df = load_metadata("short_window_metadata.csv", config)
        sw_df = sw_df[sw_df["window_sec"] == window]
        full_train = filter_metadata(sw_df[sw_df["split"] == "train"], max_samples=max_train)
        eval_df = filter_metadata(sw_df[sw_df["split"] == "eval"], max_samples=max_eval)

        # Split train into train + validation
        train_df, val_df = split_train_val(full_train, val_ratio=0.2, random_state=SEED)
        print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Eval: {len(eval_df)}")

        # ---- Baselines (MFCC features) ----
        print("  Extracting MFCC features for baselines...")
        X_train, y_train = prepare_flat_features(
            train_df, condition="clean", window_sec=window,
            feature_type="mfcc", config=config)
        X_val, y_val = prepare_flat_features(
            val_df, condition="clean", window_sec=window,
            feature_type="mfcc", config=config)
        X_eval, y_eval = prepare_flat_features(
            eval_df, condition="clean", window_sec=window,
            feature_type="mfcc", config=config)

        if len(X_train) == 0 or len(X_eval) == 0:
            print("  No features — skipping.")
            continue

        for model_type in ["logistic_regression", "random_forest", "svm"]:
            model = BaselineModel(model_type)
            model.fit(X_train, y_train)

            # Find EER threshold on VALIDATION set
            val_scores = model.predict_scores(X_val)
            val_eer, val_threshold = compute_eer(y_val, val_scores)

            # Evaluate on EVAL set with frozen threshold
            eval_scores = model.predict_scores(X_eval)
            eval_results = evaluate_with_frozen_threshold(
                y_eval, eval_scores, val_threshold)

            print(f"  {model_type:25s}: val_eer={val_eer:.4f}  "
                  f"eval_eer={eval_results['eer']:.4f}  "
                  f"acc={eval_results['accuracy']:.4f}")

            rows.append({
                "window_sec": window, "model": model_type,
                "condition": "clean", "feature": "mfcc",
                "val_eer": val_eer, "val_threshold": val_threshold,
                "eer": eval_results["eer"], "min_tdcf": eval_results["min_tdcf"],
                "accuracy": eval_results["accuracy"],
                "precision": eval_results["precision"],
                "recall": eval_results["recall"],
                "f1": eval_results["f1"], "auc": eval_results["auc"],
            })

        # ---- CNN (Mel spectrogram) ----
        print("  Extracting Mel spectrograms for CNN...")
        set_seeds(SEED + int(window * 100))  # Different seed per window

        tr_specs, tr_labels = extract_all_spectrograms(train_df, "clean", window, config)
        v_specs, v_labels = extract_all_spectrograms(val_df, "clean", window, config)
        e_specs, e_labels = extract_all_spectrograms(eval_df, "clean", window, config)

        if tr_specs is None or e_specs is None:
            print("  CNN spectrogram extraction failed — skipping CNN.")
            continue

        cnn_results, history, n_params = train_eval_cnn_corrected(
            tr_specs, tr_labels, v_specs, v_labels, e_specs, e_labels,
            n_mels=config.n_mels, epochs=30, patience=7, batch_size=32,
        )

        print(f"  {'cnn':25s}: val_eer=N/A  "
              f"eval_eer={cnn_results['eer']:.4f}  "
              f"acc={cnn_results['accuracy']:.4f}")

        rows.append({
            "window_sec": window, "model": "cnn",
            "condition": "clean", "feature": "mel",
            "val_eer": None, "val_threshold": cnn_results["frozen_threshold"],
            "eer": cnn_results["eer"], "min_tdcf": cnn_results["min_tdcf"],
            "accuracy": cnn_results["accuracy"],
            "precision": cnn_results["precision"],
            "recall": cnn_results["recall"],
            "f1": cnn_results["f1"], "auc": cnn_results["auc"],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        print("\n" + "=" * 70)
        print("SHORT-DURATION SUMMARY (EER %)")
        print("=" * 70)
        pivot = df.pivot_table(values="eer", index="window_sec",
                               columns="model", aggfunc="mean") * 100
        print(pivot.to_string())

    path = config.output_dir / "corrected_short_duration_results.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved to {path}")
    return df


# ============================================================
# EXPERIMENT 2: PHONE ROBUSTNESS
# ============================================================

def run_phone_robustness_corrected(config, max_train=600, max_eval=400):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PHONE / CODEC ROBUSTNESS (Corrected Protocol)")
    print("  Train on clean | Validate on clean | Evaluate on each condition")
    print("=" * 70)

    window = 2.0
    conditions = ["clean", "telephone", "telephone_codec"]
    rows = []

    set_seeds(SEED)

    sw_df = load_metadata("short_window_metadata.csv", config)
    sw_df = sw_df[sw_df["window_sec"] == window]
    full_train = filter_metadata(sw_df[sw_df["split"] == "train"], max_samples=max_train)
    eval_df = filter_metadata(sw_df[sw_df["split"] == "eval"], max_samples=max_eval)
    train_df, val_df = split_train_val(full_train, val_ratio=0.2, random_state=SEED)

    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Eval: {len(eval_df)}")

    # ---- Baselines ----
    print("\n  Extracting MFCC features for baselines (clean train/val)...")
    X_train, y_train = prepare_flat_features(
        train_df, condition="clean", window_sec=window,
        feature_type="mfcc", config=config)
    X_val, y_val = prepare_flat_features(
        val_df, condition="clean", window_sec=window,
        feature_type="mfcc", config=config)

    for model_type in ["logistic_regression", "random_forest", "svm"]:
        model = BaselineModel(model_type)
        model.fit(X_train, y_train)

        # Frozen threshold from validation
        val_scores = model.predict_scores(X_val)
        val_eer, val_threshold = compute_eer(y_val, val_scores)

        for eval_cond in conditions:
            X_eval, y_eval = prepare_flat_features(
                eval_df, condition=eval_cond, window_sec=window,
                feature_type="mfcc", config=config)
            if len(X_eval) == 0:
                continue

            eval_scores = model.predict_scores(X_eval)
            eval_results = evaluate_with_frozen_threshold(
                y_eval, eval_scores, val_threshold)

            print(f"  {model_type:25s} / {eval_cond:16s}: "
                  f"eer={eval_results['eer']:.4f}  acc={eval_results['accuracy']:.4f}")

            rows.append({
                "train_condition": "clean", "eval_condition": eval_cond,
                "window_sec": window, "model": model_type,
                "val_threshold": val_threshold,
                "eer": eval_results["eer"], "min_tdcf": eval_results["min_tdcf"],
                "accuracy": eval_results["accuracy"], "f1": eval_results["f1"],
            })

    # ---- CNN ----
    print("\n  Extracting Mel spectrograms for CNN (clean train/val)...")
    set_seeds(SEED)

    tr_specs, tr_labels = extract_all_spectrograms(train_df, "clean", window, config)
    v_specs, v_labels = extract_all_spectrograms(val_df, "clean", window, config)

    if tr_specs is not None and v_specs is not None:
        for eval_cond in conditions:
            print(f"\n  CNN eval on: {eval_cond}")
            e_specs, e_labels = extract_all_spectrograms(
                eval_df, eval_cond, window, config)
            if e_specs is None:
                continue

            # Retrain CNN from scratch (same train/val, fresh seed)
            set_seeds(SEED)
            cnn_results, history, n_params = train_eval_cnn_corrected(
                tr_specs, tr_labels, v_specs, v_labels,
                e_specs, e_labels,
                n_mels=config.n_mels, epochs=30, patience=7, batch_size=32,
            )

            print(f"  {'cnn':25s} / {eval_cond:16s}: "
                  f"eer={cnn_results['eer']:.4f}  acc={cnn_results['accuracy']:.4f}")

            rows.append({
                "train_condition": "clean", "eval_condition": eval_cond,
                "window_sec": window, "model": "cnn",
                "val_threshold": cnn_results["frozen_threshold"],
                "eer": cnn_results["eer"], "min_tdcf": cnn_results["min_tdcf"],
                "accuracy": cnn_results["accuracy"], "f1": cnn_results["f1"],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        print("\n" + "=" * 70)
        print("PHONE ROBUSTNESS SUMMARY (EER %)")
        print("=" * 70)
        pivot = df.pivot_table(values="eer", index="eval_condition",
                               columns="model", aggfunc="mean") * 100
        print(pivot.to_string())

    path = config.output_dir / "corrected_phone_robustness_results.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved to {path}")
    return df


# ============================================================
# GENERATE CORRECTED GRAPHS
# ============================================================

def generate_corrected_graphs(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR = config.output_dir / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_COLORS = {
        "logistic_regression": "#2196F3",
        "random_forest": "#4CAF50",
        "svm": "#FF5722",
        "cnn": "#9C27B0",
    }
    MODEL_LABELS = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "svm": "SVM (RBF)",
        "cnn": "CNN",
    }
    MARKERS = {"logistic_regression": "o", "random_forest": "s",
               "svm": "D", "cnn": "^"}

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    })

    sd_df = pd.read_csv(config.output_dir / "corrected_short_duration_results.csv")
    ph_df = pd.read_csv(config.output_dir / "corrected_phone_robustness_results.csv")

    # --- Fig 1: EER vs Window (corrected, y from 0) ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in sd_df["model"].unique():
        sub = sd_df[sd_df["model"] == model].sort_values("window_sec", ascending=False)
        ls = "--" if model == "cnn" else "-"
        lw = 2.5 if model == "cnn" else 2
        ax.plot(sub["window_sec"].to_numpy(), (sub["eer"] * 100).to_numpy(),
                marker=MARKERS.get(model, "o"), linewidth=lw, markersize=9,
                color=MODEL_COLORS.get(model, "gray"),
                label=MODEL_LABELS.get(model, model), linestyle=ls)
    ax.set_xlabel("Window Duration (seconds)")
    ax.set_ylabel("EER (%)")
    ax.set_title("EER vs Window Duration — Corrected Protocol (No Noise Injection)")
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=-1)
    fig.savefig(FIGURES_DIR / "corrected_eer_vs_window.png")
    plt.close(fig)
    print("  corrected_eer_vs_window.png")

    # --- Fig 2: Accuracy vs Window (corrected, y from 0) ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in sd_df["model"].unique():
        sub = sd_df[sd_df["model"] == model].sort_values("window_sec", ascending=False)
        ls = "--" if model == "cnn" else "-"
        lw = 2.5 if model == "cnn" else 2
        ax.plot(sub["window_sec"].to_numpy(), (sub["accuracy"] * 100).to_numpy(),
                marker=MARKERS.get(model, "o"), linewidth=lw, markersize=9,
                color=MODEL_COLORS.get(model, "gray"),
                label=MODEL_LABELS.get(model, model), linestyle=ls)
    ax.set_xlabel("Window Duration (seconds)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Window Duration — Corrected Protocol")
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_ylim(0, 101)  # FIXED: start at 0, not 75
    fig.savefig(FIGURES_DIR / "corrected_accuracy_vs_window.png")
    plt.close(fig)
    print("  corrected_accuracy_vs_window.png")

    # --- Fig 3: Phone robustness EER (corrected) ---
    conditions = ["clean", "telephone", "telephone_codec"]
    models = ph_df["model"].unique()
    n_models = len(models)
    bar_width = 0.17

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(conditions))
    for i, model in enumerate(models):
        vals = []
        for cond in conditions:
            row = ph_df[(ph_df["model"] == model) & (ph_df["eval_condition"] == cond)]
            vals.append(row["eer"].values[0] * 100 if len(row) > 0 else 0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        hatch = "//" if model == "cnn" else ""
        ax.bar(x + offset, vals, bar_width,
               label=MODEL_LABELS.get(model, model),
               color=MODEL_COLORS.get(model, "gray"),
               edgecolor="white", linewidth=0.5, hatch=hatch)
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 1, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=9)
    ax.set_xlabel("Evaluation Condition")
    ax.set_ylabel("EER (%)")
    ax.set_title("Phone / Codec Robustness — Corrected Protocol (No Noise, Frozen Threshold)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean\n(16 kHz)", "Telephone\n(8 kHz band-limited)",
                         "Telephone + Codec\n(8 kHz + G.711 A-law)"])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(ph_df["eer"]) * 100 + 10)
    fig.savefig(FIGURES_DIR / "corrected_phone_robustness_eer.png")
    plt.close(fig)
    print("  corrected_phone_robustness_eer.png")

    # --- Fig 4: Corrected dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) EER
    ax = axes[0, 0]
    for model in sd_df["model"].unique():
        sub = sd_df[sd_df["model"] == model].sort_values("window_sec", ascending=False)
        ls = "--" if model == "cnn" else "-"
        ax.plot(sub["window_sec"].to_numpy(), (sub["eer"] * 100).to_numpy(),
                marker=MARKERS.get(model, "o"), linewidth=2, markersize=7,
                color=MODEL_COLORS.get(model, "gray"),
                label=MODEL_LABELS.get(model, model), linestyle=ls)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("EER (%)")
    ax.set_title("(a) EER vs Window Size (Corrected)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=-1)

    # (b) Phone robustness
    ax = axes[0, 1]
    for i, model in enumerate(models):
        vals = []
        for cond in conditions:
            row = ph_df[(ph_df["model"] == model) & (ph_df["eval_condition"] == cond)]
            vals.append(row["eer"].values[0] * 100 if len(row) > 0 else 0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        hatch = "//" if model == "cnn" else ""
        ax.bar(x + offset, vals, bar_width,
               label=MODEL_LABELS.get(model, model),
               color=MODEL_COLORS.get(model, "gray"),
               edgecolor="white", linewidth=0.5, hatch=hatch)
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean", "Tel", "Tel+Codec"], fontsize=10)
    ax.set_ylabel("EER (%)")
    ax.set_title("(b) Phone Robustness (Corrected)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    # (c) Accuracy (y from 0)
    ax = axes[1, 0]
    for model in sd_df["model"].unique():
        sub = sd_df[sd_df["model"] == model].sort_values("window_sec", ascending=False)
        ls = "--" if model == "cnn" else "-"
        ax.plot(sub["window_sec"].to_numpy(), (sub["accuracy"] * 100).to_numpy(),
                marker=MARKERS.get(model, "o"), linewidth=2, markersize=7,
                color=MODEL_COLORS.get(model, "gray"),
                label=MODEL_LABELS.get(model, model), linestyle=ls)
    ax.set_xlabel("Window (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(c) Accuracy vs Window Size (Corrected)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 101)

    # (d) Protocol summary
    ax = axes[1, 1]
    ax.axis("off")
    protocol_text = [
        "CORRECTED EVALUATION PROTOCOL",
        "",
        "1. Train/Val/Eval split (80/20/held-out)",
        "2. NO calibrated noise injection",
        "3. CNN early stopping on validation loss",
        "4. EER threshold frozen from validation set",
        "5. Eval set evaluated exactly once",
        "6. Same splits for all four models",
        "7. Deterministic seeds (SEED=42)",
        "8. Accuracy graph y-axis starts at 0",
        "",
        f"Dataset: 1000 synthetic files",
        f"  Train: 480  Val: 120  Eval: 400",
        f"  Balanced 50/50 bonafide/spoof",
        "",
        "Note: Results on synthetic data only.",
        "Run on real ASVspoof 2019 LA for",
        "actual research performance claims.",
    ]
    ax.text(0.05, 0.95, "\n".join(protocol_text), transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#FFF3E0", edgecolor="#FF9800"))

    fig.suptitle("AI Voice Vishing Detection — Corrected Results Dashboard",
                 fontsize=15, y=1.01)
    fig.savefig(FIGURES_DIR / "corrected_dashboard.png")
    plt.close(fig)
    print("  corrected_dashboard.png")


# ============================================================
# MAIN
# ============================================================

def main():
    config = Config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(SEED)

    print("=" * 70)
    print("AI VOICE VISHING DETECTION — CORRECTED EXPERIMENTS")
    print(f"  Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Seed: {SEED}")
    print(f"  Protocol: Train/Val/Eval | No noise | Frozen threshold")
    print("=" * 70)

    # Verify data exists
    stats_df = load_metadata("asvspoof2019_la_audio_stats.csv", config)
    print(f"  Dataset: {len(stats_df)} files")
    print(f"  Splits: {stats_df['split'].value_counts().to_dict()}")
    print(f"  Labels: {stats_df['label'].value_counts().to_dict()}")

    # Experiment 1
    sd_df = run_short_duration_corrected(config, max_train=600, max_eval=400)

    # Experiment 2
    ph_df = run_phone_robustness_corrected(config, max_train=600, max_eval=400)

    # Generate graphs
    print("\n" + "=" * 70)
    print("GENERATING CORRECTED GRAPHS")
    print("=" * 70)
    generate_corrected_graphs(config)

    print("\n" + "=" * 70)
    print("ALL CORRECTED EXPERIMENTS COMPLETE")
    print(f"Results in: {config.output_dir}")
    print(f"Figures in: {config.output_dir / 'figures'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
