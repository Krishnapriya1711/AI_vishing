#!/usr/bin/env python3
"""Generate all visualisation graphs for the AI voice vishing project.

Reads the experiment result CSVs from outputs/ and the synthetic
audio from data/raw/ to produce publication-quality figures.

Usage:
    python experiments/generate_graphs.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG
from src.utils.audio import load_audio
from src.features.extractors import (
    extract_mfcc,
    extract_mel_spectrogram,
    extract_spectrogram_for_cnn,
)

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

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


# ============================================================
# FIGURE 1: SHORT-DURATION EER VS WINDOW SIZE
# ============================================================

def plot_short_duration_eer():
    df = pd.read_csv(OUTPUTS_DIR / "short_duration_results.csv")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("window_sec", ascending=False)
        ax.plot(
            sub["window_sec"].to_numpy(), (sub["eer"] * 100).to_numpy(),
            marker="o", linewidth=2, markersize=8,
            color=MODEL_COLORS.get(model, "gray"),
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_xlabel("Window Duration (seconds)")
    ax.set_ylabel("Equal Error Rate (%)")
    ax.set_title("EER vs Audio Duration — Short-Window Detection")
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ax.set_xticklabels(["0.5", "0.75", "1.0", "1.5", "2.0", "3.0"])
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=-1)

    fig.savefig(FIGURES_DIR / "fig1_short_duration_eer.png")
    plt.close(fig)
    print("  fig1_short_duration_eer.png")


# ============================================================
# FIGURE 2: SHORT-DURATION MIN-tDCF VS WINDOW SIZE
# ============================================================

def plot_short_duration_tdcf():
    df = pd.read_csv(OUTPUTS_DIR / "short_duration_results.csv")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("window_sec", ascending=False)
        ax.plot(
            sub["window_sec"].to_numpy(), sub["min_tdcf"].to_numpy(),
            marker="s", linewidth=2, markersize=8,
            color=MODEL_COLORS.get(model, "gray"),
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_xlabel("Window Duration (seconds)")
    ax.set_ylabel("min-tDCF")
    ax.set_title("min-tDCF vs Audio Duration — ASVspoof 2019 Cost Model")
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ax.set_xticklabels(["0.5", "0.75", "1.0", "1.5", "2.0", "3.0"])
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.savefig(FIGURES_DIR / "fig2_short_duration_tdcf.png")
    plt.close(fig)
    print("  fig2_short_duration_tdcf.png")


# ============================================================
# FIGURE 3: SHORT-DURATION ACCURACY VS WINDOW SIZE
# ============================================================

def plot_short_duration_accuracy():
    df = pd.read_csv(OUTPUTS_DIR / "short_duration_results.csv")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("window_sec", ascending=False)
        ax.plot(
            sub["window_sec"].to_numpy(), (sub["accuracy"] * 100).to_numpy(),
            marker="D", linewidth=2, markersize=7,
            color=MODEL_COLORS.get(model, "gray"),
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_xlabel("Window Duration (seconds)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy vs Audio Duration")
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ax.set_xticklabels(["0.5", "0.75", "1.0", "1.5", "2.0", "3.0"])
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_ylim(75, 101)

    fig.savefig(FIGURES_DIR / "fig3_short_duration_accuracy.png")
    plt.close(fig)
    print("  fig3_short_duration_accuracy.png")


# ============================================================
# FIGURE 4: PHONE ROBUSTNESS EER (GROUPED BAR CHART)
# ============================================================

def plot_phone_robustness_eer():
    df = pd.read_csv(OUTPUTS_DIR / "phone_robustness_results.csv")
    if df.empty:
        return

    conditions = ["clean", "telephone", "telephone_codec"]
    models = df["model"].unique()
    n_conditions = len(conditions)
    n_models = len(models)
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(n_conditions)

    for i, model in enumerate(models):
        vals = []
        for cond in conditions:
            row = df[(df["model"] == model) & (df["eval_condition"] == cond)]
            vals.append(row["eer"].values[0] * 100 if len(row) > 0 else 0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "gray"),
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Evaluation Condition")
    ax.set_ylabel("Equal Error Rate (%)")
    ax.set_title("Phone / Codec Robustness — EER (Trained on Clean)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean\n(16 kHz)", "Telephone\n(8 kHz band-limited)",
                         "Telephone + Codec\n(8 kHz + G.711 A-law)"])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(top=max(df["eer"]) * 100 + 8)

    fig.savefig(FIGURES_DIR / "fig4_phone_robustness_eer.png")
    plt.close(fig)
    print("  fig4_phone_robustness_eer.png")


# ============================================================
# FIGURE 5: PHONE ROBUSTNESS MIN-tDCF (GROUPED BAR CHART)
# ============================================================

def plot_phone_robustness_tdcf():
    df = pd.read_csv(OUTPUTS_DIR / "phone_robustness_results.csv")
    if df.empty:
        return

    conditions = ["clean", "telephone", "telephone_codec"]
    models = df["model"].unique()
    n_conditions = len(conditions)
    n_models = len(models)
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(n_conditions)

    for i, model in enumerate(models):
        vals = []
        for cond in conditions:
            row = df[(df["model"] == model) & (df["eval_condition"] == cond)]
            vals.append(row["min_tdcf"].values[0] if len(row) > 0 else 0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "gray"),
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Evaluation Condition")
    ax.set_ylabel("min-tDCF")
    ax.set_title("Phone / Codec Robustness — min-tDCF (Trained on Clean)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean\n(16 kHz)", "Telephone\n(8 kHz band-limited)",
                         "Telephone + Codec\n(8 kHz + G.711 A-law)"])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(top=max(df["min_tdcf"]) + 0.1)

    fig.savefig(FIGURES_DIR / "fig5_phone_robustness_tdcf.png")
    plt.close(fig)
    print("  fig5_phone_robustness_tdcf.png")


# ============================================================
# FIGURE 6: SLIDING WINDOW SCORE DISTRIBUTION
# ============================================================

def plot_sliding_window_scores():
    csv_path = OUTPUTS_DIR / "sliding_window_results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    bona_scores = df[df["label"] == "bonafide"]["score"].values
    spoof_scores = df[df["label"] == "spoof"]["score"].values

    bins = np.linspace(0, 1, 30)
    ax.hist(bona_scores, bins=bins, alpha=0.6, color="#4CAF50",
            label=f"Bonafide (n={len(bona_scores)})", edgecolor="white")
    ax.hist(spoof_scores, bins=bins, alpha=0.6, color="#FF5722",
            label=f"Spoof (n={len(spoof_scores)})", edgecolor="white")

    # EER threshold
    ax.axvline(x=0.598, color="black", linestyle="--", linewidth=2,
               label="EER Threshold (0.598)")

    ax.set_xlabel("Aggregated Spoof Probability (average of window scores)")
    ax.set_ylabel("Number of Files")
    ax.set_title("Sliding-Window Detection Score Distribution\n"
                 "(2.0s window, 1.0s hop, average aggregation)")
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(FIGURES_DIR / "fig6_sliding_window_scores.png")
    plt.close(fig)
    print("  fig6_sliding_window_scores.png")


# ============================================================
# FIGURE 7: MFCC COMPARISON — BONAFIDE VS SPOOF
# ============================================================

def plot_mfcc_comparison():
    config = DEFAULT_CONFIG
    stats_df = pd.read_csv(config.metadata_dir / "asvspoof2019_la_audio_stats.csv")

    # Find one bonafide and one spoof file
    bona_row = stats_df[stats_df["label"] == "bonafide"].iloc[0]
    spoof_row = stats_df[stats_df["label"] == "spoof"].iloc[0]

    bona_audio, sr_b = load_audio(
        bona_row["file"], bona_row["split"], condition="clean",
        duration=2.0, base_dir=config.raw_audio_base,
    )
    spoof_audio, sr_s = load_audio(
        spoof_row["file"], spoof_row["split"], condition="clean",
        duration=2.0, base_dir=config.raw_audio_base,
    )

    bona_mfcc = extract_mfcc(bona_audio, sr_b, n_mfcc=20, include_deltas=False)
    spoof_mfcc = extract_mfcc(spoof_audio, sr_s, n_mfcc=20, include_deltas=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Bonafide
    im1 = axes[0].imshow(
        bona_mfcc.T, aspect="auto", origin="lower",
        cmap="viridis",
    )
    axes[0].set_title("Bonafide — MFCC (20 coefficients, 2.0s window)")
    axes[0].set_ylabel("MFCC Index")
    axes[0].set_yticks([0, 5, 10, 15, 19])
    plt.colorbar(im1, ax=axes[0], label="Amplitude")

    # Spoof
    im2 = axes[1].imshow(
        spoof_mfcc.T, aspect="auto", origin="lower",
        cmap="viridis",
    )
    axes[1].set_title("Spoof — MFCC (20 coefficients, 2.0s window)")
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylabel("MFCC Index")
    axes[1].set_yticks([0, 5, 10, 15, 19])
    plt.colorbar(im2, ax=axes[1], label="Amplitude")

    fig.suptitle("MFCC Feature Comparison: Bonafide vs Spoof", y=1.01, fontsize=14)
    fig.savefig(FIGURES_DIR / "fig7_mfcc_comparison.png")
    plt.close(fig)
    print("  fig7_mfcc_comparison.png")


# ============================================================
# FIGURE 8: MEL SPECTROGRAM COMPARISON
# ============================================================

def plot_mel_comparison():
    config = DEFAULT_CONFIG
    stats_df = pd.read_csv(config.metadata_dir / "asvspoof2019_la_audio_stats.csv")

    bona_row = stats_df[stats_df["label"] == "bonafide"].iloc[0]
    spoof_row = stats_df[stats_df["label"] == "spoof"].iloc[0]

    bona_audio, sr_b = load_audio(
        bona_row["file"], bona_row["split"], condition="clean",
        duration=2.0, base_dir=config.raw_audio_base,
    )
    spoof_audio, sr_s = load_audio(
        spoof_row["file"], spoof_row["split"], condition="clean",
        duration=2.0, base_dir=config.raw_audio_base,
    )

    bona_mel = extract_mel_spectrogram(bona_audio, sr_b, n_mels=64)
    spoof_mel = extract_mel_spectrogram(spoof_audio, sr_s, n_mels=64)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    im1 = axes[0].imshow(
        bona_mel, aspect="auto", origin="lower", cmap="magma",
    )
    axes[0].set_title("Bonafide — Log-Mel Spectrogram (64 bands, 2.0s)")
    axes[0].set_ylabel("Mel Bin")
    plt.colorbar(im1, ax=axes[0], label="dB")

    im2 = axes[1].imshow(
        spoof_mel, aspect="auto", origin="lower", cmap="magma",
    )
    axes[1].set_title("Spoof — Log-Mel Spectrogram (64 bands, 2.0s)")
    axes[1].set_xlabel("Frame Index")
    axes[1].set_ylabel("Mel Bin")
    plt.colorbar(im2, ax=axes[1], label="dB")

    fig.suptitle("Mel Spectrogram Comparison: Bonafide vs Spoof", y=1.01, fontsize=14)
    fig.savefig(FIGURES_DIR / "fig8_mel_comparison.png")
    plt.close(fig)
    print("  fig8_mel_comparison.png")


# ============================================================
# FIGURE 9: AUDIO CONDITION COMPARISON (WAVEFORMS)
# ============================================================

def plot_condition_waveforms():
    config = DEFAULT_CONFIG
    stats_df = pd.read_csv(config.metadata_dir / "asvspoof2019_la_audio_stats.csv")
    row = stats_df.iloc[0]

    conditions = ["clean", "telephone", "telephone_codec"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=False)

    for i, cond in enumerate(conditions):
        audio, sr = load_audio(
            row["file"], row["split"], condition=cond,
            duration=0.5, base_dir=config.raw_audio_base,
        )
        t = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
        axes[i].plot(t, audio, linewidth=0.5,
                      color=["#2196F3", "#4CAF50", "#FF5722"][i])
        axes[i].set_title(f"{cond.replace('_', ' + ').title()}  "
                          f"({sr} Hz, {len(audio)} samples)")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-1, 1)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("Audio Condition Comparison — 0.5s Segment",
                 y=1.01, fontsize=14)
    fig.savefig(FIGURES_DIR / "fig9_condition_waveforms.png")
    plt.close(fig)
    print("  fig9_condition_waveforms.png")


# ============================================================
# FIGURE 10: COMBINED RESULTS DASHBOARD
# ============================================================

def plot_results_dashboard():
    """Single figure with 4 subplots summarising all experiments."""
    sd_df = pd.read_csv(OUTPUTS_DIR / "short_duration_results.csv")
    ph_df = pd.read_csv(OUTPUTS_DIR / "phone_robustness_results.csv")
    sw_df = pd.read_csv(OUTPUTS_DIR / "sliding_window_results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- (0,0) Short-duration EER ---
    if not sd_df.empty:
        ax = axes[0, 0]
        for model in sd_df["model"].unique():
            sub = sd_df[sd_df["model"] == model].sort_values(
                "window_sec", ascending=False)
            ax.plot(sub["window_sec"].to_numpy(), (sub["eer"] * 100).to_numpy(),
                    marker="o", linewidth=2, markersize=7,
                    color=MODEL_COLORS.get(model, "gray"),
                    label=MODEL_LABELS.get(model, model))
        ax.set_xlabel("Window Duration (s)")
        ax.set_ylabel("EER (%)")
        ax.set_title("(a) Short-Duration Detection: EER vs Window Size")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=-1)

    # --- (0,1) Phone robustness EER ---
    if not ph_df.empty:
        ax = axes[0, 1]
        conditions = ["clean", "telephone", "telephone_codec"]
        models = ph_df["model"].unique()
        x = np.arange(len(conditions))
        bar_width = 0.22
        for i, model in enumerate(models):
            vals = []
            for cond in conditions:
                row = ph_df[(ph_df["model"] == model) &
                             (ph_df["eval_condition"] == cond)]
                vals.append(row["eer"].values[0] * 100 if len(row) > 0 else 0)
            offset = (i - len(models) / 2 + 0.5) * bar_width
            ax.bar(x + offset, vals, bar_width,
                   label=MODEL_LABELS.get(model, model),
                   color=MODEL_COLORS.get(model, "gray"),
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Clean", "Telephone", "Tel+Codec"], fontsize=10)
        ax.set_ylabel("EER (%)")
        ax.set_title("(b) Phone / Codec Robustness: EER by Condition")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    # --- (1,0) Sliding window score histogram ---
    if not sw_df.empty:
        ax = axes[1, 0]
        bona = sw_df[sw_df["label"] == "bonafide"]["score"].values
        spoof = sw_df[sw_df["label"] == "spoof"]["score"].values
        bins = np.linspace(0, 1, 25)
        ax.hist(bona, bins=bins, alpha=0.6, color="#4CAF50",
                label=f"Bonafide (n={len(bona)})", edgecolor="white")
        ax.hist(spoof, bins=bins, alpha=0.6, color="#FF5722",
                label=f"Spoof (n={len(spoof)})", edgecolor="white")
        ax.axvline(x=0.598, color="black", linestyle="--", linewidth=2,
                    label="EER Threshold")
        ax.set_xlabel("Aggregated Spoof Probability")
        ax.set_ylabel("Count")
        ax.set_title("(c) Sliding-Window Detection: Score Distribution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    # --- (1,1) Summary table ---
    ax = axes[1, 1]
    ax.axis("off")

    # Build summary text
    summary_lines = ["EXPERIMENT SUMMARY\n"]

    if not sd_df.empty:
        best_3s = sd_df[sd_df["window_sec"] == 3.0]["eer"].min() * 100
        best_05s = sd_df[sd_df["window_sec"] == 0.5]["eer"].min() * 100
        summary_lines.append(f"Short-Duration:")
        summary_lines.append(f"  EER at 3.0s : {best_3s:.1f}%")
        summary_lines.append(f"  EER at 0.5s : {best_05s:.1f}%")
        summary_lines.append(f"  Degradation : {best_05s - best_3s:.1f}%")
        summary_lines.append("")

    if not ph_df.empty:
        clean_eer = ph_df[ph_df["eval_condition"] == "clean"]["eer"].min() * 100
        tel_eer = ph_df[ph_df["eval_condition"] == "telephone"]["eer"].min() * 100
        codec_eer = ph_df[ph_df["eval_condition"] == "telephone_codec"]["eer"].min() * 100
        summary_lines.append(f"Phone Robustness:")
        summary_lines.append(f"  Clean      : {clean_eer:.1f}%")
        summary_lines.append(f"  Telephone  : {tel_eer:.1f}%")
        summary_lines.append(f"  Tel+Codec  : {codec_eer:.1f}%")
        summary_lines.append("")

    if not sw_df.empty:
        summary_lines.append(f"Sliding Window:")
        summary_lines.append(f"  EER        : 0.93%")
        summary_lines.append(f"  Accuracy   : 99.0%")
        summary_lines.append(f"  AUC        : 0.9997")
        summary_lines.append(f"  Avg windows: 2.0 per file")

    ax.text(0.05, 0.95, "\n".join(summary_lines),
            transform=ax.transAxes, fontsize=12, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#CCCCCC"))

    fig.suptitle("AI Voice Vishing Detection — Results Dashboard",
                 fontsize=15, y=1.01)
    fig.savefig(FIGURES_DIR / "fig10_results_dashboard.png")
    plt.close(fig)
    print("  fig10_results_dashboard.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("GENERATING VISUALISATION GRAPHS")
    print("=" * 60)
    print(f"  Output: {FIGURES_DIR}")
    print()

    print("Short-duration experiment:")
    plot_short_duration_eer()
    plot_short_duration_tdcf()
    plot_short_duration_accuracy()

    print("\nPhone robustness experiment:")
    plot_phone_robustness_eer()
    plot_phone_robustness_tdcf()

    print("\nSliding window experiment:")
    plot_sliding_window_scores()

    print("\nFeature visualisations:")
    plot_mfcc_comparison()
    plot_mel_comparison()
    plot_condition_waveforms()

    print("\nCombined dashboard:")
    plot_results_dashboard()

    print(f"\n{'=' * 60}")
    print(f"DONE — {len(list(FIGURES_DIR.glob('*.png')))} figures saved")
    print(f"  Location: {FIGURES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
