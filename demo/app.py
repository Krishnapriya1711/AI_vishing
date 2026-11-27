#!/usr/bin/env python3
"""
VishGuard Demo — AI Voice Vishing Detector
==========================================
Two detection signals combined:

  1. ACOUSTIC  — Mel-CNN ensemble trained on ASVspoof 2019 detects AI-generated
                 voices by analysing mel-spectrogram patterns window by window.

  2. SEMANTIC  — Whisper (base) transcribes the audio; a rule-based keyword
                 detector flags vishing-specific language (credential demands,
                 impersonation, urgency, transfer requests, threats).

Final verdict fires if EITHER signal triggers.  Both indicators are shown
separately so the user knows which evidence drove the alarm.

Run from the repo root:
    python3 demo/app.py

Limitations:
  - Acoustic model trained on ASVspoof 2019 (19 TTS/VC systems). Will not detect
    modern neural TTS (OpenAI, ElevenLabs, newer RVC) — shown as "Model blind spot".
  - Semantic detector is English-only and rule-based; novel phrasing may be missed.
  - Research demonstration only. Not intended for production use.
"""

import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── repo path setup ──────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "experiments", ROOT / "features", DEMO_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.chdir(ROOT)

import extract_features as fx                            # noqa: E402
from stream_common import SR, WIN_SEC, HOP_SEC, WARNING, Scorer, discover  # noqa: E402
from vishing_keywords import analyse_transcript          # noqa: E402

# ── load acoustic models once at startup ─────────────────────────────────────
MODEL_DIRS = [ROOT / "outputs" / "seeds5" / "models",
              ROOT / "outputs" / "models"]
_all_models = discover(MODEL_DIRS)

CNN_SCORERS = {
    "aug":   [Scorer(p) for (reg, kind, _), p in _all_models.items()
              if reg == "aug"   and kind == "cnn"],
    "clean": [Scorer(p) for (reg, kind, _), p in _all_models.items()
              if reg == "clean" and kind == "cnn"],
}

# ── load per-model normalisation stats ───────────────────────────────────────
def _load_norm_stats(regime: str):
    path = DEMO_DIR / f"scorer_norm_stats_{regime}.json"
    return json.loads(path.read_text()) if path.exists() else None

NORM_STATS   = {"aug": _load_norm_stats("aug"), "clean": _load_norm_stats("clean")}
NORM_EER_THR = {"aug": 0.85, "clean": 0.75}

# Hard-alarm threshold on the normalised ensemble score.
# Chosen at the call-level operating point on banking vishing + VishGuard data:
# thr=0.3 → ~35% TPR, ~10% FPR on out-of-distribution spoof calls.
HARD_ALARM_OFFSET = 0.3

# VAD gate: skip windows with RMS below this — silence peak-normalises to
# full-scale noise which the CNN scores as strongly spoofed (+7 to +9).
VAD_RMS_THR = 0.01

# ── load Whisper model once ───────────────────────────────────────────────────
print("Loading Whisper (base)…", flush=True)
import whisper as _whisper
WHISPER_MODEL = _whisper.load_model("base")
print("Whisper ready.", flush=True)


# ── audio helpers ─────────────────────────────────────────────────────────────
def apply_channel(audio: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return audio
    import librosa
    os.chdir(ROOT)
    from preprocessing.audio_loader import apply_alaw, normalize, telephone_filter
    a = telephone_filter(audio, SR)
    a = librosa.resample(a, orig_sr=SR, target_sr=8000)
    a = normalize(a)
    if condition == "telephone_codec":
        a = apply_alaw(a)
    a = librosa.resample(a, orig_sr=8000, target_sr=SR)
    return a.astype(np.float32)


def load_audio(path: str) -> np.ndarray:
    import librosa
    a, sr = librosa.load(path, sr=None, mono=True)
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a.astype(np.float32)


# ── acoustic scoring ──────────────────────────────────────────────────────────
def _ensemble_norm_score(mfcc, mel, scorers, norm_stats, eer_thr):
    per_model = []
    for i, sc in enumerate(scorers):
        raw_m = float(sc.score(mfcc, mel)[0]) - sc.thr
        if norm_stats is not None and i < len(norm_stats):
            st = norm_stats[i]
            raw_m = (raw_m - st["mean"]) / (st["std"] + 1e-6)
        per_model.append(raw_m)
    return float(np.mean(per_model)) - eer_thr


def score_call(audio: np.ndarray, scorers, rule: str, k_consec: int,
               regime: str = "aug"):
    n_win  = int(WIN_SEC * SR)
    n_hop  = int(HOP_SEC * SR)
    starts = list(range(0, len(audio) - n_win + 1, n_hop))
    if not starts:
        return [], None, []

    norm_stats = NORM_STATS.get(regime)
    eer_thr    = NORM_EER_THR.get(regime, 0.85)
    hard_thr   = HARD_ALARM_OFFSET
    margins, run, alarm_at, times = [], 0, None, []

    for idx, s in enumerate(starts):
        raw_win = audio[s: s + n_win]
        rms = float(np.sqrt(np.mean(raw_win ** 2)))
        if rms < VAD_RMS_THR:
            run = 0
            continue

        w   = fx.peak_norm(raw_win)
        mf  = fx.mfcc_stats(w)[None]
        mel = fx.log_mel(w, fx.frames_for(WIN_SEC))[None]
        m   = _ensemble_norm_score(mf, mel, scorers, norm_stats, eer_thr)

        t_end = WIN_SEC + idx * HOP_SEC
        margins.append(m)
        times.append(t_end)
        above = m >= hard_thr
        run   = run + 1 if above else 0

        if alarm_at is None:
            if rule == "consecutive" and run >= k_consec:
                alarm_at = t_end
            elif rule == "vote":
                frac = float(np.mean(np.array(margins) >= 0))
                if len(margins) >= k_consec and frac >= 0.5:
                    alarm_at = t_end

    # mean rule fires on final mean (not running mean — avoids early FP)
    if rule == "mean" and alarm_at is None and margins:
        if float(np.mean(margins)) >= hard_thr:
            alarm_at = times[-1]

    return margins, alarm_at, times


# ── plot ──────────────────────────────────────────────────────────────────────
def make_plot(margins, times, alarm_at, duration_sec):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    def bar_color(m):
        if m >= HARD_ALARM_OFFSET: return "#ef5350"
        if m >= 0:                 return "#ffa726"
        return "#66bb6a"

    ax.bar(times, margins, width=HOP_SEC * 0.85,
           color=[bar_color(m) for m in margins], zorder=3)
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=2)
    ax.axhline(HARD_ALARM_OFFSET, color="#ef5350", linewidth=1.0, linestyle=":", zorder=2)

    if alarm_at is not None:
        ax.axvline(alarm_at, color="#ff9800", linewidth=2, linestyle=":", zorder=4)
        ax.text(alarm_at + 0.05, ax.get_ylim()[1] * 0.9,
                f"Alarm @ {alarm_at:.1f}s", color="#ff9800", fontsize=9)

    ax.set_xlim(0, max(times[-1] + HOP_SEC, duration_sec))
    ax.set_xlabel("Time (s)", color="white", fontsize=10)
    ax.set_ylabel("Normalised score", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.legend(handles=[
        mpatches.Patch(color="#66bb6a", label="Likely real"),
        mpatches.Patch(color="#ffa726", label="Uncertain"),
        mpatches.Patch(color="#ef5350", label="Likely AI-generated"),
    ], fontsize=9, facecolor="#1e1e2e", labelcolor="white", framealpha=0.8, loc="upper right")
    ax.set_title("Window-by-window voice scores", color="white", fontsize=11)
    fig.tight_layout()

    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))


# ── combined analysis ─────────────────────────────────────────────────────────
def analyse(audio_path, regime, condition, rule, k_consec):
    if audio_path is None:
        return "⬆️  Please upload or record audio first.", None, "", ""

    try:
        t0      = time.perf_counter()
        raw     = load_audio(audio_path)
        audio   = apply_channel(raw, condition)
        duration = len(raw) / SR

        scorers = CNN_SCORERS.get(regime, [])
        if not scorers:
            return "❌  No models found for this regime.", None, "", ""

        # ── 1. Acoustic analysis ──────────────────────────────────────────
        margins, alarm_at, times = score_call(audio, scorers, rule, int(k_consec), regime)
        if not margins:
            return "⚠️  Audio too short — need at least 1 second.", None, "", ""

        arr         = np.array(margins)
        mean_margin = float(arr.mean())
        frac_eer    = float(np.mean(arr >= 0))
        frac_hard   = float(np.mean(arr >= HARD_ALARM_OFFSET))
        score_std   = float(arr.std())
        no_signal   = (mean_margin < -1.2) and (score_std <= 0.8)

        acoustic_alarm    = (alarm_at is not None) or (frac_hard >= 0.3)
        acoustic_suspicious = frac_eer >= 0.5

        # ── 2. Semantic analysis (transcribe + keyword match) ─────────────
        transcript = WHISPER_MODEL.transcribe(raw, language="en")["text"].strip()
        kw = analyse_transcript(transcript)

        elapsed = time.perf_counter() - t0

        # ── 3. Combined verdict ───────────────────────────────────────────
        verdict = _build_verdict(
            acoustic_alarm, acoustic_suspicious, no_signal,
            mean_margin, frac_eer, frac_hard, alarm_at,
            kw, elapsed, duration
        )

        # ── 4. Stat line ──────────────────────────────────────────────────
        stats = (
            f"**Duration:** {duration:.1f} s  |  "
            f"**Windows:** {len(margins)}  |  "
            f"**Voice score:** {mean_margin:+.3f} (std {score_std:.2f})  |  "
            f"**Above alarm line:** {frac_hard*100:.0f}%  |  "
            f"**Vishing risk:** {kw['risk_level']} ({kw['risk_score']:.2f})  |  "
            f"**Time:** {elapsed*1000:.0f} ms"
        )

        # ── 5. Transcript display ─────────────────────────────────────────
        transcript_md = _format_transcript(transcript, kw)

        return verdict, make_plot(margins, times, alarm_at, duration), stats, transcript_md

    except Exception as e:
        import traceback
        return f"❌ **Error:** `{e}`\n\n```\n{traceback.format_exc()}\n```", None, "", ""


def _build_verdict(acoustic_alarm, acoustic_suspicious, no_signal,
                   mean_margin, frac_eer, frac_hard, alarm_at,
                   kw, elapsed, duration):
    """Combine acoustic and semantic signals into a single verdict string."""

    sem_alarm  = kw["triggered"]                     # HIGH or MEDIUM keyword match
    sem_risk   = kw["risk_level"]
    highlights = kw["highlights"][:4]

    # ── Hard alarm: either signal fires ──────────────────────────────────
    if acoustic_alarm or sem_alarm:
        lines = [f"# 🚨 ALARM — Vishing attempt likely detected\n\n> {WARNING}\n"]

        if acoustic_alarm:
            lines.append(
                f"**🔊 Voice:** AI-generated voice signal detected "
                f"({frac_hard*100:.0f}% of windows above alarm threshold, "
                f"mean score {mean_margin:+.2f})."
            )
        else:
            lines.append(
                f"**🔊 Voice:** Acoustic model inconclusive "
                f"(mean score {mean_margin:+.2f}) — alarm driven by content alone."
            )

        if sem_alarm:
            hi_str = " · ".join(f'`{h}`' for h in highlights) if highlights else ""
            lines.append(
                f"**📝 Content ({sem_risk} risk):** Vishing language detected — "
                f"{', '.join(kw['matched'])}."
                + (f"\n  > Flagged phrases: {hi_str}" if hi_str else "")
            )
        else:
            lines.append(f"**📝 Content:** No vishing keywords detected ({sem_risk} risk).")

        return "\n\n".join(lines)

    # ── Suspicious (acoustic) + possible semantic ─────────────────────────
    if acoustic_suspicious or sem_risk in ("LOW", "MEDIUM"):
        lines = ["## ⚠️ Suspicious — treat with caution\n"]

        if acoustic_suspicious:
            lines.append(
                f"**🔊 Voice:** {frac_eer*100:.0f}% of windows above EER line "
                f"(mean {mean_margin:+.2f}). "
                "May be AI-generated by a system the model partially recognises."
            )
        else:
            lines.append(f"**🔊 Voice:** No strong AI signal (mean {mean_margin:+.2f}).")

        hi_str = " · ".join(f'`{h}`' for h in highlights) if highlights else ""
        lines.append(
            f"**📝 Content ({sem_risk} risk):** {kw['summary']}"
            + (f"\n  > {hi_str}" if hi_str else "")
        )
        return "\n\n".join(lines)

    # ── Model blind spot ──────────────────────────────────────────────────
    if no_signal:
        return (
            "## ⚠️ Model blind spot\n\n"
            "**🔊 Voice:** All windows score far below threshold with low variance.  "
            "Pattern typical of **modern neural TTS (OpenAI, ElevenLabs, newer RVC)** "
            "not in the ASVspoof 2019 training set.  "
            "A negative acoustic result here **does not mean the audio is genuine.**\n\n"
            f"**📝 Content ({sem_risk} risk):** {kw['summary']}"
        )

    # ── Low confidence (near zero) ────────────────────────────────────────
    if mean_margin > -0.3:
        return (
            f"## 🟡 Low confidence — leaning real\n\n"
            f"**🔊 Voice:** Score {mean_margin:+.2f} — below alarm threshold but not far.  "
            "May be AI-generated by a system the model struggles with.\n\n"
            f"**📝 Content ({sem_risk} risk):** {kw['summary']}"
        )

    # ── All clear ─────────────────────────────────────────────────────────
    return (
        f"## ✅ No alarm — audio appears genuine\n\n"
        f"**🔊 Voice:** Score {mean_margin:+.2f} — below detection threshold.  "
        "No strong AI-voice signal.\n\n"
        f"**📝 Content ({sem_risk} risk):** {kw['summary']}"
    )


def _format_transcript(text: str, kw: dict) -> str:
    """Return transcript as markdown, with flagged phrases bolded."""
    if not text:
        return "_No transcript available._"
    # Bold the flagged highlight phrases in the transcript
    highlighted = text
    for phrase in kw.get("highlights", []):
        import re
        highlighted = re.sub(
            re.escape(phrase), f"**{phrase}**", highlighted, flags=re.IGNORECASE
        )
    risk_badge = {"HIGH": "🔴 HIGH", "MEDIUM": "🟠 MEDIUM",
                  "LOW": "🟡 LOW", "NONE": "🟢 NONE"}.get(kw["risk_level"], kw["risk_level"])
    return f"**Content risk: {risk_badge}**\n\n{highlighted}"


# ── shared controls ───────────────────────────────────────────────────────────
def controls_block():
    with gr.Row():
        regime = gr.Dropdown(
            choices=["aug", "clean"], value="aug",
            label="Training regime",
            info="'aug' = trained on clean + telephone + codec (recommended)",
        )
        condition = gr.Dropdown(
            choices=["clean", "telephone", "telephone_codec"], value="clean",
            label="Channel condition",
            info="Simulate the channel the recording went through",
        )
    with gr.Row():
        rule = gr.Dropdown(
            choices=["consecutive", "mean", "vote"], value="mean",
            label="Alarm rule",
            info="consecutive = k windows in a row | mean = final mean | vote = majority",
        )
        k_consec = gr.Slider(minimum=1, maximum=6, value=3, step=1,
                             label="k  (windows required to fire alarm)")
    return regime, condition, rule, k_consec


def output_block():
    verdict   = gr.Markdown()
    plot_out  = gr.Image(label="Voice detection scores", type="numpy")
    stats_md  = gr.Markdown()
    trans_md  = gr.Markdown(label="Transcript")
    return verdict, plot_out, stats_md, trans_md


# ── Gradio UI ─────────────────────────────────────────────────────────────────
DESCRIPTION = """
## 🛡️ VishGuard — AI Voice Vishing Detector

Analyses audio for **two independent risk signals**:
- 🔊 **Voice:** Mel-CNN ensemble detects AI-generated voices (ASVspoof 2019 trained)
- 📝 **Content:** Whisper transcription + keyword detector flags vishing language

An alarm fires if **either** signal triggers.

> ⚠️ Acoustic model covers ASVspoof 2019 systems only. Modern TTS (OpenAI, ElevenLabs) shows as "Model blind spot". Semantic detector is English-only. Research demo — not for production use.
"""

with gr.Blocks(title="VishGuard Demo") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.Tab("📂 Upload Audio"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_in = gr.Audio(label="Upload audio (WAV / FLAC)", type="filepath")
                    r1, c1, ru1, k1 = controls_block()
                    btn1 = gr.Button("🔍 Analyse", variant="primary")
                with gr.Column(scale=2):
                    v1, p1, s1, t1 = output_block()
            btn1.click(fn=analyse,
                       inputs=[audio_in, r1, c1, ru1, k1],
                       outputs=[v1, p1, s1, t1])

        with gr.Tab("🎙️ Live Mic"):
            gr.Markdown(
                "Press **Record**, speak for a few seconds, press **Stop**, "
                "then click **Analyse**.\n\n"
                "> Your browser will ask for microphone permission — allow it."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    mic_in = gr.Audio(label="Record from microphone",
                                      sources=["microphone"], type="filepath")
                    r2, c2, ru2, k2 = controls_block()
                    btn2 = gr.Button("🔍 Analyse Recording", variant="primary")
                with gr.Column(scale=2):
                    v2, p2, s2, t2 = output_block()
            btn2.click(fn=analyse,
                       inputs=[mic_in, r2, c2, ru2, k2],
                       outputs=[v2, p2, s2, t2])

    gr.Markdown("_Research demonstration only. Not intended for production use._")

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860,
                theme=gr.themes.Soft())
