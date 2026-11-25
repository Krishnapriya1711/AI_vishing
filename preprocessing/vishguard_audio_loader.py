from pathlib import Path

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banking_vishing"
    / "audio"
)


# ============================================================
# AUDIO SETTINGS
# ============================================================

CLEAN_SR = 16000
PHONE_SR = 8000

LOWCUT = 300
HIGHCUT = 3400
FILTER_ORDER = 6


# ============================================================
# FILE FINDER
# ============================================================

def find_audio_file(file_name, label):
    """
    Find a VISHGUARD WAV file.

    label:
        spoof
        bonafide
    """

    file_name = Path(file_name).name

    path = AUDIO_DIR / label / file_name

    if not path.exists():
        raise FileNotFoundError(
            f"VISHGUARD audio not found:\n{path}"
        )

    return path


# ============================================================
# BANDPASS FILTER
# ============================================================

def telephone_filter(audio, sr):
    """
    Apply telephone-band filtering:
    300 Hz - 3400 Hz
    """

    nyquist = sr / 2

    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist

    if high >= 1.0:
        high = 0.99

    sos = butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(sos, audio)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_audio(audio):
    """
    Peak normalize to approximately 0.95.
    """

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


# ============================================================
# MAIN LOADER
# ============================================================

def load_vishguard_audio(
    file_name,
    label,
    condition="clean",
    duration=None,
    start_sec=0.0
):
    """
    Load VISHGUARD audio dynamically.

    Parameters
    ----------
    file_name : str
        WAV filename.

    label : str
        'spoof' or 'bonafide'

    condition : str
        'clean'
        'telephone'
        'telephone_codec'

    duration : float or None
        Duration to crop in seconds.

    start_sec : float
        Starting position in seconds.

    Returns
    -------
    audio : np.ndarray
        Mono float32 audio.

    sample_rate : int
        Output sample rate.
    """

    if label not in {"spoof", "bonafide"}:
        raise ValueError(
            "label must be 'spoof' or 'bonafide'"
        )

    if condition not in {
        "clean",
        "telephone",
        "telephone_codec"
    }:
        raise ValueError(
            "condition must be clean, telephone, "
            "or telephone_codec"
        )

    path = find_audio_file(file_name, label)

    # --------------------------------------------------------
    # Load original audio WITHOUT resampling
    # --------------------------------------------------------

    audio, original_sr = librosa.load(
        path,
        sr=None,
        mono=False
    )

    # --------------------------------------------------------
    # Stereo -> mono
    # --------------------------------------------------------

    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)

    audio = audio.astype(np.float32)

    # --------------------------------------------------------
    # CLEAN CONDITION
    # --------------------------------------------------------

    if condition == "clean":

        if original_sr != CLEAN_SR:
            audio = librosa.resample(
                audio,
                orig_sr=original_sr,
                target_sr=CLEAN_SR
            )

        output_sr = CLEAN_SR

    # --------------------------------------------------------
    # TELEPHONE CONDITIONS
    # --------------------------------------------------------

    else:

        # First convert to 16 kHz
        if original_sr != CLEAN_SR:
            audio = librosa.resample(
                audio,
                orig_sr=original_sr,
                target_sr=CLEAN_SR
            )

        # Telephone frequency range
        audio = telephone_filter(
            audio,
            CLEAN_SR
        )

        # Downsample to 8 kHz
        audio = librosa.resample(
            audio,
            orig_sr=CLEAN_SR,
            target_sr=PHONE_SR
        )

        output_sr = PHONE_SR

        # ----------------------------------------------------
        # G.711 A-law simulation
        # ----------------------------------------------------

        if condition == "telephone_codec":

            # Safe approximation of int16 codec processing
            audio_int16 = np.clip(
                audio * 32767,
                -32768,
                32767
            ).astype(np.int16)

            # Quantization through int16 representation
            audio = (
                audio_int16.astype(np.float32)
                / 32767.0
            )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    audio = normalize_audio(audio)

    # --------------------------------------------------------
    # Crop requested window
    # --------------------------------------------------------

    if duration is not None:

        start_sample = int(
            round(start_sec * output_sr)
        )

        num_samples = int(
            round(duration * output_sr)
        )

        end_sample = start_sample + num_samples

        if end_sample > len(audio):
            raise ValueError(
                f"Requested window exceeds audio length: "
                f"start={start_sec}s, "
                f"duration={duration}s"
            )

        audio = audio[
            start_sample:end_sample
        ]

    return audio, output_sr