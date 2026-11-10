from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

INPUT_DIR = Path("data/raw/LA/LA/ASVspoof2019_LA_train/flac")
OUTPUT_DIR = Path("data/processed/phone_test")

TARGET_SR = 8000

# Approximate telephone speech band
LOW_FREQ = 300
HIGH_FREQ = 3400

# Small amount of background noise
NOISE_LEVEL = 0.005


# ---------------------------------------------------------
# TELEPHONE BANDPASS FILTER
# ---------------------------------------------------------

def telephone_filter(audio, sr):
    """
    Keep approximately the telephone speech frequency band.
    """

    sos = butter(
        6,
        [LOW_FREQ, HIGH_FREQ],
        btype="bandpass",
        fs=sr,
        output="sos"
    )

    filtered = sosfilt(sos, audio)

    return filtered


# ---------------------------------------------------------
# ADD SMALL AMOUNT OF NOISE
# ---------------------------------------------------------

def add_noise(audio, noise_level=NOISE_LEVEL):
    """
    Add low-level Gaussian noise.
    """

    noise = np.random.normal(
        0,
        noise_level,
        size=audio.shape
    )

    noisy_audio = audio + noise

    return noisy_audio


# ---------------------------------------------------------
# NORMALIZE
# ---------------------------------------------------------

def normalize(audio):
    """
    Prevent clipping.
    """

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak * 0.95

    return audio


# ---------------------------------------------------------
# TELEPHONE SIMULATION
# ---------------------------------------------------------

def simulate_phone(
    input_file,
    output_file,
    add_background_noise=False
):
    """
    Convert clean audio into an approximate
    telephone-channel version.
    """

    # Load original audio
    audio, sr = librosa.load(
        input_file,
        sr=None,
        mono=True
    )

    original_duration = len(audio) / sr

    # Band-limit at original sample rate
    audio = telephone_filter(audio, sr)

    # Downsample to 8 kHz
    audio = librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=TARGET_SR
    )

    # Optional noise
    if add_background_noise:
        audio = add_noise(audio)

    # Prevent clipping
    audio = normalize(audio)

    # Save
    sf.write(
        output_file,
        audio,
        TARGET_SR
    )

    return original_duration


# ---------------------------------------------------------
# TEST ON 10 FILES
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(INPUT_DIR.glob("*.flac"))[:10]

    print("=" * 60)
    print("PHONE CONDITION SIMULATOR TEST")
    print("=" * 60)

    print("\nInput files:", len(files))
    print("Output directory:", OUTPUT_DIR)

    for i, input_file in enumerate(files, start=1):

        output_file = (
            OUTPUT_DIR /
            f"{input_file.stem}_phone.wav"
        )

        duration = simulate_phone(
            input_file,
            output_file,
            add_background_noise=False
        )

        print(
            f"[{i:02d}/10] "
            f"{input_file.name} "
            f"-> {output_file.name} "
            f"({duration:.2f} sec)"
        )

    print("\nTest generation complete.")


if __name__ == "__main__":
    main()