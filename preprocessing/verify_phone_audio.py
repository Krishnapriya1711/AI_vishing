from pathlib import Path

import librosa
import numpy as np


AUDIO_DIR = Path("data/processed/phone_test")


print("=" * 60)
print("VERIFYING PHONE AUDIO")
print("=" * 60)

files = sorted(AUDIO_DIR.glob("*.wav"))

print("\nFiles found:", len(files))

for file in files:

    audio, sr = librosa.load(
        file,
        sr=None,
        mono=False
    )

    if audio.ndim == 1:
        channels = 1
        samples = len(audio)
    else:
        channels = audio.shape[0]
        samples = audio.shape[-1]

    duration = samples / sr

    print(
        f"{file.name}\n"
        f"  Sample rate : {sr} Hz\n"
        f"  Channels    : {channels}\n"
        f"  Duration    : {duration:.3f} sec\n"
        f"  Min         : {np.min(audio):.4f}\n"
        f"  Max         : {np.max(audio):.4f}"
    )

print("\nVerification complete.")