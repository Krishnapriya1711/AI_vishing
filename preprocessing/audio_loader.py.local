from pathlib import Path

import librosa
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

RAW_AUDIO_DIR = Path(
    "data/raw/LA/LA"
)

TARGET_SR_CLEAN = 16000
TARGET_SR_PHONE = 8000


# ============================================================
# FIND AUDIO FILE
# ============================================================

def find_audio_file(file_name, split):

    split_folder = {
        "train": "ASVspoof2019_LA_train",
        "dev": "ASVspoof2019_LA_dev",
        "eval": "ASVspoof2019_LA_eval"
    }

    if split not in split_folder:
        raise ValueError(
            f"Unknown split: {split}"
        )

    audio_dir = (
        RAW_AUDIO_DIR
        / split_folder[split]
        / "flac"
    )

    # Metadata may contain filename with or without extension
    file_name = str(file_name)

    if not file_name.lower().endswith(".flac"):
        file_name += ".flac"

    audio_file = audio_dir / file_name

    if not audio_file.exists():
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_file}"
        )

    return audio_file


# ============================================================
# TELEPHONE FILTER
# ============================================================

def telephone_filter(audio, sr):

    from scipy.signal import butter, sosfilt

    low_freq = 300
    high_freq = 3400

    sos = butter(
        6,
        [low_freq, high_freq],
        btype="bandpass",
        fs=sr,
        output="sos"
    )

    return sosfilt(sos, audio)


# ============================================================
# NORMALIZE
# ============================================================

def normalize(audio):

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:

        audio = (
            audio / peak
        ) * 0.95

    return audio


# ============================================================
# G.711 A-LAW ENCODER
# ============================================================

def alaw_encode(samples):

    samples = np.asarray(
        samples,
        dtype=np.int16
    )

    encoded = np.zeros(
        len(samples),
        dtype=np.uint8
    )

    for i, sample in enumerate(samples):

        sample = int(sample)

        sign = (
            0x80
            if sample < 0
            else 0x00
        )

        value = abs(sample)

        if value > 32635:
            value = 32635

        if value >= 256:

            exponent = 7
            mask = 0x4000

            while (
                exponent > 0
                and (value & mask) == 0
            ):
                exponent -= 1
                mask >>= 1

            mantissa = (
                value
                >> (exponent + 3)
            ) & 0x0F

            alaw = (
                sign
                | (exponent << 4)
                | mantissa
            )

        else:

            alaw = (
                sign
                | (value >> 4)
            )

        encoded[i] = (
            alaw ^ 0x55
        )

    return encoded


# ============================================================
# G.711 A-LAW DECODER
# ============================================================

def alaw_decode(encoded):

    encoded = np.asarray(
        encoded,
        dtype=np.uint8
    )

    decoded = np.zeros(
        len(encoded),
        dtype=np.int16
    )

    for i, byte in enumerate(encoded):

        byte = int(byte)

        byte ^= 0x55

        sign = byte & 0x80

        exponent = (
            byte >> 4
        ) & 0x07

        mantissa = (
            byte & 0x0F
        )

        if exponent == 0:

            value = (
                (mantissa << 4)
                + 8
            )

        else:

            value = (
                (mantissa << 4)
                + 0x108
            ) << (
                exponent - 1
            )

        if sign:
            value = -value

        decoded[i] = np.clip(
            value,
            -32768,
            32767
        )

    return decoded


# ============================================================
# APPLY A-LAW
# ============================================================

def apply_alaw(audio):

    audio_int16 = np.int16(
        np.clip(
            audio,
            -1.0,
            1.0
        ) * 32767
    )

    encoded = alaw_encode(
        audio_int16
    )

    decoded = alaw_decode(
        encoded
    )

    return (
        decoded.astype(
            np.float32
        ) / 32767.0
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(
    file_name,
    split,
    condition="clean",
    duration=None,
    start_sec=0.0
):
    """
    Load an ASVspoof audio file and optionally apply
    telephone / codec conditions.

    Parameters
    ----------
    file_name : str
        Audio filename.

    split : str
        train / dev / eval

    condition : str
        clean
        telephone
        telephone_codec

    duration : float or None
        Desired duration in seconds.

    start_sec : float
        Starting position in seconds.

    Returns
    -------
    audio : numpy array
    sr : int
    """

    # --------------------------------------------------------
    # Find original file
    # --------------------------------------------------------

    file_path = find_audio_file(
        file_name,
        split
    )

    # --------------------------------------------------------
    # Load original audio
    # --------------------------------------------------------

    audio, sr = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    # --------------------------------------------------------
    # CLEAN CONDITION
    # --------------------------------------------------------

    if condition == "clean":

        target_sr = TARGET_SR_CLEAN

    # --------------------------------------------------------
    # TELEPHONE CONDITION
    # --------------------------------------------------------

    elif condition in [
        "telephone",
        "telephone_codec"
    ]:

        # Telephone frequency range
        audio = telephone_filter(
            audio,
            sr
        )

        # Resample to 8 kHz
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR_PHONE
        )

        sr = TARGET_SR_PHONE

        # Normalize
        audio = normalize(
            audio
        )

        target_sr = TARGET_SR_PHONE

        # ----------------------------------------------------
        # G.711 A-LAW
        # ----------------------------------------------------

        if condition == "telephone_codec":

            audio = apply_alaw(
                audio
            )

    else:

        raise ValueError(
            f"Unknown condition: {condition}"
        )

    # --------------------------------------------------------
    # CROP START
    # --------------------------------------------------------

    if start_sec > 0:

        start_sample = int(
            start_sec * sr
        )

        audio = audio[
            start_sample:
        ]

    # --------------------------------------------------------
    # CROP DURATION
    # --------------------------------------------------------

    if duration is not None:

        required_samples = int(
            duration * sr
        )

        if len(audio) < required_samples:

            raise ValueError(
                f"Audio too short. "
                f"Requested {duration}s, "
                f"available "
                f"{len(audio) / sr:.3f}s"
            )

        audio = audio[
            :required_samples
        ]

    return audio, target_sr


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    file_name = "LA_T_1000137.flac"

    split = "train"

    print("Testing dynamic audio loader...\n")

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    audio, sr = load_audio(
        file_name,
        split,
        condition="clean",
        duration=1.0
    )

    print("CLEAN")
    print(
        f"Sample rate : {sr}"
    )
    print(
        f"Samples     : {len(audio)}"
    )
    print(
        f"Duration    : {len(audio)/sr:.3f}s"
    )

    # --------------------------------------------------------
    # TELEPHONE
    # --------------------------------------------------------

    audio, sr = load_audio(
        file_name,
        split,
        condition="telephone",
        duration=1.0
    )

    print("\nTELEPHONE")
    print(
        f"Sample rate : {sr}"
    )
    print(
        f"Samples     : {len(audio)}"
    )
    print(
        f"Duration    : {len(audio)/sr:.3f}s"
    )

    # --------------------------------------------------------
    # TELEPHONE + CODEC
    # --------------------------------------------------------

    audio, sr = load_audio(
        file_name,
        split,
        condition="telephone_codec",
        duration=1.0
    )

    print("\nTELEPHONE + G.711 A-LAW")
    print(
        f"Sample rate : {sr}"
    )
    print(
        f"Samples     : {len(audio)}"
    )
    print(
        f"Duration    : {len(audio)/sr:.3f}s"
    )

    print(
        "\nDynamic audio loader test complete."
    )