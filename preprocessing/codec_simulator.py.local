from pathlib import Path
import numpy as np
import soundfile as sf


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR = Path("data/processed/phone_test")
OUTPUT_DIR = Path("data/processed/phone_codec_test")

TARGET_SR = 8000


# ============================================================
# G.711 A-LAW ENCODER
# ============================================================

def alaw_encode(samples):

    samples = np.asarray(samples, dtype=np.int16)

    encoded = np.zeros(len(samples), dtype=np.uint8)

    for i, sample in enumerate(samples):

        sample = int(sample)

        # Sign
        sign = 0x80 if sample < 0 else 0x00

        # Absolute value
        value = abs(sample)

        # Limit
        if value > 32635:
            value = 32635

        # Calculate A-law value
        if value >= 256:

            exponent = 7
            mask = 0x4000

            while exponent > 0 and (value & mask) == 0:
                exponent -= 1
                mask >>= 1

            mantissa = (value >> (exponent + 3)) & 0x0F

            alaw = sign | (exponent << 4) | mantissa

        else:

            alaw = sign | (value >> 4)

        # A-law XOR mask
        encoded[i] = alaw ^ 0x55

    return encoded


# ============================================================
# G.711 A-LAW DECODER
# ============================================================

def alaw_decode(encoded):

    encoded = np.asarray(encoded, dtype=np.uint8)

    decoded = np.zeros(len(encoded), dtype=np.int16)

    for i, byte in enumerate(encoded):

        # Convert NumPy uint8 to Python integer
        byte = int(byte)

        # Reverse A-law XOR
        byte ^= 0x55

        # Extract components
        sign = byte & 0x80
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F

        # Decode
        if exponent == 0:

            value = (mantissa << 4) + 8

        else:

            value = ((mantissa << 4) + 0x108) << (exponent - 1)

        # Apply sign
        if sign:
            value = -value

        decoded[i] = np.clip(
            value,
            -32768,
            32767
        )

    return decoded


# ============================================================
# APPLY G.711 A-LAW
# ============================================================

def apply_alaw(input_file, output_file):

    # Read telephone audio
    audio, sr = sf.read(input_file)

    # Make sure audio is mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Check sampling rate
    if sr != TARGET_SR:

        raise ValueError(
            f"Expected {TARGET_SR} Hz, "
            f"but got {sr} Hz"
        )

    # --------------------------------------------------------
    # FLOAT [-1, 1] → INT16
    # --------------------------------------------------------

    audio_int16 = np.int16(
        np.clip(audio, -1.0, 1.0) * 32767
    )

    # --------------------------------------------------------
    # A-LAW ENCODE
    # --------------------------------------------------------

    encoded = alaw_encode(audio_int16)

    # --------------------------------------------------------
    # A-LAW DECODE
    # --------------------------------------------------------

    decoded = alaw_decode(encoded)

    # --------------------------------------------------------
    # INT16 → FLOAT
    # --------------------------------------------------------

    output_audio = (
        decoded.astype(np.float32) / 32767.0
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    sf.write(
        output_file,
        output_audio,
        TARGET_SR
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        INPUT_DIR.glob("*.wav")
    )

    print(f"Found {len(files)} WAV files\n")

    if len(files) == 0:

        print(
            "No WAV files found in:"
        )

        print(INPUT_DIR)

        return

    for input_file in files:

        output_file = (
            OUTPUT_DIR /
            input_file.name
        )

        try:

            apply_alaw(
                input_file,
                output_file
            )

            print(
                f"Processed: "
                f"{input_file.name}"
            )

        except Exception as e:

            print(
                f"ERROR: "
                f"{input_file.name}"
            )

            print(e)

    print(
        "\nG.711 A-law codec simulation complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()