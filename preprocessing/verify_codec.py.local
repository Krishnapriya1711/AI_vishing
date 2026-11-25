from pathlib import Path
import soundfile as sf
import numpy as np


INPUT_DIR = Path("data/processed/phone_test")
CODEC_DIR = Path("data/processed/phone_codec_test")


files = sorted(INPUT_DIR.glob("*.wav"))

print(f"Checking {len(files)} files...\n")

for phone_file in files:

    codec_file = CODEC_DIR / phone_file.name

    phone_audio, phone_sr = sf.read(phone_file)
    codec_audio, codec_sr = sf.read(codec_file)

    print(phone_file.name)

    print(f"  Telephone : {phone_sr} Hz | {len(phone_audio)/phone_sr:.3f} sec")
    print(f"  Codec     : {codec_sr} Hz | {len(codec_audio)/codec_sr:.3f} sec")

    print(
        f"  Max difference: "
        f"{np.max(np.abs(phone_audio - codec_audio)):.6f}"
    )

    print()


print("Codec verification complete.")