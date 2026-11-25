from pathlib import Path
import soundfile as sf
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banking_vishing"
    / "audio"
)

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".flac"
}


def inspect_directory(directory, expected_label):

    files = []

    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))

    print("\n" + "=" * 60)
    print(f"{expected_label.upper()} AUDIO")
    print("=" * 60)

    print("Files:", len(files))

    records = []

    for audio_file in sorted(files):

        try:

            info = sf.info(str(audio_file))

            records.append({
                "file": audio_file.name,
                "label": expected_label,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration": info.duration,
                "format": info.format,
                "subtype": info.subtype
            })

        except Exception as e:

            print(
                f"ERROR: {audio_file.name} -> {e}"
            )

    return records


bonafide_records = inspect_directory(
    AUDIO_ROOT / "bonafide",
    "bonafide"
)

spoof_records = inspect_directory(
    AUDIO_ROOT / "spoof",
    "spoof"
)

records = bonafide_records + spoof_records

df = pd.DataFrame(records)


print("\n" + "=" * 60)
print("BANKING AUDIO SUMMARY")
print("=" * 60)

if df.empty:

    print("\nNo banking audio files found yet.")
    print("This is expected at the current stage.")

else:

    print("\nTotal files:", len(df))

    print("\nLabels:")
    print(df["label"].value_counts())

    print("\nSample rates:")
    print(df["sample_rate"].value_counts())

    print("\nChannels:")
    print(df["channels"].value_counts())

    print("\nDuration statistics:")
    print(df["duration"].describe())

    invalid_sr = df[df["sample_rate"] != 16000]

    invalid_channels = df[df["channels"] != 1]

    print(
        "\nFiles not at 16 kHz:",
        len(invalid_sr)
    )

    print(
        "Files not mono:",
        len(invalid_channels)
    )

print("\nValidation complete.")
