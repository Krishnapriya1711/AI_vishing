from pathlib import Path
import pandas as pd

from preprocessing.vishguard_audio_loader import (
    load_vishguard_audio
)


# ============================================================
# LOAD METADATA
# ============================================================

METADATA_PATH = Path(
    "data/metadata/vishguard_final_metadata.csv"
)

df = pd.read_csv(METADATA_PATH)


# Select representative files from different
# sample-rate/channel combinations if available.

test_files = (
    df[
        ["file", "label", "sample_rate", "channels"]
    ]
    .drop_duplicates()
    .sort_values(["sample_rate", "channels"])
)


# Take several representative examples
test_files = test_files.head(8)


print("=" * 70)
print("VISHGUARD AUDIO LOADER TEST")
print("=" * 70)

print(f"Testing {len(test_files)} files")


# ============================================================
# TEST CONDITIONS
# ============================================================

conditions = [
    "clean",
    "telephone",
    "telephone_codec"
]


passed = 0
failed = 0


for _, row in test_files.iterrows():

    file_name = row["file"]
    label = row["label"]

    print("\n" + "-" * 70)

    print(
        f"File       : {file_name}"
    )

    print(
        f"Label      : {label}"
    )

    print(
        f"Original SR: {row['sample_rate']}"
    )

    print(
        f"Channels   : {row['channels']}"
    )

    for condition in conditions:

        try:

            audio, sr = load_vishguard_audio(
                file_name=file_name,
                label=label,
                condition=condition,
                duration=1.0,
                start_sec=0.0
            )

            expected_samples = sr

            checks = [
                len(audio) == expected_samples,
                audio.ndim == 1,
                audio.dtype.name == "float32",
                sr in {16000, 8000},
            ]

            if all(checks):

                print(
                    f"  {condition:18s} "
                    f"PASS | SR={sr} | "
                    f"samples={len(audio)}"
                )

                passed += 1

            else:

                print(
                    f"  {condition:18s} FAIL"
                )

                failed += 1

        except Exception as e:

            print(
                f"  {condition:18s} FAIL | {e}"
            )

            failed += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed == 0:

    print("\nVISHGUARD AUDIO LOADER PASSED")

else:

    print("\nVISHGUARD AUDIO LOADER FAILED")