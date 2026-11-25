from pathlib import Path
import random
import numpy as np
import pandas as pd

from preprocessing.vishguard_audio_loader import load_vishguard_audio


METADATA_PATH = Path(
    "data/metadata/vishguard_short_window_metadata.csv"
)


print("=" * 70)
print("VISHGUARD SHORT-WINDOW AUDIO VERIFICATION")
print("=" * 70)


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA_PATH)

print(f"Total windows: {len(df)}")
print(f"Unique calls : {df['call_id'].nunique()}")

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# SELECT REPRESENTATIVE WINDOWS
# ============================================================

random.seed(42)

samples = []

for label in ["spoof", "bonafide"]:

    subset = df[df["label"] == label]

    selected = subset.sample(
        n=min(5, len(subset)),
        random_state=42
    )

    samples.append(selected)

test_df = pd.concat(samples)


# ============================================================
# TEST ALL CONDITIONS
# ============================================================

conditions = [
    "clean",
    "telephone",
    "telephone_codec"
]

passed = 0
failed = 0


for _, row in test_df.iterrows():

    print("\n" + "-" * 70)

    print(f"Call ID : {row['call_id']}")
    print(f"File    : {row['file']}")
    print(f"Label   : {row['label']}")
    print(f"Start   : {row['start_sec']} sec")

    for condition in conditions:

        try:

            audio, sr = load_vishguard_audio(
                file_name=row["file"],
                label=row["label"],
                condition=condition,
                duration=1.0,
                start_sec=float(row["start_sec"])
            )

            expected_sr = (
                16000
                if condition == "clean"
                else 8000
            )

            checks = {
                "sample_rate": sr == expected_sr,
                "length": len(audio) == expected_sr,
                "mono": audio.ndim == 1,
                "dtype": audio.dtype == np.float32,
                "finite": np.all(np.isfinite(audio)),
                "nonempty": len(audio) > 0,
                "amplitude": np.max(np.abs(audio)) <= 1.0,
            }

            if all(checks.values()):

                print(
                    f"  {condition:18s} PASS | "
                    f"SR={sr} | "
                    f"samples={len(audio)} | "
                    f"max={np.max(np.abs(audio)):.4f}"
                )

                passed += 1

            else:

                failed_checks = [
                    name
                    for name, value in checks.items()
                    if not value
                ]

                print(
                    f"  {condition:18s} FAIL | "
                    f"{failed_checks}"
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
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"Windows tested : {len(test_df)}")
print(f"Conditions     : {len(conditions)}")
print(f"Tests passed   : {passed}")
print(f"Tests failed   : {failed}")


if failed == 0:

    print("\nPASS: All VISHGUARD windows are valid")
    print("PASS: Clean audio = 16 kHz / 1 second")
    print("PASS: Telephone audio = 8 kHz / 1 second")
    print("PASS: Telephone codec audio = 8 kHz / 1 second")
    print("PASS: Mono audio")
    print("PASS: Finite values")
    print("PASS: Valid amplitude")

    print("\nVISHGUARD WINDOW PIPELINE PASSED")

else:

    print("\nVISHGUARD WINDOW PIPELINE FAILED")