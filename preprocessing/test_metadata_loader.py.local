from pathlib import Path
import pandas as pd

from audio_loader import load_audio


# ============================================================
# SETTINGS
# ============================================================

METADATA_FILE = Path(
    "data/metadata/short_window_metadata.csv"
)

DURATIONS = [
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0
]

CONDITIONS = [
    "clean",
    "telephone",
    "telephone_codec"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TESTING METADATA → DYNAMIC AUDIO PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print("\nLoading metadata...")

    df = pd.read_csv(METADATA_FILE)

    print(f"Metadata entries: {len(df):,}")

    print("\nColumns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "file",
        "split",
        "label",
        "window_sec",
        "start_sec"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print("\nERROR: Missing columns:")
        print(missing)

        return

    # --------------------------------------------------------
    # Select one file that supports a 3-second window
    # --------------------------------------------------------

    candidates = df[
        df["window_sec"] >= 3.0
    ]

    if len(candidates) == 0:

        print(
            "\nERROR: No metadata entry "
            "supports a 3-second window."
        )

        return

    # Pick the first suitable row
    row = candidates.iloc[0]

    file_name = row["file"]
    split = row["split"]
    label = row["label"]

    print("\n" + "-" * 60)
    print("SELECTED METADATA")
    print("-" * 60)

    print(f"File       : {file_name}")
    print(f"Split      : {split}")
    print(f"Label      : {label}")
    print(f"Window     : {row['window_sec']} sec")
    print(f"Start      : {row['start_sec']} sec")

    # --------------------------------------------------------
    # Test every duration × condition
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TESTING AUDIO EXTRACTION")
    print("=" * 60)

    successful = 0
    total = len(DURATIONS) * len(CONDITIONS)

    for condition in CONDITIONS:

        print("\n" + "-" * 60)
        print(f"CONDITION: {condition}")
        print("-" * 60)

        for duration in DURATIONS:

            try:

                audio, sr = load_audio(
                    file_name=file_name,
                    split=split,
                    condition=condition,
                    duration=duration,
                    start_sec=0.0
                )

                actual_duration = len(audio) / sr

                print(
                    f"{duration:>4.2f}s"
                    f" -> "
                    f"{sr:>5} Hz"
                    f" | "
                    f"{len(audio):>6} samples"
                    f" | "
                    f"{actual_duration:.3f}s"
                    f" | "
                    f"label={label}"
                )

                # ------------------------------------------------
                # Basic validation
                # ------------------------------------------------

                expected_samples = int(
                    duration * sr
                )

                if len(audio) != expected_samples:

                    print(
                        "       WARNING: "
                        f"Expected {expected_samples} samples"
                    )

                if not pd.Series(audio).notna().all():

                    print(
                        "       ERROR: "
                        "NaN detected"
                    )

                else:

                    successful += 1

            except Exception as e:

                print(
                    f"{duration:>4.2f}s"
                    f" -> ERROR: {e}"
                )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print(
        f"Successful: "
        f"{successful}/{total}"
    )

    if successful == total:

        print(
            "\nALL TESTS PASSED."
        )

        print(
            "Metadata → dynamic audio pipeline "
            "is working correctly."
        )

    else:

        print(
            "\nSome tests failed."
        )

        print(
            "Check the errors above."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()