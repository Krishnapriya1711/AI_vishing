from pathlib import Path
import numpy as np

from audio_loader import load_audio


# ============================================================
# TEST SETTINGS
# ============================================================

FILE_NAME = "LA_T_1001074.flac"
SPLIT = "train"

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
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("TESTING WINDOW × CONDITION PIPELINE")
    print("=" * 60)

    print(f"\nFile: {FILE_NAME}")
    print(f"Split: {SPLIT}")

    total_tests = 0
    successful_tests = 0

    for condition in CONDITIONS:

        print("\n" + "-" * 60)
        print(f"CONDITION: {condition}")
        print("-" * 60)

        for duration in DURATIONS:

            total_tests += 1

            try:

                audio, sr = load_audio(
                    file_name=FILE_NAME,
                    split=SPLIT,
                    condition=condition,
                    duration=duration,
                    start_sec=0.0
                )

                actual_duration = (
                    len(audio) / sr
                )

                print(
                    f"{duration:>4.2f} sec"
                    f" -> "
                    f"{sr:>5} Hz"
                    f" | "
                    f"{len(audio):>6} samples"
                    f" | "
                    f"{actual_duration:.3f} sec"
                )

                # Basic validation
                expected_samples = int(
                    duration * sr
                )

                if len(audio) != expected_samples:

                    print(
                        "     WARNING: "
                        "unexpected sample count"
                    )

                if not np.isfinite(audio).all():

                    print(
                        "     ERROR: "
                        "NaN/Inf detected"
                    )

                else:

                    successful_tests += 1

            except Exception as e:

                print(
                    f"{duration:>4.2f} sec"
                    f" -> ERROR: {e}"
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    print(
        f"Successful: "
        f"{successful_tests}/{total_tests}"
    )

    if successful_tests == total_tests:

        print(
            "\nALL TESTS PASSED."
        )

        print(
            "The preprocessing pipeline is ready "
            "for feature extraction."
        )

    else:

        print(
            "\nSome tests failed."
        )


if __name__ == "__main__":
    main()