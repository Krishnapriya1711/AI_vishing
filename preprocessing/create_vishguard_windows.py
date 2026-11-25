from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vishguard_final_metadata.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vishguard_short_window_metadata.csv"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SEC = 1.0
HOP_SEC = 0.5


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("CREATING VISHGUARD SHORT-WINDOW METADATA")
print("=" * 70)

print(f"Input calls : {len(df)}")
print(f"Window size : {WINDOW_SEC} sec")
print(f"Hop size    : {HOP_SEC} sec")


# ============================================================
# CREATE WINDOWS
# ============================================================

records = []

for _, row in df.iterrows():

    duration = float(row["duration_sec"])

    call_id = row["call_id"]
    file = row["file"]
    label = row["label"]

    # Last valid starting point
    max_start = duration - WINDOW_SEC

    if max_start < 0:
        continue

    starts = np.arange(
        0,
        max_start + 1e-9,
        HOP_SEC
    )

    for start in starts:

        start = round(float(start), 3)

        end = round(
            start + WINDOW_SEC,
            3
        )

        records.append({
            "call_id": call_id,
            "file": file,
            "label": label,
            "window_sec": WINDOW_SEC,
            "start_sec": start,
            "end_sec": end,
            "sample_rate_original": row["sample_rate"],
            "source_dataset": "VISHGUARD",
        })


# ============================================================
# DATAFRAME
# ============================================================

windows = pd.DataFrame(records)

windows.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("WINDOW SUMMARY")
print("=" * 70)

print(f"Total windows: {len(windows)}")

print()
print("Windows by label:")
print(
    windows["label"].value_counts()
)

print()
print("Windows per call:")
print(
    windows.groupby("call_id")
    .size()
    .describe()
)

print()
print("Minimum windows per call:",
      windows.groupby("call_id").size().min())

print("Maximum windows per call:",
      windows.groupby("call_id").size().max())


# ============================================================
# CHECKS
# ============================================================

print()
print("=" * 70)
print("CHECKS")
print("=" * 70)

checks_passed = True


if windows["call_id"].nunique() == 40:
    print("PASS: All 40 calls represented")
else:
    print("FAIL: Not all calls represented")
    checks_passed = False


if (windows["window_sec"] == 1.0).all():
    print("PASS: All windows are 1.0 second")
else:
    print("FAIL: Incorrect window duration")
    checks_passed = False


if (windows["end_sec"] - windows["start_sec"]).round(3).eq(1.0).all():
    print("PASS: All window boundaries are valid")
else:
    print("FAIL: Invalid window boundaries")
    checks_passed = False


if (
    windows["label"].isin(
        ["spoof", "bonafide"]
    ).all()
):
    print("PASS: Labels are valid")
else:
    print("FAIL: Invalid labels")
    checks_passed = False


if checks_passed:
    print()
    print("VISHGUARD SHORT-WINDOW METADATA PASSED")
else:
    print()
    print("VISHGUARD SHORT-WINDOW METADATA NEEDS ATTENTION")


print()
print("Saved to:")
print(OUTPUT_CSV)

print("=" * 70)