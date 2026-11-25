from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTED_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vishguard_selected_40.csv"
)

VALIDATION_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vishguard_audio_validation.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vishguard_final_metadata.csv"
)


# ============================================================
# LOAD
# ============================================================

selected = pd.read_csv(SELECTED_CSV)
validation = pd.read_csv(VALIDATION_CSV)

print("=" * 70)
print("CREATING FINAL VISHGUARD METADATA")
print("=" * 70)

print(f"Selected records   : {len(selected)}")
print(f"Validation records : {len(validation)}")

print()
print("Selected columns:")
print(list(selected.columns))


# ============================================================
# CREATE AUDIO FILENAME
# ============================================================

# VISHGUARD ID:
# lot11_en_fraud_1040
#
# Actual extracted audio:
# lot11_en_fraud_1040.wav

if "file" not in selected.columns:

    selected["file"] = (
        selected["id"].astype(str) + ".wav"
    )


# ============================================================
# MERGE AUDIO VALIDATION
# ============================================================

validation_subset = validation[
    [
        "file",
        "sample_rate",
        "channels",
        "duration_sec",
        "frames",
        "subtype",
        "status",
    ]
].copy()


df = selected.merge(
    validation_subset,
    on="file",
    how="left",
    validate="one_to_one"
)


# ============================================================
# STANDARD PROJECT LABEL
# ============================================================

df["label"] = df["type"].map({
    "fraudulent": "spoof",
    "legitimate": "bonafide"
})


# ============================================================
# CALL ID
# ============================================================

df["call_id"] = df["id"]


# ============================================================
# DATASET INFORMATION
# ============================================================

df["source_dataset"] = "VISHGUARD"

df["audio_condition"] = "original"

df["audio_format"] = "WAV"

df["language"] = (
    df["language"]
    .astype(str)
    .str.lower()
)


# ============================================================
# REORDER COLUMNS
# ============================================================

columns = [
    "call_id",
    "id",
    "file",

    "label",
    "type",

    "language",
    "lot",

    "scenario_id",
    "scenario",
    "category",

    "script",

    "duration",
    "duration_sec",

    "sample_rate",
    "channels",
    "frames",
    "subtype",
    "status",

    "emotion",
    "tone",
    "persuasion_patterns",
    "behavior",

    "audio_path",

    "source_dataset",
    "audio_condition",
    "audio_format",
]

# Keep only columns that actually exist.
columns = [
    col
    for col in columns
    if col in df.columns
]

df = df[columns]


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL VISHGUARD METADATA")
print("=" * 70)

print(f"Total records: {len(df)}")

print()
print("Labels:")
print(df["label"].value_counts())

print()
print("Original types:")
print(df["type"].value_counts())

print()
print("Languages:")
print(df["language"].value_counts())

print()
print("Audio conditions:")
print(df["audio_condition"].value_counts())

print()
print("Validation status:")
print(df["status"].value_counts())

print()
print("Sample rates:")
print(df["sample_rate"].value_counts())

print()
print("Channels:")
print(df["channels"].value_counts())


# ============================================================
# CHECKS
# ============================================================

print()
print("=" * 70)
print("CHECKS")
print("=" * 70)

checks_passed = True


# Total
if len(df) == 40:
    print("PASS: 40 records")
else:
    print(f"FAIL: Expected 40, found {len(df)}")
    checks_passed = False


# Unique calls
if df["call_id"].nunique() == 40:
    print("PASS: 40 unique call IDs")
else:
    print(
        f"FAIL: Expected 40 unique call IDs, "
        f"found {df['call_id'].nunique()}"
    )
    checks_passed = False


# Spoof
spoof_count = (
    df["label"] == "spoof"
).sum()

if spoof_count == 20:
    print("PASS: 20 spoof")
else:
    print(
        f"FAIL: Expected 20 spoof, found {spoof_count}"
    )
    checks_passed = False


# Bonafide
bonafide_count = (
    df["label"] == "bonafide"
).sum()

if bonafide_count == 20:
    print("PASS: 20 bonafide")
else:
    print(
        f"FAIL: Expected 20 bonafide, found {bonafide_count}"
    )
    checks_passed = False


# English
if (df["language"] == "en").all():
    print("PASS: All calls are English")
else:
    print("FAIL: Non-English calls found")
    checks_passed = False


# Audio validation
if (df["status"] == "OK").all():
    print("PASS: All audio files validated")
else:
    print("FAIL: Audio validation errors found")
    checks_passed = False


# Filename matching
missing_files = df["duration_sec"].isna().sum()

if missing_files == 0:
    print("PASS: All selected audio files matched")
else:
    print(
        f"FAIL: {missing_files} audio files "
        f"could not be matched"
    )
    checks_passed = False


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)

if checks_passed:

    print("VISHGUARD FINAL METADATA PASSED")

else:

    print("VISHGUARD FINAL METADATA NEEDS ATTENTION")

print("=" * 70)

print()
print("Saved to:")
print(OUTPUT_CSV)