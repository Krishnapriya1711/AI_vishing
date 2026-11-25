from pathlib import Path
import soundfile as sf
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banking_vishing"
    / "audio"
)

SPOOF_DIR = AUDIO_ROOT / "spoof"
BONAFIDE_DIR = AUDIO_ROOT / "bonafide"

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vishguard_audio_validation.csv"
)


# ============================================================
# VALIDATE
# ============================================================

records = []

print("=" * 70)
print("VISHGUARD AUDIO VALIDATION")
print("=" * 70)


for label, folder in [
    ("spoof", SPOOF_DIR),
    ("bonafide", BONAFIDE_DIR),
]:

    files = sorted(folder.glob("*.wav"))

    print()
    print(f"{label.upper()}: {len(files)} files")

    for audio_file in files:

        try:

            info = sf.info(audio_file)

            duration = info.duration
            sample_rate = info.samplerate
            channels = info.channels
            frames = info.frames
            subtype = info.subtype

            status = "OK"

        except Exception as e:

            duration = None
            sample_rate = None
            channels = None
            frames = None
            subtype = None
            status = f"ERROR: {e}"

        records.append({
            "file": audio_file.name,
            "label": label,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_sec": duration,
            "frames": frames,
            "subtype": subtype,
            "status": status,
        })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total files : {len(df)}")

print()
print("Class distribution:")
print(df["label"].value_counts())

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
# DURATION
# ============================================================

valid_df = df[df["status"] == "OK"].copy()

if not valid_df.empty:

    print()
    print("Duration statistics:")
    print(
        valid_df["duration_sec"].describe()
    )

    print()
    print("Duration by class:")

    print(
        valid_df
        .groupby("label")["duration_sec"]
        .agg(
            ["count", "mean", "min", "max"]
        )
    )


# ============================================================
# CHECK EXPECTATIONS
# ============================================================

print()
print("=" * 70)
print("CHECKS")
print("=" * 70)

expected_total = 40
expected_spoof = 20
expected_bonafide = 20

if len(df) == expected_total:
    print("PASS: 40 audio files found")
else:
    print(
        f"FAIL: Expected 40 files, found {len(df)}"
    )


spoof_count = (df["label"] == "spoof").sum()
bonafide_count = (df["label"] == "bonafide").sum()

if spoof_count == expected_spoof:
    print("PASS: 20 spoof files")
else:
    print(
        f"FAIL: Expected 20 spoof, found {spoof_count}"
    )

if bonafide_count == expected_bonafide:
    print("PASS: 20 bonafide files")
else:
    print(
        f"FAIL: Expected 20 bonafide, found {bonafide_count}"
    )


errors = (df["status"] != "OK").sum()

if errors == 0:
    print("PASS: No corrupted/unreadable WAV files")
else:
    print(
        f"FAIL: {errors} files could not be read"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)

if (
    len(df) == 40
    and spoof_count == 20
    and bonafide_count == 20
    and errors == 0
):
    print("VISHGUARD AUDIO VALIDATION PASSED")
else:
    print("VISHGUARD AUDIO VALIDATION NEEDS ATTENTION")

print("=" * 70)

print()
print(f"Validation metadata saved to:")
print(OUTPUT_CSV)