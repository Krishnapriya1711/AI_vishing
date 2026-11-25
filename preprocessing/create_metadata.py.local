from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("data/raw/LA/LA")

PROTOCOL_DIR = BASE_DIR / "ASVspoof2019_LA_cm_protocols"

OUTPUT_DIR = Path("data/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET SPLITS
# ============================================================

splits = {
    "train": {
        "audio_dir": BASE_DIR / "ASVspoof2019_LA_train" / "flac",
        "protocol": PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt",
    },

    "dev": {
        "audio_dir": BASE_DIR / "ASVspoof2019_LA_dev" / "flac",
        "protocol": PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt",
    },

    "eval": {
        "audio_dir": BASE_DIR / "ASVspoof2019_LA_eval" / "flac",
        "protocol": PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt",
    },
}


# ============================================================
# READ PROTOCOL
# ============================================================

all_rows = []

for split, info in splits.items():

    protocol_path = info["protocol"]
    audio_dir = info["audio_dir"]

    print("\n" + "=" * 60)
    print(f"Processing: {split}")
    print("=" * 60)

    if not protocol_path.exists():
        print("Protocol not found:")
        print(protocol_path)
        continue

    if not audio_dir.exists():
        print("Audio folder not found:")
        print(audio_dir)
        continue

    with open(protocol_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            speaker_id = parts[0]
            utterance_id = parts[1]
            system_id = parts[3]
            label = parts[4]

            audio_path = audio_dir / f"{utterance_id}.flac"

            all_rows.append({
                "file": utterance_id,
                "file_path": str(audio_path),
                "split": split,
                "speaker_id": speaker_id,
                "system_id": system_id,
                "label": label
            })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(all_rows)


# ============================================================
# SAVE
# ============================================================

output_file = OUTPUT_DIR / "asvspoof2019_la_metadata.csv"

df.to_csv(output_file, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("METADATA CREATION COMPLETE")
print("=" * 60)

print("\nTotal entries:", len(df))

print("\nEntries by split:")
print(df["split"].value_counts())

print("\nLabels:")
print(df["label"].value_counts())

print("\nSplit + label:")
print(
    df.groupby(["split", "label"])
      .size()
)


print("\nSaved to:")
print(output_file)