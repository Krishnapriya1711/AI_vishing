from pathlib import Path

# Change this only if your extracted folder has a different name
DATASET_DIR = Path("data/raw/LA/LA/ASVspoof2019_LA_train")

print("=" * 60)
print("ASVspoof Dataset Inspection")
print("=" * 60)

if not DATASET_DIR.exists():
    print("\nERROR: Dataset folder not found!")
    print("Expected location:")
    print(DATASET_DIR.resolve())
    exit()

print("\nDataset location:")
print(DATASET_DIR.resolve())

# ---------------------------------------------------------
# Count files
# ---------------------------------------------------------

all_files = list(DATASET_DIR.rglob("*"))

audio_files = [
    f for f in all_files
    if f.suffix.lower() in [".flac", ".wav", ".mp3"]
]

text_files = [
    f for f in all_files
    if f.suffix.lower() in [".txt", ".csv"]
]

print("\nTotal files:", len(all_files))
print("Audio files:", len(audio_files))
print("Text/CSV files:", len(text_files))

# ---------------------------------------------------------
# Show folders
# ---------------------------------------------------------

print("\nFolders found:")
folders = [f for f in all_files if f.is_dir()]

for folder in folders[:30]:
    print("  ", folder.relative_to(DATASET_DIR))

# ---------------------------------------------------------
# Show audio examples
# ---------------------------------------------------------

print("\nFirst 10 audio files:")

for file in audio_files[:10]:
    print("  ", file.relative_to(DATASET_DIR))

# ---------------------------------------------------------
# Show protocol/text files
# ---------------------------------------------------------

print("\nProtocol/text files:")

for file in text_files[:30]:
    print("  ", file.relative_to(DATASET_DIR))

print("\n" + "=" * 60)
print("Inspection complete")
print("=" * 60)