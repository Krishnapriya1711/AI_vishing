from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/metadata/asvspoof2019_la_audio_stats.csv")
OUTPUT_FILE = Path("data/metadata/short_window_metadata.csv")

# Short-window durations for our experiments
WINDOWS = [3.0, 2.0, 1.5, 1.0, 0.75, 0.5]

df = pd.read_csv(INPUT_FILE)

rows = []

print("=" * 60)
print("CREATING SHORT-WINDOW METADATA")
print("=" * 60)

for _, row in df.iterrows():

    duration = row["duration_sec"]

    for window in WINDOWS:

        # Only create an entry if the original audio
        # is long enough for this window
        if duration >= window:

            rows.append({
                "file": row["file"],
                "file_path": row["file_path"],
                "split": row["split"],
                "speaker_id": row["speaker_id"],
                "system_id": row["system_id"],
                "label": row["label"],
                "window_sec": window,
                "start_sec": 0.0
            })

short_df = pd.DataFrame(rows)

short_df.to_csv(OUTPUT_FILE, index=False)

print("\nTotal original audio files:", len(df))
print("Total short-window entries:", len(short_df))

print("\nEntries by window:")
print(
    short_df["window_sec"]
    .value_counts()
    .sort_index(ascending=False)
)

print("\nEntries by split:")
print(
    short_df.groupby(["split", "window_sec"])
    .size()
)

print("\nEntries by label:")
print(
    short_df.groupby(["label", "window_sec"])
    .size()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)