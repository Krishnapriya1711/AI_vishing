from pathlib import Path
import pandas as pd
import librosa
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

METADATA_FILE = Path("data/metadata/asvspoof2019_la_metadata.csv")

OUTPUT_FILE = Path("data/metadata/asvspoof2019_la_audio_stats.csv")


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA_FILE)

print("=" * 60)
print("ASVspoof 2019 LA Audio Analysis")
print("=" * 60)

print("\nTotal files:", len(df))


# ============================================================
# ANALYZE AUDIO
# ============================================================

durations = []
sample_rates = []
channels = []

for path in tqdm(df["file_path"], desc="Analyzing audio"):

    path = Path(path)

    try:

        # Load audio without resampling
        y, sr = librosa.load(
            path,
            sr=None,
            mono=False
        )

        # Duration
        if y.ndim == 1:
            duration = len(y) / sr
            channel_count = 1
        else:
            duration = y.shape[-1] / sr
            channel_count = y.shape[0]

        durations.append(duration)
        sample_rates.append(sr)
        channels.append(channel_count)

    except Exception as e:

        print(f"\nError reading: {path}")
        print(e)

        durations.append(None)
        sample_rates.append(None)
        channels.append(None)


# ============================================================
# ADD RESULTS
# ============================================================

df["duration_sec"] = durations
df["sample_rate"] = sample_rates
df["channels"] = channels


# ============================================================
# SAVE
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("AUDIO ANALYSIS COMPLETE")
print("=" * 60)

print("\nSample rates:")
print(df["sample_rate"].value_counts())

print("\nChannels:")
print(df["channels"].value_counts())

print("\nDuration statistics:")
print(df["duration_sec"].describe())

print("\nDuration by split:")
print(
    df.groupby("split")["duration_sec"]
      .describe()
)

print("\nDuration by label:")
print(
    df.groupby("label")["duration_sec"]
      .describe()
)

print("\nSaved to:")
print(OUTPUT_FILE)