from pathlib import Path
import pandas as pd

FILE = Path("data/metadata/asvspoof2019_la_audio_stats.csv")

df = pd.read_csv(FILE)

windows = [3.0, 2.0, 1.5, 1.0, 0.75, 0.5]

print("=" * 60)
print("SHORT-WINDOW AVAILABILITY")
print("=" * 60)

for w in windows:
    count = (df["duration_sec"] >= w).sum()
    percentage = count / len(df) * 100

    print(
        f"{w:>4.2f} sec : "
        f"{count:>7} files "
        f"({percentage:>6.2f}%)"
    )

print("\nBy split:")

for split in ["train", "dev", "eval"]:
    subset = df[df["split"] == split]

    print(f"\n{split.upper()}")

    for w in windows:
        count = (subset["duration_sec"] >= w).sum()
        percentage = count / len(subset) * 100

        print(
            f"{w:>4.2f} sec : "
            f"{count:>7} "
            f"({percentage:>6.2f}%)"
        )