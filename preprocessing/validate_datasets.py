from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


FILES = {
    "ASVspoof 2019": METADATA_DIR / "asvspoof2019_la_metadata.csv",
    "Short windows": METADATA_DIR / "short_window_metadata.csv",
    "Phone conditions": METADATA_DIR / "phone_conditions_metadata.csv",
    "Banking scenarios": (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "banking_vishing"
        / "scripts"
        / "vishing_scenarios.csv"
    ),
}


def check_file(name, path):

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    if not path.exists():
        print("MISSING:")
        print(path)
        return

    print("File:", path)

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    if "label" in df.columns:
        print("\nLabel distribution:")
        print(df["label"].value_counts())

    if "split" in df.columns:
        print("\nSplit distribution:")
        print(df["split"].value_counts())

    if "window_sec" in df.columns:
        print("\nWindow distribution:")
        print(df["window_sec"].value_counts().sort_index())

    if "condition" in df.columns:
        print("\nCondition distribution:")
        print(df["condition"].value_counts())

    if "scenario_id" in df.columns:
        print("\nScenario distribution:")
        print(df["scenario_id"].nunique(), "unique scenarios")


print("=" * 65)
print("AI VOICE VISHING DATASET VALIDATION")
print("=" * 65)

for name, path in FILES.items():
    check_file(name, path)

print("\n" + "=" * 65)
print("VALIDATION COMPLETE")
print("=" * 65)