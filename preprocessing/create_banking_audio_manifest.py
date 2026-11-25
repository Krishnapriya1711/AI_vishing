from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENARIO_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banking_vishing"
    / "scripts"
    / "vishing_scenarios.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "banking_vishing_metadata.csv"
)

# Load the 18 banking scenarios
df = pd.read_csv(SCENARIO_FILE)

# Keep one representative utterance per scenario for the
# initial controlled banking audio validation set.
scenario_df = (
    df.groupby(
        ["scenario_id", "scenario", "category"],
        as_index=False
    )
    .first()
)

records = []

for _, row in scenario_df.iterrows():

    scenario_id = row["scenario_id"]
    scenario = row["scenario"]
    category = row["category"]

    # One bonafide and one spoof file per scenario
    records.append({
        "file": f"{scenario_id}_bonafide.wav",
        "label": "bonafide",
        "speaker_id": "BONAFIDE_S01",
        "scenario_id": scenario_id,
        "scenario": scenario,
        "category": category,
        "language": "English",
        "sample_rate": 16000,
        "channels": 1,
        "duration": "",
        "source": ""
    })

    records.append({
        "file": f"{scenario_id}_spoof.wav",
        "label": "spoof",
        "speaker_id": "SPOOF_V01",
        "scenario_id": scenario_id,
        "scenario": scenario,
        "category": category,
        "language": "English",
        "sample_rate": 16000,
        "channels": 1,
        "duration": "",
        "source": ""
    })

manifest = pd.DataFrame(records)

# Save
manifest.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("BANKING AUDIO MANIFEST CREATED")
print("=" * 60)

print("\nTotal files planned:", len(manifest))

print("\nFiles by label:")
print(manifest["label"].value_counts())

print("\nScenarios:")
print(manifest["scenario_id"].nunique())

print("\nOutput:")
print(OUTPUT_FILE)

print("\nPreview:")
print(manifest.head(6).to_string(index=False))