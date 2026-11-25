from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

JSON_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vishguard"
    / "annotations_with_audio.json"
)

print("=" * 70)
print("VISHGUARD ANNOTATION INSPECTION")
print("=" * 70)

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nTop-level type:", type(data).__name__)

if isinstance(data, dict):
    print("\nTop-level keys:")
    for key in data.keys():
        print(" -", key)

    # Try to identify the actual annotation list
    for key, value in data.items():
        if isinstance(value, list):
            print(f"\nPossible annotation list: '{key}'")
            print("Number of entries:", len(value))

            if len(value) > 0 and isinstance(value[0], dict):
                print("\nFirst entry:")
                print(json.dumps(value[0], indent=2, ensure_ascii=False)[:5000])

elif isinstance(data, list):
    print("\nNumber of entries:", len(data))

    if len(data) > 0:
        print("\nFirst entry:")
        print(json.dumps(data[0], indent=2, ensure_ascii=False)[:5000])

print("\nInspection complete.")