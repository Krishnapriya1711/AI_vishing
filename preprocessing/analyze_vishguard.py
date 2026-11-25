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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vishguard_english_candidates.csv"
)

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("=" * 70)
print("VISHGUARD DATASET ANALYSIS")
print("=" * 70)

print("\nTotal calls:", len(df))

print("\nLanguages:")
print(df["language"].value_counts().to_string())

print("\nCall types:")
print(df["type"].value_counts().to_string())

print("\nLanguage × Type:")
print(
    pd.crosstab(
        df["language"],
        df["type"]
    ).to_string()
)

# ------------------------------------------------------------
# English calls
# ------------------------------------------------------------

english = df[df["language"] == "en"].copy()

print("\n" + "=" * 70)
print("ENGLISH CALLS")
print("=" * 70)

print("\nTotal English calls:", len(english))

print("\nEnglish by type:")
print(english["type"].value_counts().to_string())

# ------------------------------------------------------------
# Banking-related keyword search
# ------------------------------------------------------------

banking_keywords = [
    "bank",
    "account",
    "card",
    "credit card",
    "debit card",
    "transfer",
    "payment",
    "transaction",
    "otp",
    "pin",
    "password",
    "wallet",
    "fund",
    "money",
    "refund",
    "verify",
    "verification",
    "cash",
    "atm",
    "loan",
    "finance",
    "financial"
]

pattern = "|".join(banking_keywords)

english["banking_relevant"] = (
    english["script"]
    .fillna("")
    .str.lower()
    .str.contains(pattern, regex=True)
)

banking = english[english["banking_relevant"]].copy()

print("\n" + "=" * 70)
print("ENGLISH BANKING-RELEVANT CALLS")
print("=" * 70)

print("\nTotal:", len(banking))

print("\nBy type:")
print(banking["type"].value_counts().to_string())

# ------------------------------------------------------------
# Save candidates
# ------------------------------------------------------------

columns = [
    "id",
    "type",
    "language",
    "script",
    "duration",
    "emotion",
    "tone",
    "persuasion_patterns",
    "behavior",
    "metadata",
    "audio_path"
]

banking[columns].to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved candidate metadata to:")
print(OUTPUT_FILE)

# ------------------------------------------------------------
# Show examples
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE ENGLISH BANKING CALLS")
print("=" * 70)

for _, row in banking.head(10).iterrows():

    print("\nID:", row["id"])
    print("Type:", row["type"])
    print("Duration:", row["duration"])
    print("Audio:", row["audio_path"])
    print("Script:", row["script"][:400])

print("\nAnalysis complete.")