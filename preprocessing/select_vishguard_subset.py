from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vishguard_banking_candidates.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vishguard_selected_40.csv"
)

# ---------------------------------------------------------
# Load candidates
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# ---------------------------------------------------------
# Banking relevance scoring
# ---------------------------------------------------------

BANK_TERMS = [
    r"\bbank\b",
    r"\bbanking\b",
    r"\bbank account\b",
    r"\bchecking account\b",
    r"\bsavings account\b",
    r"\bonline banking\b",
    r"\bmobile banking\b",
    r"\bcredit card\b",
    r"\bdebit card\b",
    r"\baccount number\b",
    r"\brouting number\b",
    r"\bbank transfer\b",
    r"\bwire transfer\b",
    r"\bbank wire\b",
    r"\btransfer your funds\b",
    r"\btransfer funds\b",
    r"\baccount balance\b",
    r"\boverdraft\b",
    r"\bbank transaction\b",
    r"\bunauthorized transaction\b",
    r"\baccount transaction\b",
    r"\bdirect debit\b",
    r"\bbeneficiary\b",
]

BANK_PATTERN = re.compile(
    "|".join(BANK_TERMS),
    re.IGNORECASE
)

df["bank_score"] = df["script"].fillna("").apply(
    lambda x: len(BANK_PATTERN.findall(x))
)

# ---------------------------------------------------------
# Fraud-specific banking actions
# ---------------------------------------------------------

FRAUD_BANK_ACTIONS = [
    r"\bgive me your bank",
    r"\bprovide.*bank",
    r"\bconfirm.*bank",
    r"\bconfirm.*account",
    r"\bgive me.*account",
    r"\bprovide.*account",
    r"\bconfirm.*card",
    r"\bconfirm.*pin",
    r"\bconfirm.*password",
    r"\bsecurity code",
    r"\btransfer.*funds",
    r"\btransfer.*account",
    r"\bwire.*money",
    r"\bwire.*funds",
    r"\baccount.*hacked",
    r"\baccount.*locked",
    r"\bunauthorized.*transaction",
    r"\bfraudulent.*transaction",
]

FRAUD_ACTION_PATTERN = re.compile(
    "|".join(FRAUD_BANK_ACTIONS),
    re.IGNORECASE
)

df["fraud_action_score"] = df["script"].fillna("").apply(
    lambda x: len(FRAUD_ACTION_PATTERN.findall(x))
)

# ---------------------------------------------------------
# Exclude obvious non-banking fraud scenarios
# ---------------------------------------------------------

EXCLUDE_FRAUD = [
    r"\blottery\b",
    r"\bwon\b.*\bprize\b",
    r"\bplaystation\b",
    r"\bamazon gift card\b",
    r"\bnetflix gift card\b",
    r"\bsamsung smartphone\b",
    r"\bcruise\b",
    r"\bdating app\b",
    r"\bgranddaughter\b",
    r"\bstudent loan\b",
    r"\birs\b",
    r"\bdepartment of justice\b",
    r"\bfederal agent\b",
    r"\bgovernment grant\b",
    r"\bgovernment tax rebate\b",
    r"\butility\b",
    r"\bgift card\b",
    r"\bprepaid gift card\b",
    r"\bbitcoin\b",
]

EXCLUDE_PATTERN = re.compile(
    "|".join(EXCLUDE_FRAUD),
    re.IGNORECASE
)

df["excluded"] = df["script"].fillna("").apply(
    lambda x: bool(EXCLUDE_PATTERN.search(x))
)

# Must contain banking terminology
df = df[df["bank_score"] > 0].copy()

# Remove non-banking fraudulent calls
df = df[
    ~(
        (df["type"] == "fraudulent")
        & (df["excluded"])
    )
].copy()

# ---------------------------------------------------------
# Remove duplicate / near-identical scripts
# ---------------------------------------------------------

df["script_normalized"] = (
    df["script"]
    .fillna("")
    .str.lower()
    .str.replace(r"[^a-z0-9 ]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# ---------------------------------------------------------
# Automatic selection
# ---------------------------------------------------------

selected_parts = []

for label in ["fraudulent", "legitimate"]:

    subset = df[df["type"] == label].copy()

    subset = subset.sort_values(
        ["bank_score", "fraud_action_score", "duration"],
        ascending=[False, False, True]
    )

    subset = subset.drop_duplicates(
        subset=["script_normalized"]
    )

    selected = subset.head(20).copy()

    selected_parts.append(selected)

selected = pd.concat(
    selected_parts,
    ignore_index=True
)

# ---------------------------------------------------------
# MANUAL CORRECTIONS
# ---------------------------------------------------------

# Calls that should NOT be in the final banking-vishing subset
REMOVE_IDS = {
    "lot16_en_fraud_1514",   # Social Security scam
    "lot16_en_fraud_1545",   # Sweepstakes/prize scam
    "lot21_en_fraud_2020",   # Social Security scam
    "lot25_en_legit_2471",   # Gym payment call
}

# Banking-vishing replacements
REPLACEMENT_IDS = {
    "lot2_en_fraud_145b",
    "lot3_en_fraud_234",
    "lot3_en_fraud_264",
    "lot30_en_legit_2995",
}

# Remove unsuitable calls
selected = selected[
    ~selected["id"].isin(REMOVE_IDS)
].copy()

# Add replacements
replacements = df[
    df["id"].isin(REPLACEMENT_IDS)
].copy()

selected = pd.concat(
    [selected, replacements],
    ignore_index=True
)

# ---------------------------------------------------------
# Final safety check
# ---------------------------------------------------------

# Make sure no unwanted IDs remain
selected = selected[
    ~selected["id"].isin(REMOVE_IDS)
].copy()

# Remove accidental duplicate IDs
selected = selected.drop_duplicates(
    subset=["id"]
).copy()

# ---------------------------------------------------------
# Remove temporary scoring columns
# ---------------------------------------------------------

selected = selected.drop(
    columns=[
        "bank_score",
        "fraud_action_score",
        "excluded",
        "script_normalized",
    ],
    errors="ignore"
)

# ---------------------------------------------------------
# Final ordering
# ---------------------------------------------------------

selected = selected.sort_values(
    ["type", "id"]
).reset_index(drop=True)

# ---------------------------------------------------------
# Verify exactly 40 = 20 fraudulent + 20 legitimate
# ---------------------------------------------------------

fraud_count = (selected["type"] == "fraudulent").sum()
legit_count = (selected["type"] == "legitimate").sum()

if len(selected) != 40:
    raise ValueError(
        f"Expected 40 calls, but got {len(selected)}"
    )

if fraud_count != 20:
    raise ValueError(
        f"Expected 20 fraudulent calls, but got {fraud_count}"
    )

if legit_count != 20:
    raise ValueError(
        f"Expected 20 legitimate calls, but got {legit_count}"
    )

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

selected.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nVISHGUARD subset created successfully.")
print(f"Output: {OUTPUT_FILE}")
print(f"Total selected: {len(selected)}")

print("\nClass distribution:")
print(selected["type"].value_counts())

print("\nLanguage distribution:")
print(selected["language"].value_counts())

print("\nRemoved IDs:")
for x in sorted(REMOVE_IDS):
    print(f"  {x}")

print("\nAdded replacement IDs:")
for x in sorted(REPLACEMENT_IDS):
    print(f"  {x}")

print("\nFinal selected IDs:")
print(
    selected[
        ["id", "type", "language", "duration"]
    ].to_string(index=False)
)
