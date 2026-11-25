from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "banking_text" / "train.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banking_vishing"
    / "scripts"
)

OUTPUT_FILE = OUTPUT_DIR / "banking_call_scripts.csv"


# --------------------------------------------------
# Relevant BANKING77 categories
# --------------------------------------------------
SELECTED_CATEGORIES = [
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "compromised_card",
    "beneficiary_not_allowed",
    "declined_transfer",
    "failed_transfer",
    "pending_transfer",
    "transfer_not_received_by_recipient",
    "verify_my_identity",
    "unable_to_verify_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "top_up_failed",
    "cash_withdrawal_charge",
    "card_payment_fee_charged",
    "extra_charge_on_statement",
    "direct_debit_payment_not_recognised",
    "Refund_not_showing_up",
    "request_refund",
    "card_not_working",
    "card_swallowed",
    "lost_or_stolen_card",
    "pin_blocked",
    "change_pin",
]


# --------------------------------------------------
# Load BANKING77
# --------------------------------------------------
print("=" * 60)
print("BANKING VISHING SCRIPT CREATION")
print("=" * 60)

print(f"\nReading:\n{INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Total BANKING77 training samples: {len(df)}")


# --------------------------------------------------
# Check categories
# --------------------------------------------------
available_categories = set(df["category"].unique())

missing_categories = [
    c for c in SELECTED_CATEGORIES
    if c not in available_categories
]

if missing_categories:
    print("\nWARNING: These categories were not found:")
    for category in missing_categories:
        print("  -", category)

    SELECTED_CATEGORIES = [
        c for c in SELECTED_CATEGORIES
        if c in available_categories
    ]


# --------------------------------------------------
# Filter relevant banking scenarios
# --------------------------------------------------
filtered = df[df["category"].isin(SELECTED_CATEGORIES)].copy()

print(
    f"\nSelected samples: {len(filtered)}"
)

print("\nSelected categories:")
print(
    filtered["category"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# Remove duplicate scripts
# --------------------------------------------------
filtered = filtered.drop_duplicates(subset=["text"])

filtered = filtered.reset_index(drop=True)


# --------------------------------------------------
# Add application metadata
# --------------------------------------------------
filtered["application"] = "banking_vishing"
filtered["source"] = "BANKING77"


# --------------------------------------------------
# Save
# --------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

filtered.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"\nFinal scripts: {len(filtered)}")
print(f"Saved to:\n{OUTPUT_FILE}")

print("\nFirst 10 scripts:")
print(
    filtered[
        ["text", "category"]
    ].head(10).to_string(index=False)
)