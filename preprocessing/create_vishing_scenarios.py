from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banking_vishing"
    / "scripts"
    / "banking_call_scripts.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banking_vishing"
    / "scripts"
)

OUTPUT_FILE = OUTPUT_DIR / "vishing_scenarios.csv"


# --------------------------------------------------
# Scenario definitions
# --------------------------------------------------

SCENARIOS = {
    "BV01": ("card_payment_not_recognised", "Suspicious card transaction"),
    "BV02": ("cash_withdrawal_not_recognised", "Suspicious cash withdrawal"),
    "BV03": ("compromised_card", "Compromised card"),
    "BV04": ("failed_transfer", "Failed bank transfer"),
    "BV05": ("pending_transfer", "Pending bank transfer"),
    "BV06": ("transfer_not_received_by_recipient", "Transfer not received"),
    "BV07": ("verify_my_identity", "Identity verification"),
    "BV08": ("unable_to_verify_identity", "Identity verification problem"),
    "BV09": ("verify_source_of_funds", "Source of funds verification"),
    "BV10": ("top_up_failed", "Failed account top-up"),
    "BV11": ("direct_debit_payment_not_recognised", "Unknown direct debit"),
    "BV12": ("extra_charge_on_statement", "Unexpected account charge"),
    "BV13": ("card_not_working", "Card not working"),
    "BV14": ("lost_or_stolen_card", "Lost or stolen card"),
    "BV15": ("pin_blocked", "PIN blocked"),
    "BV16": ("Refund_not_showing_up", "Refund missing"),
    "BV17": ("request_refund", "Refund request"),
    "BV18": ("beneficiary_not_allowed", "Beneficiary problem"),
}


# --------------------------------------------------
# Load BANKING77-derived scripts
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Loaded:", len(df), "BANKING77 utterances")


# --------------------------------------------------
# Create one representative script per scenario
# --------------------------------------------------

records = []

for scenario_id, (category, description) in SCENARIOS.items():

    subset = df[df["category"] == category]

    if subset.empty:
        print(f"WARNING: No data for {category}")
        continue

    # Take up to 5 examples from each category.
    examples = subset.head(5)

    for _, row in examples.iterrows():

        records.append({
            "scenario_id": scenario_id,
            "scenario": description,
            "category": category,
            "banking_text": row["text"],
            "source": "BANKING77"
        })


result = pd.DataFrame(records)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nCreated:", len(result), "scenario utterances")
print("Saved to:")
print(OUTPUT_FILE)

print("\nScenario distribution:")
print(result["scenario"].value_counts())

print("\nPreview:")
print(result.head(20).to_string(index=False))