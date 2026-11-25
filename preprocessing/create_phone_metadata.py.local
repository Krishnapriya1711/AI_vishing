from pathlib import Path
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_METADATA = Path(
    "data/metadata/asvspoof2019_la_metadata.csv"
)

OUTPUT_METADATA = Path(
    "data/metadata/phone_conditions_metadata.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading original metadata...")

    df = pd.read_csv(INPUT_METADATA)

    print(f"Original files: {len(df):,}")

    print("\nColumns found:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "file",
        "label",
        "split",
        "speaker_id"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        print("\nERROR: Missing columns:")
        print(missing)

        return

    # --------------------------------------------------------
    # Create conditions efficiently
    # --------------------------------------------------------

    conditions = []

    for condition, sample_rate, codec in [
        ("clean", 16000, "none"),
        ("telephone", 8000, "none"),
        ("telephone_codec", 8000, "G711_A-law")
    ]:

        temp = df[
            required_columns
        ].copy()

        temp["condition"] = condition
        temp["sample_rate"] = sample_rate
        temp["codec"] = codec

        conditions.append(temp)

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    phone_df = pd.concat(
        conditions,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_METADATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    phone_df.to_csv(
        OUTPUT_METADATA,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n========================================")
    print("PHONE CONDITION METADATA CREATED")
    print("========================================")

    print(
        f"\nTotal entries: {len(phone_df):,}"
    )

    print("\nConditions:")

    print(
        phone_df["condition"].value_counts()
    )

    print("\nBy split:")

    print(
        pd.crosstab(
            phone_df["split"],
            phone_df["condition"]
        )
    )

    print("\nBy label:")

    print(
        pd.crosstab(
            phone_df["label"],
            phone_df["condition"]
        )
    )

    print(
        f"\nSaved to:\n{OUTPUT_METADATA}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()