from pathlib import Path
import pandas as pd
import soundfile as sf


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banking_vishing"
)

SCRIPT_FILE = (
    AUDIO_ROOT
    / "scripts"
    / "vishing_scenarios.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "banking_vishing_metadata.csv"
)


# ============================================================
# Configuration
# ============================================================

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3"
}


# ============================================================
# Check directories
# ============================================================

BONAFIDE_DIR = AUDIO_ROOT / "bonafide"
SPOOF_DIR = AUDIO_ROOT / "spoof"

BONAFIDE_DIR.mkdir(parents=True, exist_ok=True)
SPOOF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load scenario information
# ============================================================

if SCRIPT_FILE.exists():

    scripts = pd.read_csv(SCRIPT_FILE)

    print(f"Loaded {len(scripts)} banking scenario utterances.")

else:

    print("WARNING: vishing_scenarios.csv not found.")

    scripts = pd.DataFrame()


# ============================================================
# Build scenario lookup
# ============================================================

scenario_lookup = {}

if not scripts.empty:

    for _, row in scripts.iterrows():

        scenario_id = str(row["scenario_id"])

        scenario_lookup[scenario_id] = {
            "scenario": row["scenario"],
            "category": row["category"]
        }


# ============================================================
# Scan audio
# ============================================================

records = []


def process_directory(directory, label):

    files = []

    for extension in AUDIO_EXTENSIONS:

        files.extend(directory.glob(f"*{extension}"))

    print(
        f"\n{label.upper()} audio files found: {len(files)}"
    )

    for audio_file in sorted(files):

        try:

            info = sf.info(str(audio_file))

            sample_rate = info.samplerate
            frames = info.frames
            duration = info.duration
            channels = info.channels

        except Exception as e:

            print(
                f"WARNING: Could not read {audio_file.name}: {e}"
            )

            continue


        # ----------------------------------------------------
        # Infer metadata from filename
        #
        # Expected examples:
        #
        # human_H01_BV01_001.wav
        # spoof_TTS01_BV01_001.wav
        # ----------------------------------------------------

        parts = audio_file.stem.split("_")

        speaker_id = "unknown"
        scenario_id = "unknown"

        if len(parts) >= 3:

            speaker_id = parts[1]
            scenario_id = parts[2]


        scenario = scenario_lookup.get(
            scenario_id,
            {}
        )

        records.append({

            "file": audio_file.name,

            "label": label,

            "speaker_id": speaker_id,

            "scenario_id": scenario_id,

            "scenario": scenario.get(
                "scenario",
                "unknown"
            ),

            "category": scenario.get(
                "category",
                "unknown"
            ),

            "language": "unknown",

            "sample_rate": sample_rate,

            "channels": channels,

            "duration": duration,

            "source": "banking_vishing"

        })


# ============================================================
# Process both classes
# ============================================================

process_directory(
    BONAFIDE_DIR,
    "bonafide"
)

process_directory(
    SPOOF_DIR,
    "spoof"
)


# ============================================================
# Create metadata
# ============================================================

metadata = pd.DataFrame(records)


if metadata.empty:

    print("\nNo audio files found yet.")
    print("Metadata file will not be created.")

else:

    metadata = metadata.sort_values(
        ["scenario_id", "label", "file"]
    )

    metadata.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("BANKING VISHING METADATA")
    print("=" * 60)

    print(
        f"\nTotal audio files: {len(metadata)}"
    )

    print(
        "\nLabel distribution:"
    )

    print(
        metadata["label"].value_counts()
    )

    print(
        "\nSample-rate distribution:"
    )

    print(
        metadata["sample_rate"].value_counts()
    )

    print(
        "\nSaved to:"
    )

    print(OUTPUT_FILE)