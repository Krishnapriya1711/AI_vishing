from pathlib import Path
import pandas as pd
import requests
import zipfile
import shutil
import io


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTED_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "vishguard_selected_40.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "banking_vishing"
    / "audio"
)

SPOOF_DIR = OUTPUT_ROOT / "spoof"
BONAFIDE_DIR = OUTPUT_ROOT / "bonafide"


# ============================================================
# ZENODO
# ============================================================

ZIP_URL = (
    "https://zenodo.org/records/20429763/files/"
    "vishguard_dataset.zip?download=1"
)

ZIP_PREFIX = "vishguard_dataset/audio/"


# ============================================================
# REMOTE RANGE READER
# ============================================================

class RemoteZipReader(io.RawIOBase):

    def __init__(self, url):

        self.url = url
        self.position = 0

        response = requests.head(
            url,
            allow_redirects=True,
            timeout=60
        )

        response.raise_for_status()

        self.size = int(
            response.headers["Content-Length"]
        )

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):

        if whence == io.SEEK_SET:
            new_position = offset

        elif whence == io.SEEK_CUR:
            new_position = self.position + offset

        elif whence == io.SEEK_END:
            new_position = self.size + offset

        else:
            raise ValueError("Invalid whence")

        if new_position < 0:
            raise ValueError("Negative seek position")

        self.position = new_position

        return self.position

    def read(self, size=-1):

        if self.position >= self.size:
            return b""

        if size is None or size < 0:
            size = self.size - self.position

        start = self.position
        end = min(
            self.position + size - 1,
            self.size - 1
        )

        headers = {
            "Range": f"bytes={start}-{end}"
        }

        response = requests.get(
            self.url,
            headers=headers,
            timeout=120
        )

        response.raise_for_status()

        data = response.content

        self.position += len(data)

        return data


# ============================================================
# LOAD SELECTED CALLS
# ============================================================

df = pd.read_csv(SELECTED_CSV)

print("=" * 60)
print("VISHGUARD SELECTED AUDIO EXTRACTION")
print("=" * 60)

print(f"Selected calls: {len(df)}")

if len(df) != 40:
    raise ValueError(
        f"Expected 40 selected calls, found {len(df)}"
    )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

SPOOF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BONAFIDE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONNECT
# ============================================================

print()
print("Connecting to VISHGUARD archive...")

reader = RemoteZipReader(ZIP_URL)

print(
    f"Remote ZIP size: "
    f"{reader.size / (1024 ** 3):.2f} GB"
)


# ============================================================
# OPEN REMOTE ZIP
# ============================================================

with zipfile.ZipFile(reader, "r") as z:

    print()
    print("Archive opened successfully.")

    extracted = 0
    missing = []

    print()
    print("Extracting selected files...")
    print()

    for _, row in df.iterrows():

        file_id = str(row["id"])
        original_type = str(row["type"]).strip().lower()

        filename = file_id + ".wav"

        archive_path = ZIP_PREFIX + filename

        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        try:
            info = z.getinfo(archive_path)

        except KeyError:

            print(f"[MISSING] {filename}")

            missing.append(filename)

            continue

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        if original_type == "fraudulent":

            output_dir = SPOOF_DIR

        elif original_type == "legitimate":

            output_dir = BONAFIDE_DIR

        else:

            print(
                f"[ERROR] Unknown type: "
                f"{original_type}"
            )

            continue

        output_path = output_dir / filename

        # ----------------------------------------------------
        # EXTRACT
        # ----------------------------------------------------

        with z.open(info) as source:

            with open(
                output_path,
                "wb"
            ) as target:

                shutil.copyfileobj(
                    source,
                    target
                )

        extracted += 1

        print(
            f"[{extracted:02d}/40] "
            f"{filename} -> {original_type}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)

print(f"Selected : 40")
print(f"Extracted: {extracted}")
print(f"Missing  : {len(missing)}")

print()
print(
    f"Spoof files: "
    f"{len(list(SPOOF_DIR.glob('*.wav')))}"
)

print(
    f"Bonafide files: "
    f"{len(list(BONAFIDE_DIR.glob('*.wav')))}"
)

if missing:

    print()
    print("Missing files:")

    for filename in missing:
        print(" -", filename)

else:

    print()
    print("SUCCESS!")
    print("All 40 selected VISHGUARD files were extracted.")

print("=" * 60)