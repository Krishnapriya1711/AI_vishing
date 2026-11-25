import pandas as pd
import requests
import zipfile


ZIP_URL = (
    "https://zenodo.org/records/20429763/"
    "files/vishguard_dataset.zip?download=1"
)

CSV_PATH = "data/metadata/vishguard_selected_40.csv"


class RemoteZipReader:

    def __init__(self, url):
        self.url = url
        self.pos = 0

        r = requests.head(url, allow_redirects=True)
        r.raise_for_status()

        self.size = int(r.headers["Content-Length"])

        print(
            f"Remote ZIP size: "
            f"{self.size / (1024**3):.2f} GB"
        )

    def seek(self, offset, whence=0):

        if whence == 0:
            self.pos = offset

        elif whence == 1:
            self.pos += offset

        elif whence == 2:
            self.pos = self.size + offset

        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):

        if self.pos >= self.size:
            return b""

        if size < 0:
            size = self.size - self.pos

        end = min(
            self.pos + size - 1,
            self.size - 1
        )

        r = requests.get(
            self.url,
            headers={
                "Range": f"bytes={self.pos}-{end}"
            },
            allow_redirects=True,
            timeout=120
        )

        r.raise_for_status()

        data = r.content
        self.pos += len(data)

        return data

    def close(self):
        pass


# ---------------------------------------------------------
# Load selected calls
# ---------------------------------------------------------

df = pd.read_csv(CSV_PATH)

print(f"Selected files: {len(df)}")

if "audio_path" not in df.columns:
    raise RuntimeError(
        "Column 'audio_path' not found."
    )


# ---------------------------------------------------------
# Open remote ZIP
# ---------------------------------------------------------

print("\nConnecting to remote ZIP...")

remote = RemoteZipReader(ZIP_URL)

with zipfile.ZipFile(remote) as z:

    names = set(z.namelist())

    print(
        f"Archive contains "
        f"{len(names)} entries."
    )

    found = []
    missing = []

    for _, row in df.iterrows():

        filename = str(row["audio_path"])

        # Convert Windows separator to ZIP separator
        filename = filename.replace("\\", "/")

        # Exact archive structure:
        # vishguard_dataset/audio/<filename>

        if filename.startswith("audio/"):
            archive_path = (
                "vishguard_dataset/" + filename
            )
        else:
            archive_path = (
                "vishguard_dataset/audio/" + filename
            )

        if archive_path in names:
            found.append(archive_path)
        else:
            missing.append(archive_path)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("VISHGUARD ARCHIVE CHECK")
print("=" * 60)

print(f"Selected: {len(df)}")
print(f"Found:    {len(found)}")
print(f"Missing:  {len(missing)}")

if missing:

    print("\nMissing files:")

    for filename in missing:
        print("  ", filename)

else:

    print("\nSUCCESS!")
    print(
        "All 40 selected VISHGUARD WAV files "
        "exist in the Zenodo archive."
    )