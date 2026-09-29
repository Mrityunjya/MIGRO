from pathlib import Path
from urllib.request import urlretrieve


ZENODO_URL = (
    "https://zenodo.org/records/15293313/files/"
    "coupjava-coarse.jsonl?download=1"
)


def download_dataset(
    output_dir: str = "data/raw/coupjava",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    destination = output_path / "coupjava-coarse.jsonl"

    if destination.exists():
        print(f"Dataset already exists: {destination}")
        return destination

    print("Downloading CoUpJava-Coarse...")
    print("This file is approximately 638 MB.")

    urlretrieve(ZENODO_URL, destination)

    print(f"Saved dataset to: {destination}")

    return destination


if __name__ == "__main__":
    download_dataset()