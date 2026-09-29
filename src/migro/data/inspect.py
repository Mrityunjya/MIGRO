import json
from pathlib import Path


DATASET_PATH = Path(
    "data/raw/coupjava/coupjava-coarse.jsonl"
)


def inspect_jsonl(path: str | Path = DATASET_PATH) -> None:
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    count = 0

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            count += 1

            if count == 1:
                print("=" * 60)
                print("MIGRO DATASET INSPECTION")
                print("=" * 60)
                print("\nFirst record:")
                print(json.dumps(record, indent=2))

    print(f"\nTotal records: {count}")


if __name__ == "__main__":
    inspect_jsonl()