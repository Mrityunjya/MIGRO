import json
from collections import Counter
from pathlib import Path


DATASET_PATH = Path(
    "data/raw/coupjava/coupjava-coarse.jsonl"
)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    records = []
    field_counts = Counter()

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            records.append(record)

            for field in record:
                field_counts[field] += 1

            if len(records) == 3:
                break

    print("=" * 60)
    print("MIGRO DATASET INSPECTION")
    print("=" * 60)

    print(f"\nSample records: {len(records)}")

    print("\nFields:")
    for field, count in sorted(field_counts.items()):
        print(f"  {field}: {count}")

    print("\nFirst record:")
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main()