import json
from pathlib import Path
from typing import Iterator


DEFAULT_DATASET_PATH = Path(
    "data/raw/coupjava/coupjava-coarse.jsonl"
)


def load_jsonl(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> Iterator[dict]:
    """Read JSONL records incrementally."""

    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number}"
                )

            yield record


def count_records(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> int:
    """Count records without retaining them in memory."""

    return sum(1 for _ in load_jsonl(path))


if __name__ == "__main__":
    print(f"Total records: {count_records()}")
