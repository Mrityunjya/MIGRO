import json
from pathlib import Path

import polars as pl


RAW_PATH = Path(
    "data/raw/coupjava/coupjava-coarse.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/migration_features.parquet"
)


def patch_stats(patch: str) -> tuple[int, int, int, int]:
    lines = patch.splitlines()

    files_changed = 0
    additions = 0
    deletions = 0

    for line in lines:
        if line.startswith("diff --git "):
            files_changed += 1
        elif line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1

    changed_lines = additions + deletions

    return (
        files_changed,
        additions,
        deletions,
        changed_lines,
    )


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    with RAW_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            record = json.loads(line)

            old_version = int(record["old_version"])
            new_version = int(record["new_version"])

            old_timestamp = int(record["old_timestamp"])
            new_timestamp = int(record["new_timestamp"])

            (
                files_changed,
                additions,
                deletions,
                changed_lines,
            ) = patch_stats(
                record.get("patch", "")
            )

            duration_days = max(
                0,
                new_timestamp - old_timestamp,
            ) / 86400

            rows.append(
                {
                    "repo_name": record["repo_name"],
                    "old_version": old_version,
                    "new_version": new_version,
                    "version_distance": (
                        new_version - old_version
                    ),
                    "old_commit": record["old_commit"],
                    "new_commit": record["new_commit"],
                    "old_timestamp": old_timestamp,
                    "new_timestamp": new_timestamp,
                    "migration_duration_days": duration_days,
                    "files_changed": files_changed,
                    "lines_added": additions,
                    "lines_deleted": deletions,
                    "changed_lines": changed_lines,
                }
            )

    dataframe = pl.DataFrame(rows)

    dataframe.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("=" * 60)
    print("MIGRO FEATURE DATASET")
    print("=" * 60)

    print(f"\nRows:    {dataframe.height:,}")
    print(f"Columns: {dataframe.width}")
    print(f"Output:  {OUTPUT_PATH}")

    print("\nSchema:")
    print(dataframe.schema)

    print("\nPreview:")
    print(dataframe.head(5))


if __name__ == "__main__":
    main()
