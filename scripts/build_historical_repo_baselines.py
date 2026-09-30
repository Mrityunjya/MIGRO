import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import polars as pl


RAW = Path("data/raw/coupjava/coupjava-coarse.jsonl")
OUTPUT = Path("data/processed/historical_repo_baselines.parquet")


def parse_timestamp(value):
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        if value.isdigit():
            return datetime.fromtimestamp(int(value))
    except (ValueError, OverflowError, OSError):
        pass

    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


def version_number(value):
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def median(values):
    if not values:
        return None

    values = sorted(values)
    n = len(values)
    mid = n // 2

    if n % 2:
        return float(values[mid])

    return float((values[mid - 1] + values[mid]) / 2)


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW}")

    records = []

    with RAW.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            repo = record.get("repo_name")

            if not repo:
                continue

            start = parse_timestamp(record.get("old_timestamp"))
            end = parse_timestamp(record.get("new_timestamp"))

            if start is None:
                continue

            old_version = version_number(record.get("old_version"))
            new_version = version_number(record.get("new_version"))

            duration = None

            if start is not None and end is not None:
                duration = (end - start).total_seconds() / 86400

                if duration < 0:
                    duration = None

            patch = record.get("patch") or ""

            files_changed = patch.count("diff --git ")

            additions = 0
            deletions = 0

            for patch_line in patch.splitlines():
                if patch_line.startswith("+++") or patch_line.startswith("---"):
                    continue

                if patch_line.startswith("+"):
                    additions += 1
                elif patch_line.startswith("-"):
                    deletions += 1

            changed_lines = additions + deletions

            version_distance = None

            if old_version is not None and new_version is not None:
                version_distance = new_version - old_version

            records.append(
                {
                    "repo_name": repo,
                    "old_commit": record.get("old_commit"),
                    "new_commit": record.get("new_commit"),
                    "migration_timestamp": start,
                    "migration_duration_days": duration,
                    "files_changed": files_changed,
                    "changed_lines": changed_lines,
                    "version_distance": version_distance,
                }
            )

    if not records:
        raise ValueError("No usable migration records found.")

    # Process chronologically so each migration only sees
    # information from earlier migrations of the same repository.
    records.sort(
        key=lambda x: (
            x["repo_name"],
            x["migration_timestamp"],
        )
    )

    history = defaultdict(
        lambda: {
            "durations": [],
            "files_changed": [],
            "changed_lines": [],
            "version_distances": [],
        }
    )

    output_rows = []

    for record in records:
        repo = record["repo_name"]
        previous = history[repo]

        previous_count = len(previous["durations"])

        output_rows.append(
            {
                "repo_name": repo,
                "old_commit": record["old_commit"],
                "new_commit": record["new_commit"],
                "migration_timestamp": record["migration_timestamp"],

                # Historical repository experience.
                "historical_migration_count": previous_count,

                "historical_median_duration_days": median(
                    previous["durations"]
                ),

                "historical_median_files_changed": median(
                    previous["files_changed"]
                ),

                "historical_median_changed_lines": median(
                    previous["changed_lines"]
                ),

                "historical_median_version_distance": median(
                    previous["version_distances"]
                ),

                "historical_max_duration_days": (
                    max(previous["durations"])
                    if previous["durations"]
                    else None
                ),

                "historical_large_migration_count": sum(
                    1
                    for value in previous["files_changed"]
                    if value >= 20
                ),

                "is_first_observed_migration": previous_count == 0,

                # Current migration target/context.
                "old_version": record["old_commit"],
                "version_distance": record["version_distance"],

                # Outcomes retained separately for later evaluation.
                "actual_duration_days": record[
                    "migration_duration_days"
                ],
                "actual_files_changed": record[
                    "files_changed"
                ],
                "actual_changed_lines": record[
                    "changed_lines"
                ],
            }
        )

        # Only AFTER creating the feature row do we add
        # the current migration to repository history.
        if record["migration_duration_days"] is not None:
            previous["durations"].append(
                record["migration_duration_days"]
            )

        if record["files_changed"] is not None:
            previous["files_changed"].append(
                record["files_changed"]
            )

        if record["changed_lines"] is not None:
            previous["changed_lines"].append(
                record["changed_lines"]
            )

        if record["version_distance"] is not None:
            previous["version_distances"].append(
                record["version_distance"]
            )

    df = pl.DataFrame(output_rows)

    # Correct the version field: preserve the actual old Java version.
    # Re-read it from the raw records through a stable mapping.
    version_map = {}

    for record in records:
        key = (
            record["repo_name"],
            record["old_commit"],
            record["new_commit"],
        )
        version_map[key] = None

    # Build version information directly from the source again.
    with RAW.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = (
                record.get("repo_name"),
                record.get("old_commit"),
                record.get("new_commit"),
            )

            version_map[key] = version_number(
                record.get("old_version")
            )

    old_versions = []

    for row in df.iter_rows(named=True):
        key = (
            row["repo_name"],
            row["old_commit"],
            row["new_commit"],
        )
        old_versions.append(version_map.get(key))

    df = df.with_columns(
        pl.Series("old_version", old_versions, dtype=pl.Int64)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT)

    print("Historical repository baseline analysis complete.")
    print(f"Rows: {df.height}")
    print(f"Columns: {df.width}")
    print(f"Output: {OUTPUT}")
    print()

    print(
        df.select(
            [
                "repo_name",
                "historical_migration_count",
                "historical_median_duration_days",
                "historical_median_files_changed",
                "historical_median_changed_lines",
                "historical_median_version_distance",
                "is_first_observed_migration",
            ]
        ).head(15)
    )


if __name__ == "__main__":
    main()
