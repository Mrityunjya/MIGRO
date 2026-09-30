import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import polars as pl


RAW = Path("data/raw/coupjava/coupjava-coarse.jsonl")
OUTPUT = Path("data/processed/temporal_migration_trends.parquet")


def parse_timestamp(value):
    if value is None:
        return None

    value = str(value).strip()

    # Unix timestamp
    try:
        if value.isdigit():
            return datetime.fromtimestamp(int(value))
    except (ValueError, OverflowError, OSError):
        pass

    # Common ISO formats
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


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW}")

    monthly = defaultdict(
        lambda: {
            "migration_count": 0,
            "durations": [],
            "files_changed": [],
            "changed_lines": [],
            "version_distances": [],
        }
    )

    yearly = defaultdict(
        lambda: {
            "migration_count": 0,
            "durations": [],
            "files_changed": [],
            "changed_lines": [],
            "version_distances": [],
        }
    )

    total_records = 0
    valid_timestamps = 0

    with RAW.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_records += 1

            start = parse_timestamp(record.get("old_timestamp"))
            end = parse_timestamp(record.get("new_timestamp"))

            # Use migration start timestamp for temporal grouping.
            timestamp = start or end

            if timestamp is None:
                continue

            valid_timestamps += 1

            period = timestamp.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            year = timestamp.year

            old_version = version_number(record.get("old_version"))
            new_version = version_number(record.get("new_version"))

            duration = None

            if start is not None and end is not None:
                duration = (end - start).total_seconds() / 86400

                if duration < 0:
                    duration = None

            patch = record.get("patch") or ""

            # Estimate changed files/lines from unified diff.
            files = patch.count("diff --git ")
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

            for bucket in (monthly[period], yearly[year]):
                bucket["migration_count"] += 1

                if duration is not None:
                    bucket["durations"].append(duration)

                if files > 0:
                    bucket["files_changed"].append(files)

                if changed_lines > 0:
                    bucket["changed_lines"].append(changed_lines)

                if version_distance is not None:
                    bucket["version_distances"].append(version_distance)

    if valid_timestamps == 0:
        raise ValueError(
            "No valid timestamps found in the raw dataset."
        )

    def median(values):
        if not values:
            return None
        return float(pl.Series(values).median())

    def quantile(values, q):
        if not values:
            return None
        return float(pl.Series(values).quantile(q))

    rows = []

    for period in sorted(monthly):
        bucket = monthly[period]

        rows.append(
            {
                "migration_period": period,
                "migration_year": period.year,
                "migration_month": period.month,
                "migration_count": bucket["migration_count"],
                "median_duration_days": median(bucket["durations"]),
                "p75_duration_days": quantile(bucket["durations"], 0.75),
                "median_files_changed": median(bucket["files_changed"]),
                "median_changed_lines": median(bucket["changed_lines"]),
                "median_version_distance": median(
                    bucket["version_distances"]
                ),
            }
        )

    trends = pl.DataFrame(rows).sort("migration_period")

    yearly_rows = []

    for year in sorted(yearly):
        bucket = yearly[year]

        yearly_rows.append(
            {
                "migration_year": year,
                "annual_migration_count": bucket["migration_count"],
                "annual_median_duration_days": median(
                    bucket["durations"]
                ),
                "annual_median_files_changed": median(
                    bucket["files_changed"]
                ),
                "annual_median_changed_lines": median(
                    bucket["changed_lines"]
                ),
                "annual_median_version_distance": median(
                    bucket["version_distances"]
                ),
            }
        )

    yearly_df = pl.DataFrame(yearly_rows)

    trends = trends.join(
        yearly_df,
        on="migration_year",
        how="left",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    trends.write_parquet(OUTPUT)

    print("Temporal migration analysis complete.")
    print(f"Raw records: {total_records}")
    print(f"Records with valid timestamps: {valid_timestamps}")
    print(f"Temporal periods: {trends.height}")
    print(f"Output: {OUTPUT}")
    print()
    print(trends)


if __name__ == "__main__":
    main()
