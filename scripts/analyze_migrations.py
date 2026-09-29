from pathlib import Path

import polars as pl


FEATURES_PATH = Path(
    "data/processed/migration_features.parquet"
)


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_PATH}"
        )

    df = pl.read_parquet(FEATURES_PATH)

    print("=" * 70)
    print("MIGRO - EXPLORATORY MIGRATION ANALYSIS")
    print("=" * 70)

    print(f"\nRows: {df.height:,}")
    print(f"Columns: {df.width}")

    # ---------------------------------------------------------
    # 1. Upgrade paths
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("1. JAVA UPGRADE PATHS")
    print("-" * 70)

    paths = (
        df.group_by(
            ["old_version", "new_version"]
        )
        .agg(
            pl.len().alias("cases"),
            pl.col("files_changed")
            .mean()
            .round(2)
            .alias("avg_files_changed"),
            pl.col("changed_lines")
            .mean()
            .round(2)
            .alias("avg_changed_lines"),
            pl.col("migration_duration_days")
            .mean()
            .round(2)
            .alias("avg_duration_days"),
        )
        .sort("cases", descending=True)
    )

    print(paths.head(15))

    # ---------------------------------------------------------
    # 2. Change-size distribution
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("2. CHANGE SIZE DISTRIBUTION")
    print("-" * 70)

    distribution = (
        df.select(
            [
                "files_changed",
                "changed_lines",
                "migration_duration_days",
            ]
        )
        .describe()
    )

    print(distribution)

    # ---------------------------------------------------------
    # 3. Largest migrations
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("3. LARGEST MIGRATION PATCHES")
    print("-" * 70)

    largest = (
        df.select(
            [
                "repo_name",
                "old_version",
                "new_version",
                "files_changed",
                "changed_lines",
                "migration_duration_days",
            ]
        )
        .sort("changed_lines", descending=True)
        .head(10)
    )

    print(largest)

    # ---------------------------------------------------------
    # 4. Longest migration intervals
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("4. LONGEST MIGRATION INTERVALS")
    print("-" * 70)

    longest = (
        df.select(
            [
                "repo_name",
                "old_version",
                "new_version",
                "files_changed",
                "changed_lines",
                "migration_duration_days",
            ]
        )
        .sort(
            "migration_duration_days",
            descending=True,
        )
        .head(10)
    )

    print(longest)

    # ---------------------------------------------------------
    # 5. Numeric correlations
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("5. NUMERIC CORRELATIONS")
    print("-" * 70)

    numeric_columns = [
        "old_version",
        "new_version",
        "version_distance",
        "files_changed",
        "lines_added",
        "lines_deleted",
        "changed_lines",
        "migration_duration_days",
    ]

    correlation = (
        df.select(numeric_columns)
        .corr()
    )

    print(correlation)


if __name__ == "__main__":
    main()
