from pathlib import Path

import polars as pl


INPUT_PATH = Path(
    "data/processed/migration_difficulty_analysis.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/migration_path_benchmarks.parquet"
)


METRICS = [
    "migration_duration_days",
    "files_changed",
    "changed_lines",
    "dependency_change_volume",
    "dependency_lines_added",
    "dependency_lines_removed",
    "version_change_lines",
    "java_files_changed",
    "import_change_lines",
]


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    print("=" * 70)
    print("MIGRO - MIGRATION PATH BENCHMARKS")
    print("=" * 70)

    print(f"\nMigrations: {df.height:,}")

    required = [
        "old_version",
        "new_version",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ---------------------------------------------------------
    # Canonical migration path
    # ---------------------------------------------------------

    df = df.with_columns(
        (
            pl.col("old_version").cast(pl.String)
            + "→"
            + pl.col("new_version").cast(pl.String)
        ).alias("migration_path")
    )

    metrics = [
        metric
        for metric in METRICS
        if metric in df.columns
    ]

    aggregations = [
        pl.len().alias("migration_count")
    ]

    for metric in metrics:

        column = pl.col(metric).cast(pl.Float64)

        aggregations.extend(
            [
                column.median().alias(
                    f"{metric}_median"
                ),
                column.quantile(0.75).alias(
                    f"{metric}_p75"
                ),
                column.quantile(0.90).alias(
                    f"{metric}_p90"
                ),
            ]
        )

    benchmarks = (
        df.group_by("migration_path")
        .agg(*aggregations)
        .sort(
            "migration_count",
            descending=True,
        )
    )

    # ---------------------------------------------------------
    # Print benchmark table
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("MIGRATION PATH BENCHMARKS")
    print("-" * 70)

    display_columns = [
        "migration_path",
        "migration_count",
    ]

    for metric in [
        "migration_duration_days",
        "files_changed",
        "changed_lines",
        "dependency_change_volume",
    ]:
        median_column = f"{metric}_median"

        if median_column in benchmarks.columns:
            display_columns.append(median_column)

    print(
        benchmarks.select(display_columns)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    benchmarks.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Paths: {benchmarks.height:,}")


if __name__ == "__main__":
    main()
