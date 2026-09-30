from pathlib import Path

import polars as pl


INPUT_PATH = Path(
    "data/processed/migration_feature_matrix.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/version_transition_trends.parquet"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    print("=" * 70)
    print("MIGRO - VERSION TRANSITION TREND ANALYSIS")
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
    # Normalize Java versions
    # ---------------------------------------------------------

    df = df.with_columns(
        [
            pl.col("old_version")
            .cast(pl.String)
            .alias("old_version"),

            pl.col("new_version")
            .cast(pl.String)
            .alias("new_version"),
        ]
    )

    df = df.with_columns(
        (
            pl.col("old_version")
            + "→"
            + pl.col("new_version")
        ).alias("transition")
    )

    # ---------------------------------------------------------
    # Build transition metrics
    # ---------------------------------------------------------

    aggregations = [
        pl.len().alias("migration_count"),
    ]

    if "migration_duration_days" in df.columns:
        aggregations.extend(
            [
                pl.col("migration_duration_days")
                .cast(pl.Float64)
                .median()
                .alias("median_duration_days"),

                pl.col("migration_duration_days")
                .cast(pl.Float64)
                .quantile(0.75)
                .alias("p75_duration_days"),
            ]
        )

    if "files_changed" in df.columns:
        aggregations.append(
            pl.col("files_changed")
            .cast(pl.Float64)
            .median()
            .alias("median_files_changed")
        )

    if "changed_lines" in df.columns:
        aggregations.append(
            pl.col("changed_lines")
            .cast(pl.Float64)
            .median()
            .alias("median_changed_lines")
        )

    if "version_distance" in df.columns:
        aggregations.append(
            pl.col("version_distance")
            .cast(pl.Float64)
            .median()
            .alias("median_version_distance")
        )

    if "dependency_change_volume" in df.columns:
        aggregations.append(
            pl.col("dependency_change_volume")
            .cast(pl.Float64)
            .median()
            .alias("median_dependency_change_volume")
        )

        aggregations.append(
            (
                pl.col("dependency_change_volume")
                .fill_null(0)
                .cast(pl.Float64)
                > 0
            )
            .mean()
            .alias("dependency_change_share")
        )

    trends = (
        df.group_by("transition")
        .agg(*aggregations)
        .sort(
            "migration_count",
            descending=True,
        )
    )

    # ---------------------------------------------------------
    # Add numeric ordering where possible
    # ---------------------------------------------------------

    trends = trends.with_columns(
        [
            pl.col("transition")
            .str.split_exact("→", 1)
            .struct.field("field_0")
            .cast(pl.Int64, strict=False)
            .alias("from_version"),

            pl.col("transition")
            .str.split_exact("→", 1)
            .struct.field("field_1")
            .cast(pl.Int64, strict=False)
            .alias("to_version"),
        ]
    )

    trends = trends.sort(
        [
            "from_version",
            "to_version",
        ],
        nulls_last=True,
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("VERSION TRANSITION TRENDS")
    print("-" * 70)

    print(trends)

    print("\n" + "-" * 70)
    print("MOST OBSERVED TRANSITIONS")
    print("-" * 70)

    print(
        trends
        .sort(
            "migration_count",
            descending=True,
        )
        .head(20)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    trends.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Transitions: {trends.height:,}")


if __name__ == "__main__":
    main()
