from pathlib import Path

import polars as pl


INPUT_PATH = Path(
    "data/processed/migration_difficulty_analysis.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/difficulty_driver_analysis.parquet"
)


NUMERIC_FEATURES = [
    "version_distance",
    "change_intensity",
    "files_changed",
    "changed_lines",
    "migration_duration_days",
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
    print("MIGRO - DIFFICULTY DRIVER ANALYSIS")
    print("=" * 70)

    print(f"\nMigrations: {df.height:,}")

    if "difficulty_segment" not in df.columns:
        raise ValueError(
            "difficulty_segment is missing from the input dataset."
        )

    features = [
        column
        for column in NUMERIC_FEATURES
        if column in df.columns
    ]

    if not features:
        raise ValueError(
            "No driver features found."
        )

    # ---------------------------------------------------------
    # Aggregate observed characteristics by difficulty segment
    # ---------------------------------------------------------

    aggregations = []

    for feature in features:
        aggregations.extend(
            [
                pl.col(feature)
                .cast(pl.Float64)
                .median()
                .alias(f"{feature}_median"),

                pl.col(feature)
                .cast(pl.Float64)
                .mean()
                .alias(f"{feature}_mean"),
            ]
        )

    summary = (
        df.group_by("difficulty_segment")
        .agg(
            pl.len().alias("migration_count"),
            *aggregations,
        )
        .sort(
            "migration_count",
            descending=True,
        )
    )

    print("\n" + "-" * 70)
    print("DIFFICULTY SEGMENT PROFILE")
    print("-" * 70)

    print(summary)

    # ---------------------------------------------------------
    # Build a compact driver table.
    #
    # For each feature, calculate the ratio between the median
    # observed value in multi-dimension-high migrations and
    # lower-observed-difficulty migrations.
    #
    # This is an association measure, NOT causal attribution.
    # ---------------------------------------------------------

    if "multi_dimension_high" in df["difficulty_segment"].unique():
        high = (
            df.filter(
                pl.col("difficulty_segment")
                == "multi_dimension_high"
            )
        )

        lower = (
            df.filter(
                pl.col("difficulty_segment")
                == "lower_observed_difficulty"
            )
        )

        driver_rows = []

        for feature in features:

            high_median = high.select(
                pl.col(feature)
                .cast(pl.Float64)
                .median()
            ).item()

            lower_median = lower.select(
                pl.col(feature)
                .cast(pl.Float64)
                .median()
            ).item()

            if (
                high_median is None
                or lower_median is None
            ):
                continue

            if lower_median == 0:
                median_ratio = None
            else:
                median_ratio = (
                    high_median / lower_median
                )

            driver_rows.append(
                {
                    "feature": feature,
                    "high_median": high_median,
                    "lower_median": lower_median,
                    "high_to_lower_median_ratio": median_ratio,
                }
            )

        drivers = (
            pl.DataFrame(driver_rows)
            .sort(
                "high_to_lower_median_ratio",
                descending=True,
                nulls_last=True,
            )
        )

    else:
        drivers = pl.DataFrame(
            schema={
                "feature": pl.String,
                "high_median": pl.Float64,
                "lower_median": pl.Float64,
                "high_to_lower_median_ratio": pl.Float64,
            }
        )

    print("\n" + "-" * 70)
    print("OBSERVED DRIVER DIFFERENCES")
    print("-" * 70)

    print(drivers)

    # ---------------------------------------------------------
    # Save both views
    # ---------------------------------------------------------

    summary.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    drivers.write_parquet(
        Path(
            "data/processed/difficulty_driver_comparison.parquet"
        ),
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Segment profile: {OUTPUT_PATH}")
    print(
        "Driver comparison: "
        "data/processed/difficulty_driver_comparison.parquet"
    )


if __name__ == "__main__":
    main()
