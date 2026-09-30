from pathlib import Path

import polars as pl


INPUT_PATH = Path(
    "data/processed/migration_feature_matrix.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/migration_difficulty_analysis.parquet"
)


def percentile_column(
    df: pl.DataFrame,
    column: str,
    output: str,
) -> pl.DataFrame:

    if column not in df.columns:
        return df

    value = pl.col(column).cast(pl.Float64)

    return df.with_columns(
        value
        .qcut(
            [0.25, 0.50, 0.75],
            labels=["Q1", "Q2", "Q3", "Q4"],
            allow_duplicates=True,
        )
        .alias(output)
    )


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    print("=" * 70)
    print("MIGRO - MIGRATION DIFFICULTY ANALYSIS")
    print("=" * 70)

    print(f"\nMigrations: {df.height:,}")

    # ---------------------------------------------------------
    # Difficulty dimensions
    # ---------------------------------------------------------

    if "migration_duration_days" in df.columns:
        df = percentile_column(
            df,
            "migration_duration_days",
            "duration_quartile",
        )

    if "changed_lines" in df.columns:
        df = percentile_column(
            df,
            "changed_lines",
            "lines_quartile",
        )

    if "files_changed" in df.columns:
        df = percentile_column(
            df,
            "files_changed",
            "files_quartile",
        )

    # ---------------------------------------------------------
    # Multi-dimensional difficulty indicator
    #
    # This is descriptive segmentation, NOT a probability
    # and NOT a causal risk score.
    # ---------------------------------------------------------

    difficulty_columns = [
        column
        for column in [
            "duration_quartile",
            "lines_quartile",
            "files_quartile",
        ]
        if column in df.columns
    ]

    if difficulty_columns:
        df = df.with_columns(
            sum(
                pl.when(pl.col(column) == "Q4")
                .then(1)
                .otherwise(0)
                for column in difficulty_columns
            ).alias("high_difficulty_dimensions")
        )

        df = df.with_columns(
            pl.when(
                pl.col("high_difficulty_dimensions") >= 2
            )
            .then(pl.lit("multi_dimension_high"))
            .when(
                pl.col("high_difficulty_dimensions") == 1
            )
            .then(pl.lit("single_dimension_high"))
            .otherwise(pl.lit("lower_observed_difficulty"))
            .alias("difficulty_segment")
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("DIFFICULTY SEGMENTS")
    print("-" * 70)

    if "difficulty_segment" in df.columns:
        print(
            df.group_by("difficulty_segment")
            .agg(
                pl.len().alias("migrations"),
                pl.col("migration_duration_days")
                .median()
                .alias("median_duration_days")
                if "migration_duration_days" in df.columns
                else pl.len().alias("_placeholder"),
                pl.col("files_changed")
                .median()
                .alias("median_files_changed")
                if "files_changed" in df.columns
                else pl.len().alias("_placeholder_files"),
            )
            .sort("migrations", descending=True)
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows:  {df.height:,}")
    print(f"Cols:  {df.width:,}")


if __name__ == "__main__":
    main()
