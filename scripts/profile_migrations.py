from pathlib import Path

import polars as pl


FEATURES_PATH = Path(
    "data/processed/migration_features.parquet"
)


def percentile_summary(
    df: pl.DataFrame,
    column: str,
) -> pl.DataFrame:
    return df.select(
        [
            pl.lit(column).alias("metric"),

            pl.col(column)
            .count()
            .cast(pl.Float64)
            .alias("count"),

            pl.col(column)
            .median()
            .cast(pl.Float64)
            .alias("median"),

            pl.col(column)
            .quantile(0.75)
            .cast(pl.Float64)
            .alias("p75"),

            pl.col(column)
            .quantile(0.90)
            .cast(pl.Float64)
            .alias("p90"),

            pl.col(column)
            .quantile(0.95)
            .cast(pl.Float64)
            .alias("p95"),

            pl.col(column)
            .quantile(0.99)
            .cast(pl.Float64)
            .alias("p99"),

            pl.col(column)
            .mean()
            .cast(pl.Float64)
            .alias("mean"),

            pl.col(column)
            .max()
            .cast(pl.Float64)
            .alias("max"),
        ]
    )


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_PATH}"
        )

    df = pl.read_parquet(FEATURES_PATH)

    print("=" * 70)
    print("MIGRO - ROBUST MIGRATION PROFILING")
    print("=" * 70)

    print(f"\nRows: {df.height:,}")

    # ---------------------------------------------------------
    # 1. Robust distribution statistics
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("1. ROBUST DISTRIBUTION STATISTICS")
    print("-" * 70)

    metrics = [
        "files_changed",
        "changed_lines",
        "migration_duration_days",
    ]

    summaries = [
        percentile_summary(df, column)
        for column in metrics
    ]

    print(pl.concat(summaries))

    # ---------------------------------------------------------
    # 2. IQR-based outlier thresholds
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("2. IQR OUTLIER THRESHOLDS")
    print("-" * 70)

    thresholds = {}

    for column in metrics:
        q1 = float(
            df.select(
                pl.col(column).quantile(0.25)
            ).item()
        )

        q3 = float(
            df.select(
                pl.col(column).quantile(0.75)
            ).item()
        )

        iqr = q3 - q1
        upper = q3 + (1.5 * iqr)

        thresholds[column] = upper

        print(
            f"{column}: "
            f"Q1={q1:.2f}, "
            f"Q3={q3:.2f}, "
            f"IQR={iqr:.2f}, "
            f"upper={upper:.2f}"
        )

    # ---------------------------------------------------------
    # 3. Log-transformed change features
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("3. LOG-TRANSFORMED CHANGE FEATURES")
    print("-" * 70)

    df = df.with_columns(
        [
            pl.col("files_changed")
            .cast(pl.Float64)
            .log1p()
            .alias("log_files_changed"),

            pl.col("changed_lines")
            .cast(pl.Float64)
            .log1p()
            .alias("log_changed_lines"),

            pl.col("migration_duration_days")
            .cast(pl.Float64)
            .log1p()
            .alias("log_duration_days"),
        ]
    )

    print(
        df.select(
            [
                "files_changed",
                "log_files_changed",
                "changed_lines",
                "log_changed_lines",
                "migration_duration_days",
                "log_duration_days",
            ]
        ).head(10)
    )

    # ---------------------------------------------------------
    # 4. Change intensity
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("4. CHANGE INTENSITY")
    print("-" * 70)

    df = df.with_columns(
        (
            pl.col("changed_lines")
            .cast(pl.Float64)
            / pl.col("files_changed")
            .cast(pl.Float64)
            .clip(lower_bound=1.0)
        ).alias("changed_lines_per_file")
    )

    print(
        df.select(
            [
                "repo_name",
                "old_version",
                "new_version",
                "files_changed",
                "changed_lines",
                "changed_lines_per_file",
            ]
        )
        .sort(
            "changed_lines_per_file",
            descending=True,
        )
        .head(10)
    )

    # ---------------------------------------------------------
    # 5. Flag extreme observations
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("5. EXTREME OBSERVATIONS")
    print("-" * 70)

    df = df.with_columns(
        [
            (
                pl.col("files_changed")
                > thresholds["files_changed"]
            ).alias("files_outlier"),

            (
                pl.col("changed_lines")
                > thresholds["changed_lines"]
            ).alias("lines_outlier"),

            (
                pl.col("migration_duration_days")
                > thresholds["migration_duration_days"]
            ).alias("duration_outlier"),
        ]
    )

    print(
        df.select(
            [
                pl.col("files_outlier")
                .sum()
                .alias("file_count_outliers"),

                pl.col("lines_outlier")
                .sum()
                .alias("line_count_outliers"),

                pl.col("duration_outlier")
                .sum()
                .alias("duration_outliers"),
            ]
        )
    )

    # ---------------------------------------------------------
    # 6. Robust upgrade-path comparison
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("6. UPGRADE PATHS BY MEDIAN CHANGE SIZE")
    print("-" * 70)

    path_profile = (
        df.group_by(
            [
                "old_version",
                "new_version",
            ]
        )
        .agg(
            pl.len().alias("cases"),

            pl.col("files_changed")
            .median()
            .cast(pl.Float64)
            .alias("median_files"),

            pl.col("changed_lines")
            .median()
            .cast(pl.Float64)
            .alias("median_changed_lines"),

            pl.col("changed_lines")
            .quantile(0.90)
            .cast(pl.Float64)
            .alias("p90_changed_lines"),

            pl.col("migration_duration_days")
            .median()
            .cast(pl.Float64)
            .alias("median_duration_days"),
        )
        .filter(
            pl.col("cases") >= 5
        )
        .sort(
            "median_changed_lines",
            descending=True,
        )
    )

    print(path_profile.head(20))

    # ---------------------------------------------------------
    # 7. Save enriched profiling dataset
    # ---------------------------------------------------------

    output_path = Path(
        "data/processed/migration_profile.parquet"
    )

    df.write_parquet(
        output_path,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
