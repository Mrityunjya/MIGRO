import polars as pl
from pathlib import Path

INPUT = Path("data/processed/migration_feature_matrix.parquet")
OUTPUT = Path("data/processed/migration_anomalies.parquet")


def percentile_expr(column, q):
    return pl.col(column).quantile(q)


def main():
    df = pl.read_parquet(INPUT)

    required = [
        "repo_name",
        "old_version",
        "new_version",
        "migration_duration_days",
        "files_changed",
        "changed_lines",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove invalid outcome values.
    df = df.filter(
        (pl.col("migration_duration_days") >= 0)
        & (pl.col("files_changed") >= 0)
        & (pl.col("changed_lines") >= 0)
    )

    # Calculate global empirical thresholds.
    thresholds = {}

    for column in [
        "migration_duration_days",
        "files_changed",
        "changed_lines",
    ]:
        thresholds[column] = {
            "p95": df[column].quantile(0.95),
            "p99": df[column].quantile(0.99),
        }

    # Build anomaly indicators.
    df = df.with_columns(
        [
            (
                pl.col("migration_duration_days")
                >= thresholds["migration_duration_days"]["p95"]
            ).alias("duration_anomaly"),

            (
                pl.col("files_changed")
                >= thresholds["files_changed"]["p95"]
            ).alias("file_volume_anomaly"),

            (
                pl.col("changed_lines")
                >= thresholds["changed_lines"]["p95"]
            ).alias("line_volume_anomaly"),

            (
                pl.col("migration_duration_days")
                >= thresholds["migration_duration_days"]["p99"]
            ).alias("extreme_duration_anomaly"),

            (
                pl.col("files_changed")
                >= thresholds["files_changed"]["p99"]
            ).alias("extreme_file_volume_anomaly"),

            (
                pl.col("changed_lines")
                >= thresholds["changed_lines"]["p99"]
            ).alias("extreme_line_volume_anomaly"),
        ]
    )

    # Count independent anomalous dimensions.
    df = df.with_columns(
        (
            pl.col("duration_anomaly").cast(pl.Int8)
            + pl.col("file_volume_anomaly").cast(pl.Int8)
            + pl.col("line_volume_anomaly").cast(pl.Int8)
        ).alias("anomaly_dimension_count")
    )

    df = df.with_columns(
        pl.when(pl.col("anomaly_dimension_count") >= 3)
        .then(pl.lit("multi_dimension_anomaly"))
        .when(pl.col("anomaly_dimension_count") == 2)
        .then(pl.lit("two_dimension_anomaly"))
        .when(pl.col("anomaly_dimension_count") == 1)
        .then(pl.lit("single_dimension_anomaly"))
        .otherwise(pl.lit("within_observed_range"))
        .alias("anomaly_segment")
    )

    # Identify extreme cases using the 99th percentile.
    df = df.with_columns(
        (
            pl.col("extreme_duration_anomaly")
            | pl.col("extreme_file_volume_anomaly")
            | pl.col("extreme_line_volume_anomaly")
        ).alias("extreme_anomaly")
    )

    # Human-readable evidence.
    df = df.with_columns(
        pl.when(pl.col("anomaly_dimension_count") == 0)
        .then(pl.lit("No unusually elevated dimensions observed"))

        .when(pl.col("anomaly_dimension_count") == 1)
        .then(
            pl.when(pl.col("duration_anomaly"))
            .then(pl.lit("Unusually long migration duration"))
            .when(pl.col("file_volume_anomaly"))
            .then(pl.lit("Unusually high file-change volume"))
            .otherwise(pl.lit("Unusually high line-change volume"))
        )

        .when(pl.col("anomaly_dimension_count") == 2)
        .then(pl.lit("Two migration dimensions are unusually elevated"))

        .otherwise(
            pl.lit(
                "Duration, file volume, and line volume are all unusually elevated"
            )
        )

        .alias("anomaly_evidence")
    )

    columns = [
        "repo_name",
        "old_version",
        "new_version",
        "migration_duration_days",
        "files_changed",
        "changed_lines",
        "anomaly_dimension_count",
        "anomaly_segment",
        "extreme_anomaly",
        "anomaly_evidence",
    ]

    # Preserve useful identifiers when available.
    for column in ["old_commit", "new_commit"]:
        if column in df.columns:
            columns.append(column)

    result = (
        df.select(columns)
        .sort(
            [
                "anomaly_dimension_count",
                "changed_lines",
            ],
            descending=True,
        )
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(OUTPUT)

    print("Migration anomaly analysis complete.")
    print(f"Rows analyzed: {df.height}")
    print(f"Output: {OUTPUT}")
    print()
    print("Anomaly segments:")
    print(
        result
        .group_by("anomaly_segment")
        .agg(pl.len().alias("migration_count"))
        .sort("migration_count", descending=True)
    )

    print()
    print("Top anomalous migrations:")
    print(result.head(15))


if __name__ == "__main__":
    main()
