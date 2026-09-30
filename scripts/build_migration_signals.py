from pathlib import Path

import polars as pl


INPUT_PATH = Path(
    "data/processed/migration_profile.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/migration_signals.parquet"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Profile dataset not found: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    print("=" * 70)
    print("MIGRO - MIGRATION DIFFICULTY SIGNALS")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Migration scale
    # ---------------------------------------------------------

    df = df.with_columns(
        (
            pl.col("files_changed").cast(pl.Float64)
            * pl.col("changed_lines").cast(pl.Float64)
        )
        .sqrt()
        .alias("migration_scale")
    )

    # ---------------------------------------------------------
    # 2. Change intensity
    # ---------------------------------------------------------

    df = df.with_columns(
        (
            pl.col("changed_lines").cast(pl.Float64)
            / pl.col("files_changed")
            .cast(pl.Float64)
            .clip(lower_bound=1.0)
        ).alias("change_intensity")
    )

    # ---------------------------------------------------------
    # 3. Upgrade distance
    # ---------------------------------------------------------

    df = df.with_columns(
        pl.col("version_distance")
        .abs()
        .cast(pl.Float64)
        .alias("upgrade_distance")
    )

    # ---------------------------------------------------------
    # 4. Normalized change size
    # ---------------------------------------------------------

    df = df.with_columns(
        (
            pl.col("log_changed_lines")
            + pl.col("log_files_changed")
        ).alias("normalized_change_size")
    )

    # ---------------------------------------------------------
    # 5. Extreme migration indicators
    # ---------------------------------------------------------

    df = df.with_columns(
        [
            (
                pl.col("lines_outlier")
                | pl.col("files_outlier")
            ).alias("extreme_change"),

            (
                pl.col("duration_outlier")
            ).alias("extreme_duration"),
        ]
    )

    # ---------------------------------------------------------
    # 6. Migration profile dimensions
    # ---------------------------------------------------------

    df = df.with_columns(
        [
            (
                pl.col("log_files_changed")
            ).alias("scope_signal"),

            (
                pl.col("log_changed_lines")
            ).alias("change_volume_signal"),

            (
                pl.col("change_intensity")
                .log1p()
            ).alias("intensity_signal"),

            (
                pl.col("upgrade_distance")
                .log1p()
            ).alias("version_jump_signal"),

            (
                pl.col("log_duration_days")
            ).alias("temporal_signal"),
        ]
    )

    # ---------------------------------------------------------
    # 7. Inspect signals
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("DERIVED MIGRATION SIGNALS")
    print("-" * 70)

    signal_columns = [
        "repo_name",
        "old_version",
        "new_version",
        "upgrade_distance",
        "files_changed",
        "changed_lines",
        "migration_scale",
        "change_intensity",
        "normalized_change_size",
        "extreme_change",
        "extreme_duration",
    ]

    print(
        df.select(signal_columns).head(10)
    )

    # ---------------------------------------------------------
    # 8. Signal correlations
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("SIGNAL CORRELATIONS")
    print("-" * 70)

    correlation_columns = [
        "upgrade_distance",
        "migration_scale",
        "change_intensity",
        "normalized_change_size",
        "log_duration_days",
    ]

    print(
        df.select(correlation_columns).corr()
    )

    # ---------------------------------------------------------
    # 9. Save
    # ---------------------------------------------------------

    df.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {df.height:,}")
    print(f"Columns: {df.width}")


if __name__ == "__main__":
    main()
