from pathlib import Path

import polars as pl


INPUT_PATH = Path(
    "data/processed/migration_signals.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/statistical_validation.parquet"
)


TARGETS = [
    "migration_duration_days",
    "changed_lines",
    "files_changed",
]


SIGNALS = [
    "version_distance",
    "change_intensity",
    "dependency_change_volume",
    "dependency_lines_added",
    "dependency_lines_removed",
    "version_change_lines",
    "java_files_changed",
    "import_change_lines",
]


def correlation_table(df: pl.DataFrame) -> pl.DataFrame:
    rows = []

    available_signals = [
        column
        for column in SIGNALS
        if column in df.columns
    ]

    available_targets = [
        column
        for column in TARGETS
        if column in df.columns
    ]

    for signal in available_signals:
        for target in available_targets:

            pair = (
                df.select(
                    [
                        pl.col(signal).cast(pl.Float64),
                        pl.col(target).cast(pl.Float64),
                    ]
                )
                .drop_nulls()
            )

            if pair.height < 3:
                continue

            correlation = pair.select(
                pl.corr(
                    signal,
                    target,
                ).alias("correlation")
            ).item()

            rows.append(
                {
                    "signal": signal,
                    "target": target,
                    "observations": pair.height,
                    "pearson_correlation": correlation,
                }
            )

    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("pearson_correlation")
            .round(4)
        )
        .sort(
            "pearson_correlation",
            descending=True,
        )
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    print("=" * 70)
    print("MIGRO - STATISTICAL SIGNAL VALIDATION")
    print("=" * 70)

    print(f"\nRows: {df.height:,}")

    print("\n" + "-" * 70)
    print("AVAILABLE COLUMNS")
    print("-" * 70)

    print(df.columns)

    correlations = correlation_table(df)

    print("\n" + "-" * 70)
    print("SIGNAL / OUTCOME CORRELATIONS")
    print("-" * 70)

    if correlations.height:
        print(correlations)
    else:
        print("No compatible signals found.")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    correlations.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {correlations.height:,}")


if __name__ == "__main__":
    main()
