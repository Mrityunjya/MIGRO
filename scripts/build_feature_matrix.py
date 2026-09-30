from pathlib import Path

import polars as pl


BASE_PATH = Path("data/processed/migration_signals.parquet")
DEPENDENCY_PATH = Path("data/processed/dependency_signals.parquet")
OUTPUT_PATH = Path("data/processed/migration_feature_matrix.parquet")


def safe_read(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return pl.read_parquet(path)


def main() -> None:
    print("=" * 70)
    print("MIGRO - UNIFIED MIGRATION FEATURE MATRIX")
    print("=" * 70)

    base = safe_read(BASE_PATH)

    print(f"\nBase migrations: {base.height:,}")

    # Dependency signals are generated from the same migration records.
    # Join on stable migration identifiers where available.
    dependency = safe_read(DEPENDENCY_PATH)

    join_candidates = [
        "repo_name",
        "old_commit",
        "new_commit",
    ]

    join_keys = [
        column
        for column in join_candidates
        if column in base.columns and column in dependency.columns
    ]

    if not join_keys:
        raise ValueError(
            "No shared migration identifiers found between datasets."
        )

    print(f"Join keys: {join_keys}")

    # Avoid duplicate non-key columns during the join.
    dependency_columns = [
        column
        for column in dependency.columns
        if column in join_keys
        or column not in base.columns
    ]

    dependency = dependency.select(dependency_columns)

    matrix = base.join(
        dependency,
        on=join_keys,
        how="left",
    )

    # ---------------------------------------------------------
    # Derived migration-level indicators
    # ---------------------------------------------------------

    expressions = []

    if "files_changed" in matrix.columns:
        expressions.append(
            pl.col("files_changed")
            .cast(pl.Float64)
            .log1p()
            .alias("log_files_changed")
        )

    if "changed_lines" in matrix.columns:
        expressions.append(
            pl.col("changed_lines")
            .cast(pl.Float64)
            .log1p()
            .alias("log_changed_lines")
        )

    if "migration_duration_days" in matrix.columns:
        expressions.append(
            pl.col("migration_duration_days")
            .cast(pl.Float64)
            .log1p()
            .alias("log_migration_duration")
        )

    if expressions:
        matrix = matrix.with_columns(expressions)

    # Broad structural categories.
    if "files_changed" in matrix.columns:
        matrix = matrix.with_columns(
            pl.when(pl.col("files_changed") <= 5)
            .then(pl.lit("small"))
            .when(pl.col("files_changed") <= 20)
            .then(pl.lit("medium"))
            .otherwise(pl.lit("large"))
            .alias("change_size_segment")
        )

    if "migration_duration_days" in matrix.columns:
        matrix = matrix.with_columns(
            pl.when(pl.col("migration_duration_days") <= 1)
            .then(pl.lit("fast"))
            .when(pl.col("migration_duration_days") <= 30)
            .then(pl.lit("moderate"))
            .otherwise(pl.lit("long"))
            .alias("duration_segment")
        )

    matrix.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("FEATURE MATRIX")
    print("-" * 70)

    print(f"Migrations: {matrix.height:,}")
    print(f"Features:   {matrix.width:,}")

    print("\nColumns:")
    for column in matrix.columns:
        print(f"  - {column}")

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
