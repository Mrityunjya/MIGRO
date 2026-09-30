from pathlib import Path

import polars as pl


INPUT_PATH = Path(
    "data/processed/migration_signals.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/repository_profiles.parquet"
)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Signal dataset not found: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    print("=" * 70)
    print("MIGRO - REPOSITORY MIGRATION PROFILES")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Repository-level aggregation
    # ---------------------------------------------------------

    profiles = (
        df.group_by("repo_name")
        .agg(
            pl.len().alias("migration_cases"),

            pl.col("old_version")
            .min()
            .alias("minimum_source_version"),

            pl.col("new_version")
            .max()
            .alias("maximum_target_version"),

            pl.col("version_distance")
            .mean()
            .alias("avg_version_distance"),

            pl.col("files_changed")
            .median()
            .alias("median_files_changed"),

            pl.col("files_changed")
            .max()
            .alias("max_files_changed"),

            pl.col("changed_lines")
            .median()
            .alias("median_changed_lines"),

            pl.col("changed_lines")
            .max()
            .alias("max_changed_lines"),

            pl.col("change_intensity")
            .median()
            .alias("median_change_intensity"),

            pl.col("migration_duration_days")
            .median()
            .alias("median_migration_duration"),

            pl.col("migration_duration_days")
            .max()
            .alias("max_migration_duration"),

            pl.col("extreme_change")
            .sum()
            .alias("extreme_change_cases"),

            pl.col("extreme_duration")
            .sum()
            .alias("extreme_duration_cases"),
        )
        .sort(
            "migration_cases",
            descending=True,
        )
    )

    # ---------------------------------------------------------
    # 2. Repository concentration
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("1. REPOSITORY COVERAGE")
    print("-" * 70)

    print(
        f"Unique repositories: {profiles.height:,}"
    )

    print(
        f"Total migration cases: {df.height:,}"
    )

    print(
        f"Average cases per repository: "
        f"{df.height / profiles.height:.2f}"
    )

    # ---------------------------------------------------------
    # 3. Most migration-active repositories
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("2. MOST MIGRATION-ACTIVE REPOSITORIES")
    print("-" * 70)

    print(
        profiles.head(20)
    )

    # ---------------------------------------------------------
    # 4. Repeated migration repositories
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("3. REPOSITORIES WITH REPEATED MIGRATIONS")
    print("-" * 70)

    repeated = (
        profiles
        .filter(
            pl.col("migration_cases") >= 2
        )
        .sort(
            [
                "migration_cases",
                "median_changed_lines",
            ],
            descending=True,
        )
    )

    print(
        f"Repositories with 2+ cases: "
        f"{repeated.height:,}"
    )

    print(
        repeated.head(20)
    )

    # ---------------------------------------------------------
    # 5. Repository-level extreme migrations
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("4. REPOSITORIES WITH EXTREME MIGRATION ACTIVITY")
    print("-" * 70)

    extreme_repositories = (
        profiles
        .filter(
            pl.col("extreme_change_cases") > 0
        )
        .sort(
            "extreme_change_cases",
            descending=True,
        )
    )

    print(
        f"Repositories containing extreme "
        f"change cases: {extreme_repositories.height:,}"
    )

    print(
        extreme_repositories.head(20)
    )

    # ---------------------------------------------------------
    # 6. Save repository profiles
    # ---------------------------------------------------------

    profiles.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Repositories: {profiles.height:,}")
    print(f"Columns: {profiles.width}")


if __name__ == "__main__":
    main()
