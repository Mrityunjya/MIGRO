from pathlib import Path

import polars as pl


INPUT_PATH = Path(
    "data/processed/migration_difficulty_analysis.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/migration_assessments.parquet"
)


def percentile(value: float, values: list[float]) -> float:
    if value is None or not values:
        return 0.0

    lower = sum(v <= value for v in values)

    return lower / len(values)


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    print("=" * 70)
    print("MIGRO - EXPLAINABLE MIGRATION ASSESSMENT")
    print("=" * 70)

    print(f"\nMigrations: {df.height:,}")

    required = [
        "files_changed",
        "changed_lines",
        "migration_duration_days",
    ]

    available = [
        column
        for column in required
        if column in df.columns
    ]

    if not available:
        raise ValueError(
            "No migration outcome columns available."
        )

    # ---------------------------------------------------------
    # Build empirical thresholds from the observed dataset.
    # ---------------------------------------------------------

    distributions = {}

    for column in available:
        distributions[column] = (
            df.select(
                pl.col(column)
                .cast(pl.Float64)
                .drop_nulls()
            )
            .to_series()
            .to_list()
        )

    # ---------------------------------------------------------
    # Generate evidence for every migration.
    # ---------------------------------------------------------

    assessments = []

    for row in df.iter_rows(named=True):

        evidence = []
        indicator_count = 0

        # -----------------------------------------------------
        # Change volume
        # -----------------------------------------------------

        if "files_changed" in distributions:

            value = row.get("files_changed")

            if value is not None:

                p = percentile(
                    float(value),
                    distributions["files_changed"],
                )

                if p >= 0.75:
                    evidence.append(
                        "High file-change volume"
                    )
                    indicator_count += 1

        if "changed_lines" in distributions:

            value = row.get("changed_lines")

            if value is not None:

                p = percentile(
                    float(value),
                    distributions["changed_lines"],
                )

                if p >= 0.75:
                    evidence.append(
                        "High line-change volume"
                    )
                    indicator_count += 1

        # -----------------------------------------------------
        # Migration duration
        # -----------------------------------------------------

        if "migration_duration_days" in distributions:

            value = row.get(
                "migration_duration_days"
            )

            if value is not None:

                p = percentile(
                    float(value),
                    distributions[
                        "migration_duration_days"
                    ],
                )

                if p >= 0.75:
                    evidence.append(
                        "Long migration duration"
                    )
                    indicator_count += 1

        # -----------------------------------------------------
        # Version distance
        # -----------------------------------------------------

        if "version_distance" in row:

            value = row.get("version_distance")

            if value is not None and value >= 3:
                evidence.append(
                    "Large Java version gap"
                )
                indicator_count += 1

        # -----------------------------------------------------
        # Dependency activity
        # -----------------------------------------------------

        if "dependency_change_volume" in row:

            value = row.get(
                "dependency_change_volume"
            )

            if value is not None and value > 0:
                evidence.append(
                    "Dependency/build changes present"
                )
                indicator_count += 1

        # -----------------------------------------------------
        # Source/import activity
        # -----------------------------------------------------

        if "java_source_changed" in row:

            value = row.get(
                "java_source_changed"
            )

            if value:
                evidence.append(
                    "Java source changes present"
                )

        if "imports_changed" in row:

            value = row.get(
                "imports_changed"
            )

            if value:
                evidence.append(
                    "Import changes present"
                )

        # -----------------------------------------------------
        # Descriptive assessment
        # -----------------------------------------------------

        if indicator_count >= 3:
            assessment = (
                "Multiple high-effort characteristics observed"
            )
        elif indicator_count == 2:
            assessment = (
                "Several elevated migration characteristics observed"
            )
        elif indicator_count == 1:
            assessment = (
                "One elevated migration characteristic observed"
            )
        else:
            assessment = (
                "No elevated characteristics detected"
            )

        assessments.append(
            {
                "repo_name": row.get("repo_name"),
                "old_version": row.get("old_version"),
                "new_version": row.get("new_version"),
                "old_commit": row.get("old_commit"),
                "new_commit": row.get("new_commit"),
                "difficulty_segment": row.get(
                    "difficulty_segment"
                ),
                "observed_indicator_count": indicator_count,
                "assessment": assessment,
                "evidence": " | ".join(evidence)
                if evidence
                else "None",
            }
        )

    result = pl.DataFrame(assessments)

    result.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("ASSESSMENT SUMMARY")
    print("-" * 70)

    print(
        result.group_by("assessment")
        .agg(
            pl.len().alias("migrations")
        )
        .sort(
            "migrations",
            descending=True,
        )
    )

    print("\n" + "-" * 70)
    print("SAMPLE ASSESSMENTS")
    print("-" * 70)

    print(
        result.select(
            [
                "repo_name",
                "old_version",
                "new_version",
                "observed_indicator_count",
                "assessment",
                "evidence",
            ]
        ).head(15)
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows:  {result.height:,}")


if __name__ == "__main__":
    main()
