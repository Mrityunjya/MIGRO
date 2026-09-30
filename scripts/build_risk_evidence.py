from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "historical_repo_baselines.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "migration_risk_evidence.parquet"
)


def build_evidence(row):

    evidence = []

    historical_count = row.get(
        "historical_migration_count"
    )

    historical_duration = row.get(
        "historical_median_duration_days"
    )

    historical_files = row.get(
        "historical_median_files_changed"
    )

    historical_lines = row.get(
        "historical_median_changed_lines"
    )

    historical_distance = row.get(
        "historical_median_version_distance"
    )

    current_distance = row.get(
        "version_distance"
    )

    historical_large = row.get(
        "historical_large_migration_count"
    )

    is_first = row.get(
        "is_first_observed_migration"
    )

    # ------------------------------------------------------------
    # Historical repository evidence
    # ------------------------------------------------------------

    if (
        historical_count is not None
        and historical_count >= 3
    ):
        evidence.append(
            "Repository has multiple observed migration events"
        )

    if (
        historical_duration is not None
        and historical_duration > 30
    ):
        evidence.append(
            "Historical migrations had a median duration above 30 days"
        )

    if (
        historical_files is not None
        and historical_files > 20
    ):
        evidence.append(
            "Historical migrations had relatively high median file-change volume"
        )

    if (
        historical_lines is not None
        and historical_lines > 500
    ):
        evidence.append(
            "Historical migrations had relatively high median line-change volume"
        )

    if (
        historical_large is not None
        and historical_large >= 2
    ):
        evidence.append(
            "Repository has multiple historically large migrations"
        )

    # ------------------------------------------------------------
    # Current migration context
    # ------------------------------------------------------------

    if (
        current_distance is not None
        and current_distance >= 3
    ):
        evidence.append(
            "Current Java version distance is relatively large"
        )

    if (
        historical_distance is not None
        and current_distance is not None
        and current_distance > historical_distance
    ):
        evidence.append(
            "Current version distance exceeds the repository historical median"
        )

    # ------------------------------------------------------------
    # Missing history
    # ------------------------------------------------------------

    if is_first:
        evidence.append(
            "No earlier migration history is available for this repository"
        )

    # ------------------------------------------------------------
    # Evidence level
    # ------------------------------------------------------------

    evidence_count = len(evidence)

    if evidence_count >= 4:
        evidence_level = "high_evidence"

    elif evidence_count >= 2:
        evidence_level = "moderate_evidence"

    elif evidence_count == 1:
        evidence_level = "limited_evidence"

    else:
        evidence_level = "minimal_evidence"

    return (
        evidence_level,
        evidence_count,
        evidence,
    )


def main():

    print("=" * 70)
    print("MIGRO - Migration Risk Evidence Layer")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    required_columns = [
        "repo_name",
        "migration_timestamp",
        "historical_migration_count",
        "historical_median_duration_days",
        "historical_median_files_changed",
        "historical_median_changed_lines",
        "historical_median_version_distance",
        "historical_large_migration_count",
        "is_first_observed_migration",
        "version_distance",
        "actual_duration_days",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    rows = df.to_dicts()

    output_rows = []

    for row in rows:

        (
            evidence_level,
            evidence_count,
            evidence,
        ) = build_evidence(row)

        output_rows.append(
            {
                "repo_name": row[
                    "repo_name"
                ],

                "migration_timestamp": row[
                    "migration_timestamp"
                ],

                "historical_migration_count": row[
                    "historical_migration_count"
                ],

                "historical_median_duration_days": row[
                    "historical_median_duration_days"
                ],

                "historical_median_files_changed": row[
                    "historical_median_files_changed"
                ],

                "historical_median_changed_lines": row[
                    "historical_median_changed_lines"
                ],

                "historical_median_version_distance": row[
                    "historical_median_version_distance"
                ],

                "historical_large_migration_count": row[
                    "historical_large_migration_count"
                ],

                "current_version_distance": row[
                    "version_distance"
                ],

                "is_first_observed_migration": row[
                    "is_first_observed_migration"
                ],

                "evidence_level": evidence_level,

                "evidence_count": evidence_count,

                "evidence": " | ".join(
                    evidence
                ),

                # Actual outcome is retained only for
                # retrospective evaluation.
                "actual_duration_days": row[
                    "actual_duration_days"
                ],
            }
        )

    result = pl.DataFrame(
        output_rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.write_parquet(
        OUTPUT_PATH
    )

    print(
        f"\nGenerated evidence rows: "
        f"{result.height:,}"
    )

    print("\nEvidence distribution:")

    print(
        result
        .group_by("evidence_level")
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print("\nExample evidence:")

    print(
        result.select(
            [
                "repo_name",
                "evidence_level",
                "evidence_count",
                "evidence",
            ]
        ).head(10)
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print(
        "Migration risk evidence completed."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
