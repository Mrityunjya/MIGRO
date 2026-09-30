from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "migration_risk_evidence.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "migration_decisions.parquet"
)


def build_decision(row):

    evidence_level = row.get(
        "evidence_level"
    )

    evidence_count = row.get(
        "evidence_count"
    )

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

    historical_large = row.get(
        "historical_large_migration_count"
    )

    current_distance = row.get(
        "current_version_distance"
    )

    is_first = row.get(
        "is_first_observed_migration"
    )

    actions = []
    priorities = []

    # ------------------------------------------------------------
    # 1. Repository history
    # ------------------------------------------------------------

    if (
        historical_count is not None
        and historical_count >= 3
    ):
        actions.append(
            "Review previous migration commits and failure patterns"
        )

        priorities.append(
            "historical_migration_review"
        )

    # ------------------------------------------------------------
    # 2. Historical duration
    # ------------------------------------------------------------

    if (
        historical_duration is not None
        and historical_duration > 30
    ):
        actions.append(
            "Allocate additional migration validation time"
        )

        priorities.append(
            "schedule_validation"
        )

    # ------------------------------------------------------------
    # 3. File-change complexity
    # ------------------------------------------------------------

    if (
        historical_files is not None
        and historical_files > 20
    ):
        actions.append(
            "Perform a repository-wide compatibility review"
        )

        priorities.append(
            "repository_review"
        )

    # ------------------------------------------------------------
    # 4. Line-change volume
    # ------------------------------------------------------------

    if (
        historical_lines is not None
        and historical_lines > 500
    ):
        actions.append(
            "Use staged changes and focused regression testing"
        )

        priorities.append(
            "staged_testing"
        )

    # ------------------------------------------------------------
    # 5. Historically large migrations
    # ------------------------------------------------------------

    if (
        historical_large is not None
        and historical_large >= 2
    ):
        actions.append(
            "Inspect previous large migrations for reusable migration patterns"
        )

        priorities.append(
            "large_migration_review"
        )

    # ------------------------------------------------------------
    # 6. Version distance
    # ------------------------------------------------------------

    if (
        current_distance is not None
        and current_distance >= 3
    ):
        actions.append(
            "Run explicit Java compatibility and dependency validation"
        )

        priorities.append(
            "compatibility_testing"
        )

    # ------------------------------------------------------------
    # 7. First observed migration
    # ------------------------------------------------------------

    if is_first:
        actions.append(
            "Establish a migration baseline because repository history is limited"
        )

        priorities.append(
            "baseline_establishment"
        )

    # ------------------------------------------------------------
    # 8. Evidence level
    # ------------------------------------------------------------

    if evidence_level == "high_evidence":

        decision = (
            "Prepare with expanded validation"
        )

    elif evidence_level == "moderate_evidence":

        decision = (
            "Prepare with targeted validation"
        )

    elif evidence_level == "limited_evidence":

        decision = (
            "Perform focused pre-migration review"
        )

    else:

        decision = (
            "Proceed with standard migration validation"
        )

    # ------------------------------------------------------------
    # 9. Remove duplicate actions
    # ------------------------------------------------------------

    actions = list(
        dict.fromkeys(actions)
    )

    priorities = list(
        dict.fromkeys(priorities)
    )

    # ------------------------------------------------------------
    # 10. Decision confidence descriptor
    # ------------------------------------------------------------

    if evidence_count is None:
        evidence_count = 0

    if evidence_count >= 4:
        confidence = (
            "multiple independent evidence dimensions"
        )

    elif evidence_count >= 2:
        confidence = (
            "several observed evidence dimensions"
        )

    elif evidence_count == 1:
        confidence = (
            "single observed evidence dimension"
        )

    else:
        confidence = (
            "limited observed evidence"
        )

    return (
        decision,
        confidence,
        actions,
        priorities,
    )


def main():

    print("=" * 70)
    print("MIGRO - Migration Decision Engine")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    df = pl.read_parquet(
        INPUT_PATH
    )

    required_columns = [
        "repo_name",
        "migration_timestamp",
        "evidence_level",
        "evidence_count",
        "evidence",
        "historical_migration_count",
        "historical_median_duration_days",
        "historical_median_files_changed",
        "historical_median_changed_lines",
        "historical_large_migration_count",
        "current_version_distance",
        "is_first_observed_migration",
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

    output_rows = []

    for row in df.to_dicts():

        (
            decision,
            confidence,
            actions,
            priorities,
        ) = build_decision(row)

        output_rows.append(
            {
                "repo_name": row[
                    "repo_name"
                ],

                "migration_timestamp": row[
                    "migration_timestamp"
                ],

                "evidence_level": row[
                    "evidence_level"
                ],

                "evidence_count": row[
                    "evidence_count"
                ],

                "evidence": row[
                    "evidence"
                ],

                "decision": decision,

                "decision_basis": confidence,

                "recommended_actions": (
                    " | ".join(actions)
                ),

                "action_count": len(
                    actions
                ),

                "action_priorities": (
                    " | ".join(priorities)
                ),

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

                "historical_large_migration_count": row[
                    "historical_large_migration_count"
                ],

                "current_version_distance": row[
                    "current_version_distance"
                ],

                "is_first_observed_migration": row[
                    "is_first_observed_migration"
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
        f"\nGenerated decisions: "
        f"{result.height:,}"
    )

    print("\nDecision distribution:")

    print(
        result
        .group_by("decision")
        .len()
        .sort(
            "len",
            descending=True,
        )
    )

    print("\nExample decisions:")

    print(
        result.select(
            [
                "repo_name",
                "evidence_level",
                "decision",
                "recommended_actions",
            ]
        ).head(10)
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print(
        "Migration decision engine completed."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
