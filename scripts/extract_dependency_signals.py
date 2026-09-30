from pathlib import Path
from typing import Any

import polars as pl


INPUT_PATH = Path(
    "data/processed/migration_signals.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/dependency_signals.parquet"
)


def analyze_patch(patch: str) -> dict[str, Any]:
    if not patch:
        return {
            "build_file_changed": False,
            "pom_changed": False,
            "gradle_changed": False,
            "dependency_lines_added": 0,
            "dependency_lines_removed": 0,
            "version_change_lines": 0,
            "java_files_changed": 0,
            "import_change_lines": 0,
        }

    lines = patch.splitlines()

    build_file_changed = False
    pom_changed = False
    gradle_changed = False

    dependency_lines_added = 0
    dependency_lines_removed = 0
    version_change_lines = 0

    java_files_changed = 0
    import_change_lines = 0

    current_file = ""

    for line in lines:

        # -----------------------------------------------------
        # Track changed file
        # -----------------------------------------------------

        if line.startswith("diff --git "):
            parts = line.split()

            if len(parts) >= 4:
                current_file = parts[3]

                if current_file.startswith("a/"):
                    current_file = current_file[2:]

                lower_file = current_file.lower()

                if lower_file.endswith(".java"):
                    java_files_changed += 1

                if (
                    lower_file.endswith("pom.xml")
                    or lower_file.endswith("build.gradle")
                    or lower_file.endswith("build.gradle.kts")
                    or lower_file.endswith("settings.gradle")
                    or lower_file.endswith("settings.gradle.kts")
                ):
                    build_file_changed = True

                if lower_file.endswith("pom.xml"):
                    pom_changed = True

                if (
                    lower_file.endswith("build.gradle")
                    or lower_file.endswith("build.gradle.kts")
                    or lower_file.endswith("settings.gradle")
                    or lower_file.endswith("settings.gradle.kts")
                ):
                    gradle_changed = True

            continue

        # -----------------------------------------------------
        # Ignore diff metadata
        # -----------------------------------------------------

        if line.startswith("+++") or line.startswith("---"):
            continue

        is_addition = line.startswith("+")
        is_deletion = line.startswith("-")

        if not (is_addition or is_deletion):
            continue

        content = line[1:].strip().lower()

        # -----------------------------------------------------
        # Dependency-related changes
        # -----------------------------------------------------

        dependency_keywords = [
            "<dependency",
            "<groupid>",
            "<artifactid>",
            "<version>",
            "implementation ",
            "api ",
            "compile ",
            "runtimeonly ",
            "testimplementation ",
            "classpath ",
            "dependency ",
        ]

        if any(
            keyword in content
            for keyword in dependency_keywords
        ):
            if is_addition:
                dependency_lines_added += 1
            else:
                dependency_lines_removed += 1

        # -----------------------------------------------------
        # Version changes
        # -----------------------------------------------------

        if (
            "<version>" in content
            or "version " in content
            or "version=" in content
        ):
            version_change_lines += 1

        # -----------------------------------------------------
        # Java imports
        # -----------------------------------------------------

        if content.startswith("import "):
            import_change_lines += 1

    return {
        "build_file_changed": build_file_changed,
        "pom_changed": pom_changed,
        "gradle_changed": gradle_changed,
        "dependency_lines_added": dependency_lines_added,
        "dependency_lines_removed": dependency_lines_removed,
        "version_change_lines": version_change_lines,
        "java_files_changed": java_files_changed,
        "import_change_lines": import_change_lines,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    print("=" * 70)
    print("MIGRO - DEPENDENCY & BUILD SIGNAL EXTRACTION")
    print("=" * 70)

    print(f"\nRows: {df.height:,}")

    # ---------------------------------------------------------
    # Reconstruct patch information
    # ---------------------------------------------------------
    #
    # migration_signals.parquet currently does not contain the
    # original patch. We therefore load it from the raw JSONL
    # and join the extracted signals by commit pair.
    # ---------------------------------------------------------

    raw_path = Path(
        "data/raw/coupjava/coupjava-coarse.jsonl"
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path}"
        )

    print("\nReading migration patches...")

    import json

    records = []

    with raw_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            patch_signals = analyze_patch(
                record.get("patch", "")
            )

            records.append(
                {
                    "old_commit": record["old_commit"],
                    "new_commit": record["new_commit"],
                    **patch_signals,
                }
            )

    patch_df = pl.DataFrame(records)

    print(
        f"Patch records analyzed: {patch_df.height:,}"
    )

    # ---------------------------------------------------------
    # Join structural signals
    # ---------------------------------------------------------

    result = df.join(
        patch_df,
        on=["old_commit", "new_commit"],
        how="left",
    )

    # ---------------------------------------------------------
    # Derived dependency indicators
    # ---------------------------------------------------------

    result = result.with_columns(
        [
            (
                pl.col("dependency_lines_added")
                + pl.col("dependency_lines_removed")
            ).alias("dependency_change_volume"),

            (
                pl.col("dependency_lines_added")
                > 0
            ).alias("dependencies_added"),

            (
                pl.col("dependency_lines_removed")
                > 0
            ).alias("dependencies_removed"),

            (
                pl.col("version_change_lines")
                > 0
            ).alias("dependency_version_changed"),

            (
                pl.col("java_files_changed")
                > 0
            ).alias("java_source_changed"),

            (
                pl.col("import_change_lines")
                > 0
            ).alias("imports_changed"),
        ]
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("DEPENDENCY SIGNAL SUMMARY")
    print("-" * 70)

    summary = result.select(
        [
            pl.len().alias("migration_cases"),

            pl.col("build_file_changed")
            .sum()
            .alias("build_file_cases"),

            pl.col("pom_changed")
            .sum()
            .alias("pom_cases"),

            pl.col("gradle_changed")
            .sum()
            .alias("gradle_cases"),

            pl.col("dependencies_added")
            .sum()
            .alias("dependency_addition_cases"),

            pl.col("dependencies_removed")
            .sum()
            .alias("dependency_removal_cases"),

            pl.col("dependency_version_changed")
            .sum()
            .alias("dependency_version_change_cases"),

            pl.col("java_source_changed")
            .sum()
            .alias("java_source_cases"),

            pl.col("imports_changed")
            .sum()
            .alias("import_change_cases"),
        ]
    )

    print(summary)

    # ---------------------------------------------------------
    # Highest dependency-change migrations
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("HIGHEST DEPENDENCY-CHANGE MIGRATIONS")
    print("-" * 70)

    print(
        result.select(
            [
                "repo_name",
                "old_version",
                "new_version",
                "dependency_change_volume",
                "dependency_lines_added",
                "dependency_lines_removed",
                "version_change_lines",
                "files_changed",
                "changed_lines",
            ]
        )
        .sort(
            "dependency_change_volume",
            descending=True,
        )
        .head(20)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    result.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {result.height:,}")
    print(f"Columns: {result.width}")


if __name__ == "__main__":
    main()
