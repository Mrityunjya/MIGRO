import json
import re
from pathlib import Path

import polars as pl


RAW = Path("data/raw/coupjava/coupjava-coarse.jsonl")
OUTPUT = Path("data/processed/pre_migration_features.parquet")


def version_number(value):
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def count_dependency_hints(patch):
    if not patch:
        return 0

    lines = patch.splitlines()

    patterns = [
        r"<dependency>",
        r"<version>",
        r"<groupId>",
        r"<artifactId>",
        r"implementation\s+['\"]",
        r"api\s+['\"]",
        r"compile\s+['\"]",
        r"testImplementation\s+['\"]",
    ]

    return sum(
        1
        for line in lines
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)
    )


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW}")

    rows = []

    with RAW.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            old_version = version_number(record.get("old_version"))
            new_version = version_number(record.get("new_version"))

            if old_version is None or new_version is None:
                continue

            version_distance = new_version - old_version

            patch = record.get("patch") or ""

            # Signals available from the migration specification itself.
            migration_type = (
                "major_gap"
                if version_distance >= 4
                else "moderate_gap"
                if version_distance >= 2
                else "adjacent_gap"
            )

            rows.append(
                {
                    "repo_name": record.get("repo_name"),
                    "old_commit": record.get("old_commit"),
                    "new_commit": record.get("new_commit"),
                    "old_version": old_version,
                    "new_version": new_version,
                    "version_distance": version_distance,
                    "migration_type": migration_type,

                    # Specification/build-system indicators.
                    "pom_expected": "pom.xml" in patch,
                    "gradle_expected": (
                        "build.gradle" in patch
                        or "build.gradle.kts" in patch
                    ),

                    # Lightweight dependency-context signal.
                    "dependency_hints": count_dependency_hints(patch),

                    # Presence of common Java migration touchpoints.
                    "java_file_hints": len(
                        re.findall(
                            r"diff --git .*\.java",
                            patch,
                            re.IGNORECASE,
                        )
                    ),

                    "configuration_file_hints": sum(
                        1
                        for filename in [
                            "pom.xml",
                            "build.gradle",
                            "build.gradle.kts",
                            "gradle.properties",
                            "settings.gradle",
                            "settings.gradle.kts",
                        ]
                        if filename in patch
                    ),
                }
            )

    if not rows:
        raise ValueError("No valid migration records found.")

    df = pl.DataFrame(rows)

    # Derived pre-migration feature groups.
    df = df.with_columns(
        [
            pl.col("version_distance")
            .abs()
            .alias("absolute_version_distance"),

            (
                pl.col("dependency_hints") > 0
            ).alias("dependency_context_present"),

            (
                pl.col("java_file_hints") > 0
            ).alias("java_code_context_present"),

            (
                pl.col("configuration_file_hints") > 0
            ).alias("build_configuration_context_present"),
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT)

    print("Pre-migration feature dataset created.")
    print(f"Rows: {df.height}")
    print(f"Columns: {df.width}")
    print(f"Output: {OUTPUT}")
    print()
    print(df.head(10))


if __name__ == "__main__":
    main()
