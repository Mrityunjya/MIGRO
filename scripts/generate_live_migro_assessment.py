from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_java_version(value: str | None) -> int | None:
    if not value:
        return None

    value = str(value).strip()

    # Java 8 style
    if value.startswith("1."):
        value = value[2:]

    match = re.search(r"\d+", value)

    if not match:
        return None

    try:
        return int(match.group())
    except ValueError:
        return None


def infer_current_java_version(signals: dict) -> tuple[int | None, str | None]:
    priority = [
        "maven_release",
        "maven_source",
        "maven_target",
        "java_version_property",
        "gradle_toolchain",
        "gradle_source_compatibility",
        "gradle_target_compatibility",
    ]

    for key in priority:
        values = signals.get(key, [])

        if not values:
            continue

        version = parse_java_version(values[0])

        if version is not None:
            return version, key

    return None, None


def add_evidence(
    evidence: list[dict],
    dimension: str,
    level: str,
    observation: str,
) -> None:
    evidence.append(
        {
            "dimension": dimension,
            "level": level,
            "observation": observation,
        }
    )


def build_assessment(
    data: dict,
    target_java: int | None,
) -> dict:

    analysis = data["repository_analysis"]

    java_files = int(analysis.get("java_file_count", 0))
    test_files = int(analysis.get("test_java_file_count", 0))
    dependencies = int(
        analysis.get("dependency_analysis", {})
        .get("dependency_count", 0)
    )

    frameworks = analysis.get("frameworks_detected", [])
    build_systems = analysis.get("build_systems", {})
    migration_signals = analysis.get("migration_signals", {})
    version_signals = analysis.get("java_version_signals", {})

    current_java, version_source = infer_current_java_version(
        version_signals
    )

    evidence = []
    guidance = []

    # ---------------------------------------------------------
    # Repository scale
    # ---------------------------------------------------------

    if java_files >= 1000:
        add_evidence(
            evidence,
            "source_surface",
            "high",
            f"{java_files} Java source/test files were detected.",
        )
        guidance.append(
            "Plan repository-wide compatibility and regression validation."
        )

    elif java_files >= 250:
        add_evidence(
            evidence,
            "source_surface",
            "moderate",
            f"{java_files} Java files were detected.",
        )
        guidance.append(
            "Review compatibility across the main Java source surface."
        )

    elif java_files > 0:
        add_evidence(
            evidence,
            "source_surface",
            "limited",
            f"{java_files} Java files were detected.",
        )

    # ---------------------------------------------------------
    # Dependency surface
    # ---------------------------------------------------------

    if dependencies >= 100:
        add_evidence(
            evidence,
            "dependency_surface",
            "high",
            f"{dependencies} dependencies were detected.",
        )
        guidance.append(
            "Perform dependency compatibility analysis before changing Java versions."
        )

    elif dependencies >= 30:
        add_evidence(
            evidence,
            "dependency_surface",
            "moderate",
            f"{dependencies} dependencies were detected.",
        )
        guidance.append(
            "Validate dependency and plugin compatibility with the target Java version."
        )

    elif dependencies > 0:
        add_evidence(
            evidence,
            "dependency_surface",
            "limited",
            f"{dependencies} dependencies were detected.",
        )

    # ---------------------------------------------------------
    # Build system
    # ---------------------------------------------------------

    if build_systems.get("maven") and build_systems.get("gradle"):
        add_evidence(
            evidence,
            "build_complexity",
            "high",
            "Both Maven and Gradle configuration were detected.",
        )
        guidance.append(
            "Validate both build paths independently during migration."
        )

    elif build_systems.get("maven"):
        add_evidence(
            evidence,
            "build_system",
            "moderate",
            "Maven configuration was detected.",
        )

    elif build_systems.get("gradle"):
        add_evidence(
            evidence,
            "build_system",
            "moderate",
            "Gradle configuration was detected.",
        )

    else:
        add_evidence(
            evidence,
            "build_system",
            "limited",
            "No standard Maven or Gradle build configuration was detected.",
        )

    # ---------------------------------------------------------
    # Framework surface
    # ---------------------------------------------------------

    if len(frameworks) >= 3:
        add_evidence(
            evidence,
            "framework_surface",
            "high",
            f"{len(frameworks)} framework/test ecosystem signals were detected.",
        )
        guidance.append(
            "Review framework compatibility and integration tests before migration."
        )

    elif frameworks:
        add_evidence(
            evidence,
            "framework_surface",
            "moderate",
            "Detected: " + ", ".join(frameworks),
        )

    # ---------------------------------------------------------
    # Compiler configuration
    # ---------------------------------------------------------

    compiler_hits = migration_signals.get(
        "compiler_configuration",
        0,
    )

    if compiler_hits > 0:
        add_evidence(
            evidence,
            "compiler_configuration",
            "moderate",
            f"Java compiler/toolchain configuration was detected in {compiler_hits} file(s).",
        )
        guidance.append(
            "Review compiler source, target, release, and toolchain settings."
        )

    # ---------------------------------------------------------
    # javax / jakarta
    # ---------------------------------------------------------

    javax_hits = migration_signals.get(
        "javax_usage",
        0,
    )

    jakarta_hits = migration_signals.get(
        "jakarta_migration",
        0,
    )

    if javax_hits > 0:
        add_evidence(
            evidence,
            "javax_surface",
            "high",
            f"javax usage was detected in {javax_hits} file(s).",
        )
        guidance.append(
            "Inspect javax compatibility carefully, especially when moving into Jakarta-based frameworks."
        )

    if jakarta_hits > 0:
        add_evidence(
            evidence,
            "jakarta_surface",
            "moderate",
            f"Jakarta-related usage was detected in {jakarta_hits} file(s).",
        )

    # ---------------------------------------------------------
    # Module system
    # ---------------------------------------------------------

    module_hits = migration_signals.get(
        "module_system",
        0,
    )

    if module_hits > 0:
        add_evidence(
            evidence,
            "module_system",
            "moderate",
            f"module-info.java was detected in {module_hits} file(s).",
        )
        guidance.append(
            "Include JPMS/module compatibility in migration validation."
        )

    # ---------------------------------------------------------
    # Current ? target Java version
    # ---------------------------------------------------------

    version_distance = None

    if current_java is not None and target_java is not None:
        version_distance = target_java - current_java

        if version_distance <= 0:
            add_evidence(
                evidence,
                "java_version_distance",
                "limited",
                f"Detected Java {current_java}; requested target is Java {target_java}.",
            )

        elif version_distance >= 4:
            add_evidence(
                evidence,
                "java_version_distance",
                "high",
                f"Java {current_java} ? {target_java} represents a version distance of {version_distance}.",
            )
            guidance.append(
                "Use staged compatibility validation because the Java version distance is substantial."
            )

        elif version_distance >= 2:
            add_evidence(
                evidence,
                "java_version_distance",
                "moderate",
                f"Java {current_java} ? {target_java} represents a version distance of {version_distance}.",
            )
            guidance.append(
                "Validate language, compiler, dependency, and runtime compatibility."
            )

        else:
            add_evidence(
                evidence,
                "java_version_distance",
                "limited",
                f"Java {current_java} ? {target_java} represents a version distance of {version_distance}.",
            )

    elif current_java is not None:
        add_evidence(
            evidence,
            "java_version",
            "moderate",
            f"Current Java version appears to be {current_java}, inferred from {version_source}.",
        )

    else:
        add_evidence(
            evidence,
            "java_version",
            "limited",
            "A reliable Java version declaration was not detected.",
        )
        guidance.append(
            "Establish the current Java runtime/compiler version before migration planning."
        )

    # ---------------------------------------------------------
    # Test surface
    # ---------------------------------------------------------

    if test_files >= 250:
        add_evidence(
            evidence,
            "test_surface",
            "high",
            f"{test_files} Java test files were detected.",
        )
        guidance.append(
            "Use the existing test suite as the primary regression validation layer."
        )

    elif test_files >= 50:
        add_evidence(
            evidence,
            "test_surface",
            "moderate",
            f"{test_files} Java test files were detected.",
        )

    elif test_files > 0:
        add_evidence(
            evidence,
            "test_surface",
            "limited",
            f"{test_files} Java test files were detected.",
        )

    else:
        add_evidence(
            evidence,
            "test_surface",
            "limited",
            "No Java test files were detected using the current heuristic.",
        )
        guidance.append(
            "Establish migration regression coverage before making broad compatibility changes."
        )

    # ---------------------------------------------------------
    # Evidence strength
    # ---------------------------------------------------------

    level_weights = {
        "high": 3,
        "moderate": 2,
        "limited": 1,
    }

    evidence_score = sum(
        level_weights.get(item["level"], 0)
        for item in evidence
    )

    high_count = sum(
        item["level"] == "high"
        for item in evidence
    )

    if high_count >= 3 or evidence_score >= 12:
        evidence_strength = "strong"
    elif high_count >= 1 or evidence_score >= 7:
        evidence_strength = "moderate"
    else:
        evidence_strength = "limited"

    # ---------------------------------------------------------
    # Overall profile
    # ---------------------------------------------------------

    if high_count >= 3:
        profile = "high migration complexity signals"

    elif high_count >= 1 or evidence_score >= 7:
        profile = "moderate migration complexity signals"

    else:
        profile = "limited migration complexity signals"

    return {
        "assessment_type": "live_repository_migration_assessment",
        "repository": data.get("repository"),
        "repository_url": data.get("repository_url"),
        "current_java_version": current_java,
        "current_java_version_source": version_source,
        "target_java_version": target_java,
        "version_distance": version_distance,
        "profile": profile,
        "evidence_strength": evidence_strength,
        "evidence_score": evidence_score,
        "evidence_dimensions": evidence,
        "engineering_guidance": list(dict.fromkeys(guidance)),
        "frameworks": frameworks,
        "dependency_count": dependencies,
        "java_file_count": java_files,
        "test_java_file_count": test_files,
        "build_systems": build_systems,
        "methodology_note": (
            "This is an evidence-based repository assessment. "
            "It is not a guarantee of migration success or a causal prediction."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a live MIGRO assessment for a GitHub repository."
    )

    parser.add_argument(
        "repository",
        help="Repository name in owner__repo format",
    )

    parser.add_argument(
        "--target-java",
        type=int,
        default=None,
        help="Optional target Java version, for example 21",
    )

    args = parser.parse_args()

    analysis_path = (
        METADATA_DIR
        / f"{args.repository}__analysis.json"
    )

    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Repository analysis not found: {analysis_path}"
        )

    data = load_json(analysis_path)

    assessment = build_assessment(
        data,
        args.target_java,
    )

    output_path = (
        METADATA_DIR
        / f"{args.repository}__assessment.json"
    )

    output_path.write_text(
        json.dumps(assessment, indent=2),
        encoding="utf-8",
    )

    print("\nMIGRO live assessment complete.")
    print("--------------------------------")
    print(f"Repository:       {assessment['repository']}")
    print(f"Current Java:     {assessment['current_java_version']}")
    print(f"Target Java:      {assessment['target_java_version']}")
    print(f"Version distance: {assessment['version_distance']}")
    print(f"Profile:          {assessment['profile']}")
    print(f"Evidence:         {assessment['evidence_strength']}")
    print(f"Evidence score:   {assessment['evidence_score']}")
    print(
        f"Evidence items:   {len(assessment['evidence_dimensions'])}"
    )
    print(f"Guidance items:   {len(assessment['engineering_guidance'])}")
    print(f"Output:           {output_path}")


if __name__ == "__main__":
    main()
