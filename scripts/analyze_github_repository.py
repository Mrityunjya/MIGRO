from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_CACHE = PROJECT_ROOT / "data" / "github_repos"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


IGNORE_DIRS = {
    ".git",
    ".idea",
    ".gradle",
    "node_modules",
    "target",
    "build",
    "out",
}


def load_metadata(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def collect_files(repo_path: Path) -> list[Path]:
    files = []

    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        files.append(path)

    return files


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return ""


def detect_java_versions(files: list[Path]) -> dict:
    patterns = {
        "maven_source": re.compile(
            r"<maven\.compiler\.source>\s*([^<]+)\s*</maven\.compiler\.source>",
            re.IGNORECASE,
        ),
        "maven_target": re.compile(
            r"<maven\.compiler\.target>\s*([^<]+)\s*</maven\.compiler\.target>",
            re.IGNORECASE,
        ),
        "maven_release": re.compile(
            r"<maven\.compiler\.release>\s*([^<]+)\s*</maven\.compiler\.release>",
            re.IGNORECASE,
        ),
        "java_version_property": re.compile(
            r"<java\.version>\s*([^<]+)\s*</java\.version>",
            re.IGNORECASE,
        ),
        "gradle_source_compatibility": re.compile(
            r"sourceCompatibility\s*=?\s*['\"]?([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE,
        ),
        "gradle_target_compatibility": re.compile(
            r"targetCompatibility\s*=?\s*['\"]?([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE,
        ),
        "gradle_toolchain": re.compile(
            r"JavaLanguageVersion\.of\(\s*([0-9]+)\s*\)",
            re.IGNORECASE,
        ),
    }

    matches: dict[str, list[str]] = {
        key: [] for key in patterns
    }

    interesting_files = {
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "gradle.properties",
    }

    for path in files:
        if path.name not in interesting_files:
            continue

        text = read_text_file(path)

        for name, pattern in patterns.items():
            found = pattern.findall(text)

            for value in found:
                value = value.strip()

                if value not in matches[name]:
                    matches[name].append(value)

    return matches


def extract_dependencies(files: list[Path]) -> dict:
    dependency_names = set()
    dependency_versions = set()

    for path in files:
        if path.name == "pom.xml":
            text = read_text_file(path)

            dependencies = re.findall(
                r"<dependency>\s*"
                r"(?:.*?\n)*?"
                r"\s*<groupId>\s*([^<]+)\s*</groupId>\s*"
                r"(?:.*?\n)*?"
                r"\s*<artifactId>\s*([^<]+)\s*</artifactId>",
                text,
                re.IGNORECASE,
            )

            for group_id, artifact_id in dependencies:
                dependency_names.add(
                    f"{group_id.strip()}:{artifact_id.strip()}"
                )

            versions = re.findall(
                r"<version>\s*([^<]+)\s*</version>",
                text,
                re.IGNORECASE,
            )

            for version in versions:
                dependency_versions.add(version.strip())

        elif path.name in {
            "build.gradle",
            "build.gradle.kts",
        }:
            text = read_text_file(path)

            dependencies = re.findall(
                r"(?:implementation|api|compileOnly|runtimeOnly|testImplementation)"
                r"\s*\(?\s*[\"']([^\"']+)[\"']",
                text,
                re.IGNORECASE,
            )

            for dependency in dependencies:
                dependency_names.add(dependency.strip())

    return {
        "dependency_count": len(dependency_names),
        "dependencies": sorted(dependency_names)[:100],
        "dependency_version_count": len(dependency_versions),
    }


def detect_frameworks(files: list[Path]) -> list[str]:
    frameworks = set()

    for path in files:
        if path.name not in {
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
        }:
            continue

        text = read_text_file(path).lower()

        framework_signals = {
            "Spring Boot": [
                "spring-boot",
                "springframework.boot",
            ],
            "Spring Framework": [
                "spring-core",
                "spring-context",
                "springframework",
            ],
            "Quarkus": [
                "quarkus",
            ],
            "Micronaut": [
                "micronaut",
            ],
            "Jakarta EE": [
                "jakarta.",
                "jakarta-",
            ],
            "JavaFX": [
                "javafx",
            ],
            "JUnit": [
                "junit",
            ],
            "TestNG": [
                "testng",
            ],
        }

        for framework, signals in framework_signals.items():
            if any(signal in text for signal in signals):
                frameworks.add(framework)

    return sorted(frameworks)


def detect_migration_signals(files: list[Path]) -> dict:
    signal_patterns = {
        "compiler_configuration": [
            "maven.compiler",
            "sourceCompatibility",
            "targetCompatibility",
            "JavaLanguageVersion",
        ],
        "java_version_configuration": [
            "java.version",
            "javaVersion",
            "JAVA_VERSION",
        ],
        "module_system": [
            "module-info.java",
        ],
        "jakarta_migration": [
            "jakarta.",
            "jakarta-",
        ],
        "javax_usage": [
            "javax.",
        ],
        "build_plugin_configuration": [
            "maven-compiler-plugin",
            "maven-surefire-plugin",
            "maven-failsafe-plugin",
            "toolchain",
        ],
    }

    counts = {
        key: 0
        for key in signal_patterns
    }

    for path in files:
        if path.suffix.lower() not in {
            ".java",
            ".xml",
            ".gradle",
            ".kts",
            ".properties",
        }:
            continue

        text = read_text_file(path)

        for signal, patterns in signal_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    counts[signal] += 1

    return counts


def analyze_repository(repo_path: Path) -> dict:
    files = collect_files(repo_path)

    java_files = [
        path for path in files
        if path.suffix.lower() == ".java"
    ]

    test_files = [
        path for path in files
        if (
            path.suffix.lower() == ".java"
            and (
                "test" in {part.lower() for part in path.parts}
                or path.name.lower().endswith("test.java")
                or path.name.lower().endswith("tests.java")
            )
        )
    ]

    pom_files = [
        path for path in files
        if path.name == "pom.xml"
    ]

    gradle_files = [
        path for path in files
        if path.name in {
            "build.gradle",
            "build.gradle.kts",
        }
    ]

    versions = detect_java_versions(files)
    dependencies = extract_dependencies(files)
    frameworks = detect_frameworks(files)
    migration_signals = detect_migration_signals(files)

    return {
        "repository_path": str(repo_path),
        "file_count": len(files),
        "java_file_count": len(java_files),
        "test_java_file_count": len(test_files),
        "pom_file_count": len(pom_files),
        "gradle_file_count": len(gradle_files),
        "build_systems": {
            "maven": bool(pom_files),
            "gradle": bool(gradle_files),
        },
        "java_version_signals": versions,
        "dependency_analysis": dependencies,
        "frameworks_detected": frameworks,
        "migration_signals": migration_signals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a cloned GitHub repository for MIGRO."
    )

    parser.add_argument(
        "repository",
        help="Repository name in owner__repo format",
    )

    args = parser.parse_args()

    repo_name = args.repository
    repo_path = REPO_CACHE / repo_name

    if not repo_path.exists():
        raise FileNotFoundError(
            f"Repository not found: {repo_path}"
        )

    metadata_path = METADATA_DIR / f"{repo_name}.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Repository metadata not found: {metadata_path}"
        )

    metadata = load_metadata(metadata_path)
    analysis = analyze_repository(repo_path)

    result = {
        **metadata,
        "repository_analysis": analysis,
    }

    output_path = (
        METADATA_DIR
        / f"{repo_name}__analysis.json"
    )

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nMIGRO repository analysis complete.")
    print("-----------------------------------")
    print(f"Repository:       {metadata.get('repository')}")
    print(f"Files:            {analysis['file_count']}")
    print(f"Java files:       {analysis['java_file_count']}")
    print(f"Test files:       {analysis['test_java_file_count']}")
    print(f"Maven:            {analysis['build_systems']['maven']}")
    print(f"Gradle:           {analysis['build_systems']['gradle']}")
    print(f"Dependencies:     {analysis['dependency_analysis']['dependency_count']}")
    print(
        "Frameworks:       "
        + (
            ", ".join(analysis["frameworks_detected"])
            or "None detected"
        )
    )
    print(f"Output:           {output_path}")


if __name__ == "__main__":
    main()
