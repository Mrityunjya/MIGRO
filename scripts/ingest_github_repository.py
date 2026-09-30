from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_CACHE = PROJECT_ROOT / "data" / "github_repos"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def validate_github_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Repository URL must start with http:// or https://")

    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError("Only GitHub repository URLs are supported.")

    parts = [part for part in parsed.path.strip("/").split("/") if part]

    if len(parts) < 2:
        raise ValueError("URL must look like https://github.com/owner/repository")

    owner = parts[0]
    repo = parts[1]

    if repo.endswith(".git"):
        repo = repo[:-4]

    return owner, repo


def clone_repository(url: str, owner: str, repo: str) -> Path:
    REPO_CACHE.mkdir(parents=True, exist_ok=True)

    repo_path = REPO_CACHE / f"{owner}__{repo}"

    if repo_path.exists():
        print(f"Repository already exists: {repo_path}")
        return repo_path

    print(f"Cloning {url} ...")

    result = subprocess.run(
        ["git", "clone", "--depth", "50", url, str(repo_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)

        raise RuntimeError(
            "Git clone failed:\n"
            + (result.stderr.strip() or result.stdout.strip())
        )

    return repo_path


def detect_project(repo_path: Path) -> dict:
    files = [p for p in repo_path.rglob("*") if p.is_file()]

    java_files = [
        p for p in files
        if p.suffix.lower() == ".java"
    ]

    pom_files = [
        p for p in files
        if p.name == "pom.xml"
    ]

    gradle_files = [
        p for p in files
        if p.name in {
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
        }
    ]

    return {
        "total_files": len(files),
        "java_files": len(java_files),
        "pom_files": len(pom_files),
        "gradle_files": len(gradle_files),
        "is_java_repository": len(java_files) > 0,
        "uses_maven": len(pom_files) > 0,
        "uses_gradle": len(gradle_files) > 0,
    }


def get_git_metadata(repo_path: Path) -> dict:
    def git_command(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return ""

        return result.stdout.strip()

    commit_count = git_command("rev-list", "--count", "HEAD")

    branch = git_command(
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )

    latest_commit = git_command(
        "rev-parse",
        "HEAD",
    )

    return {
        "commit_count": int(commit_count) if commit_count.isdigit() else None,
        "default_branch": branch or None,
        "latest_commit": latest_commit or None,
    }


def build_metadata(url: str, owner: str, repo: str, repo_path: Path) -> dict:
    project = detect_project(repo_path)
    git = get_git_metadata(repo_path)

    return {
        "repository_url": url,
        "owner": owner,
        "repository": repo,
        "local_path": str(repo_path),
        **project,
        **git,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a public GitHub repository for MIGRO analysis."
    )

    parser.add_argument(
        "url",
        help="Public GitHub repository URL",
    )

    args = parser.parse_args()

    owner, repo = validate_github_url(args.url)

    repo_path = clone_repository(
        args.url,
        owner,
        repo,
    )

    metadata = build_metadata(
        args.url,
        owner,
        repo,
        repo_path,
    )

    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        METADATA_DIR
        / f"{owner}__{repo}.json"
    )

    output_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nMIGRO repository ingestion complete.")
    print("-----------------------------------")
    print(f"Repository:       {owner}/{repo}")
    print(f"Files:            {metadata['total_files']}")
    print(f"Java files:       {metadata['java_files']}")
    print(f"Maven:            {metadata['uses_maven']}")
    print(f"Gradle:           {metadata['uses_gradle']}")
    print(f"Commits:          {metadata['commit_count']}")
    print(f"Latest commit:    {metadata['latest_commit']}")
    print(f"Metadata:         {output_path}")


if __name__ == "__main__":
    main()