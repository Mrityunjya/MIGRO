from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MigrationSignal:
    repository: str
    source_version: int
    target_version: int

    source_commit: str
    target_commit: str

    migration_duration_days: float
    patch_lines: int
    files_changed: int

    version_distance: int
    major_upgrade: bool


def extract_patch_stats(patch: str) -> tuple[int, int]:
    """Extract basic file and line-change statistics from a Git patch."""

    if not patch:
        return 0, 0

    lines = patch.splitlines()

    files_changed = sum(
        1
        for line in lines
        if line.startswith("diff --git ")
    )

    changed_lines = sum(
        1
        for line in lines
        if (
            (line.startswith("+") and not line.startswith("+++"))
            or
            (line.startswith("-") and not line.startswith("---"))
        )
    )

    return files_changed, changed_lines


def extract_signal(record: dict[str, Any]) -> MigrationSignal:
    """Convert a raw CoUpJava record into MIGRO migration signals."""

    old_timestamp = int(record["old_timestamp"])
    new_timestamp = int(record["new_timestamp"])

    duration_days = max(
        0,
        new_timestamp - old_timestamp,
    ) / 86400

    files_changed, patch_lines = extract_patch_stats(
        record.get("patch", "")
    )

    source_version = int(record["old_version"])
    target_version = int(record["new_version"])

    return MigrationSignal(
        repository=record["repo_name"],
        source_version=source_version,
        target_version=target_version,
        source_commit=record["old_commit"],
        target_commit=record["new_commit"],
        migration_duration_days=duration_days,
        patch_lines=patch_lines,
        files_changed=files_changed,
        version_distance=target_version - source_version,
        major_upgrade=(
            target_version > source_version
        ),
    )
