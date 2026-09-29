import json
from collections import Counter
from pathlib import Path


DATASET_PATH = Path(
    "data/raw/coupjava/coupjava-coarse.jsonl"
)


def patch_stats(patch: str) -> tuple[int, int]:
    lines = patch.splitlines()

    files_changed = sum(
        line.startswith("diff --git ")
        for line in lines
    )

    changed_lines = sum(
        (
            (line.startswith("+") and not line.startswith("+++"))
            or
            (line.startswith("-") and not line.startswith("---"))
        )
        for line in lines
    )

    return files_changed, changed_lines


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    total = 0
    repositories = set()
    upgrade_paths = Counter()

    total_files = 0
    total_changed_lines = 0

    durations = []

    invalid_records = 0

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

                repo = record["repo_name"]
                old_version = int(record["old_version"])
                new_version = int(record["new_version"])

                old_timestamp = int(record["old_timestamp"])
                new_timestamp = int(record["new_timestamp"])

                patch = record.get("patch", "")

                files_changed, changed_lines = patch_stats(patch)

                duration_days = max(
                    0,
                    new_timestamp - old_timestamp,
                ) / 86400

                total += 1
                repositories.add(repo)

                upgrade_paths[
                    f"Java {old_version} -> Java {new_version}"
                ] += 1

                total_files += files_changed
                total_changed_lines += changed_lines

                durations.append(duration_days)

            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ):
                invalid_records += 1

    print("=" * 70)
    print("MIGRO DATASET PROFILE")
    print("=" * 70)

    print(f"\nMigration cases:     {total:,}")
    print(f"Repositories:        {len(repositories):,}")
    print(f"Invalid records:     {invalid_records:,}")

    if total:
        print(
            f"Avg files changed:   "
            f"{total_files / total:.2f}"
        )

        print(
            f"Avg changed lines:   "
            f"{total_changed_lines / total:.2f}"
        )

        print(
            f"Avg duration:        "
            f"{sum(durations) / len(durations):.2f} days"
        )

    print("\nTop Java upgrade paths:")

    for path, count in upgrade_paths.most_common(15):
        print(
            f"  {path:<25} {count:>6,}"
        )


if __name__ == "__main__":
    main()
