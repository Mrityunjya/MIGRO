from pathlib import Path
from itertools import combinations
from collections import defaultdict
import json

import polars as pl


RAW_PATH = Path(
    "data/raw/coupjava/coupjava-coarse.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/change_coupling_graph.parquet"
)


def extract_changed_files(patch: str) -> list[str]:
    files = []

    if not patch:
        return files

    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue

        parts = line.split()

        if len(parts) < 4:
            continue

        file_path = parts[3]

        if file_path.startswith("a/"):
            file_path = file_path[2:]

        if file_path not in files:
            files.append(file_path)

    return files


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}"
        )

    print("=" * 70)
    print("MIGRO - CHANGE COUPLING GRAPH")
    print("=" * 70)

    edges = defaultdict(int)
    file_frequency = defaultdict(int)
    migration_count = 0

    with RAW_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            files = extract_changed_files(
                record.get("patch", "")
            )

            if not files:
                continue

            migration_count += 1

            for file_path in files:
                file_frequency[file_path] += 1

            # Prevent enormous quadratic graphs from very large
            # repository-wide patches.
            if len(files) > 500:
                continue

            for source, target in combinations(
                sorted(files),
                2,
            ):
                edges[(source, target)] += 1

    print(
        f"\nMigration cases analyzed: {migration_count:,}"
    )

    print(
        f"Unique files observed: {len(file_frequency):,}"
    )

    print(
        f"Unique co-change edges: {len(edges):,}"
    )

    # ---------------------------------------------------------
    # Build edge table
    # ---------------------------------------------------------

    edge_rows = [
        {
            "source_file": source,
            "target_file": target,
            "cochange_count": count,
        }
        for (source, target), count in edges.items()
    ]

    if edge_rows:
        edge_df = (
            pl.DataFrame(edge_rows)
            .sort(
                "cochange_count",
                descending=True,
            )
        )
    else:
        edge_df = pl.DataFrame(
            schema={
                "source_file": pl.String,
                "target_file": pl.String,
                "cochange_count": pl.Int64,
            }
        )

    # ---------------------------------------------------------
    # File-level graph metrics
    # ---------------------------------------------------------

    degree = defaultdict(int)
    weighted_degree = defaultdict(int)

    for (source, target), count in edges.items():
        degree[source] += 1
        degree[target] += 1

        weighted_degree[source] += count
        weighted_degree[target] += count

    node_rows = [
        {
            "file": file_path,
            "migration_frequency": file_frequency[file_path],
            "degree": degree[file_path],
            "weighted_degree": weighted_degree[file_path],
        }
        for file_path in file_frequency
    ]

    node_df = (
        pl.DataFrame(node_rows)
        .sort(
            "weighted_degree",
            descending=True,
        )
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("GRAPH SUMMARY")
    print("-" * 70)

    print(
        f"Nodes: {node_df.height:,}"
    )

    print(
        f"Edges: {edge_df.height:,}"
    )

    if node_df.height > 0:
        print(
            "\nTop migration-connected files:"
        )

        print(
            node_df.head(20)
        )

    if edge_df.height > 0:
        print(
            "\nStrongest co-change relationships:"
        )

        print(
            edge_df.head(20)
        )

    # ---------------------------------------------------------
    # Save edge graph
    # ---------------------------------------------------------

    edge_output = OUTPUT_PATH

    edge_df.write_parquet(
        edge_output,
        compression="zstd",
    )

    # ---------------------------------------------------------
    # Save node metrics
    # ---------------------------------------------------------

    node_output = Path(
        "data/processed/change_coupling_nodes.parquet"
    )

    node_df.write_parquet(
        node_output,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(
        f"Edges: {edge_output}"
    )

    print(
        f"Nodes: {node_output}"
    )


if __name__ == "__main__":
    main()
