from pathlib import Path

import polars as pl


NODES_PATH = Path(
    "data/processed/change_coupling_nodes.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/change_hotspots.parquet"
)


def main() -> None:
    if not NODES_PATH.exists():
        raise FileNotFoundError(
            f"Missing graph nodes: {NODES_PATH}"
        )

    df = pl.read_parquet(NODES_PATH)

    print("=" * 70)
    print("MIGRO - CHANGE COUPLING HOTSPOT ANALYSIS")
    print("=" * 70)

    print(f"\nFiles: {df.height:,}")

    # ---------------------------------------------------------
    # Normalized coupling
    # ---------------------------------------------------------

    df = df.with_columns(
        [
            (
                pl.col("weighted_degree")
                / pl.col("migration_frequency").clip(lower_bound=1)
            ).alias("avg_coupling_per_migration"),

            (
                pl.col("degree")
                / pl.col("migration_frequency").clip(lower_bound=1)
            ).alias("avg_neighbors_per_migration"),
        ]
    )

    # ---------------------------------------------------------
    # Relative centrality
    # ---------------------------------------------------------

    max_weighted = df.select(
        pl.col("weighted_degree").max()
    ).item()

    max_degree = df.select(
        pl.col("degree").max()
    ).item()

    if max_weighted and max_weighted > 0:
        df = df.with_columns(
            (
                pl.col("weighted_degree") / max_weighted
            ).alias("weighted_centrality")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("weighted_centrality")
        )

    if max_degree and max_degree > 0:
        df = df.with_columns(
            (
                pl.col("degree") / max_degree
            ).alias("degree_centrality")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("degree_centrality")
        )

    # ---------------------------------------------------------
    # Hotspot classification
    #
    # This is descriptive segmentation, not a risk score.
    # ---------------------------------------------------------

    df = df.with_columns(
        pl.when(
            (pl.col("weighted_centrality") >= 0.75)
            & (pl.col("migration_frequency") >= 3)
        )
        .then(pl.lit("high_coupling"))
        .when(
            (pl.col("weighted_centrality") >= 0.40)
            & (pl.col("migration_frequency") >= 2)
        )
        .then(pl.lit("moderate_coupling"))
        .otherwise(pl.lit("low_coupling"))
        .alias("coupling_segment")
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("COUPLING SEGMENTS")
    print("-" * 70)

    print(
        df.group_by("coupling_segment")
        .agg(
            [
                pl.len().alias("files"),
                pl.col("migration_frequency")
                .mean()
                .round(2)
                .alias("avg_migration_frequency"),
                pl.col("degree")
                .mean()
                .round(2)
                .alias("avg_degree"),
                pl.col("weighted_degree")
                .mean()
                .round(2)
                .alias("avg_weighted_degree"),
            ]
        )
        .sort("avg_weighted_degree", descending=True)
    )

    # ---------------------------------------------------------
    # Top hotspots
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("TOP MIGRATION HOTSPOTS")
    print("-" * 70)

    print(
        df.select(
            [
                "file",
                "migration_frequency",
                "degree",
                "weighted_degree",
                "avg_coupling_per_migration",
                "coupling_segment",
            ]
        )
        .sort(
            [
                "weighted_degree",
                "migration_frequency",
            ],
            descending=True,
        )
        .head(25)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.write_parquet(
        OUTPUT_PATH,
        compression="zstd",
    )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {df.height:,}")
    print(f"Columns: {df.width}")


if __name__ == "__main__":
    main()
