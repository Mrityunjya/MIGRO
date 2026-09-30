from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "historical_repo_baselines.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "risk_model_evaluation.parquet"
)


HISTORICAL_FEATURES = [
    "historical_migration_count",
    "historical_median_duration_days",
    "historical_median_files_changed",
    "historical_median_changed_lines",
    "historical_median_version_distance",
    "historical_max_duration_days",
    "historical_large_migration_count",
    "is_first_observed_migration",
    "version_distance",
]


def evaluate_predictions(
    name,
    y_true,
    probabilities,
):
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    result = {
        "model": name,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier_score": np.nan,
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }

    if len(np.unique(y_true)) >= 2:

        result["roc_auc"] = roc_auc_score(
            y_true,
            probabilities,
        )

        result["pr_auc"] = average_precision_score(
            y_true,
            probabilities,
        )

    result["brier_score"] = brier_score_loss(
        y_true,
        probabilities,
    )

    return result


def main():

    print("=" * 70)
    print("MIGRO - Risk Model Evaluation")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    required = HISTORICAL_FEATURES + [
        "actual_duration_days",
        "migration_timestamp",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ------------------------------------------------------------
    # 1. Timestamp preparation
    # ------------------------------------------------------------

    df = df.with_columns(
        pl.col("migration_timestamp")
        .cast(pl.Utf8)
        .str.to_datetime(strict=False)
        .alias("migration_timestamp")
    )

    df = (
        df
        .filter(
            pl.col("migration_timestamp").is_not_null()
        )
        .sort("migration_timestamp")
    )

    # ------------------------------------------------------------
    # 2. Target
    # ------------------------------------------------------------

    duration_p75 = (
        df.select(
            pl.col("actual_duration_days")
            .quantile(0.75)
        )
        .item()
    )

    df = df.with_columns(
        (
            pl.col("actual_duration_days")
            >= duration_p75
        )
        .cast(pl.Int8)
        .alias("high_duration")
    )

    # Convert boolean to numeric before pandas/sklearn.

    if "is_first_observed_migration" in df.columns:

        df = df.with_columns(
            pl.col("is_first_observed_migration")
            .cast(pl.Int8)
            .alias("is_first_observed_migration")
        )

    pdf = df.select(
        HISTORICAL_FEATURES
        + [
            "high_duration",
            "migration_timestamp",
        ]
    ).to_pandas()

    for column in HISTORICAL_FEATURES:

        pdf[column] = pd.to_numeric(
            pdf[column],
            errors="coerce",
        )

    # ------------------------------------------------------------
    # 3. Chronological split
    # ------------------------------------------------------------

    split_index = int(
        len(pdf) * 0.75
    )

    train = pdf.iloc[:split_index].copy()
    test = pdf.iloc[split_index:].copy()

    X_train = train[
        HISTORICAL_FEATURES
    ]

    y_train = train[
        "high_duration"
    ]

    X_test = test[
        HISTORICAL_FEATURES
    ]

    y_test = test[
        "high_duration"
    ]

    print(
        f"\nTrain rows: {len(train):,}"
    )

    print(
        f"Test rows:  {len(test):,}"
    )

    print(
        f"\nHigh-duration threshold: "
        f"{duration_p75:.2f} days"
    )

    # ------------------------------------------------------------
    # 4. Majority-class baseline
    # ------------------------------------------------------------

    train_positive_rate = (
        y_train.mean()
    )

    majority_class = int(
        y_train.mean() >= 0.5
    )

    majority_probability = np.full(
        len(y_test),
        train_positive_rate,
    )

    majority_predictions = np.full(
        len(y_test),
        majority_class,
    )

    # ------------------------------------------------------------
    # 5. Version-distance baseline
    # ------------------------------------------------------------

    version_feature = [
        "version_distance"
    ]

    version_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler()
                        ),
                    ]
                ),
                version_feature,
            )
        ]
    )

    version_model = Pipeline(
        steps=[
            (
                "preprocessor",
                version_preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    version_model.fit(
        X_train[version_feature],
        y_train,
    )

    version_probability = (
        version_model.predict_proba(
            X_test[version_feature]
        )[:, 1]
    )

    # ------------------------------------------------------------
    # 6. Historical-context model
    # ------------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler()
                        ),
                    ]
                ),
                HISTORICAL_FEATURES,
            )
        ]
    )

    historical_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    historical_model.fit(
        X_train,
        y_train,
    )

    historical_probability = (
        historical_model.predict_proba(
            X_test
        )[:, 1]
    )

    # ------------------------------------------------------------
    # 7. Evaluate
    # ------------------------------------------------------------

    results = []

    results.append(
        evaluate_predictions(
            "majority_baseline",
            y_test,
            majority_probability,
        )
    )

    results.append(
        evaluate_predictions(
            "version_distance_baseline",
            y_test,
            version_probability,
        )
    )

    results.append(
        evaluate_predictions(
            "historical_context_model",
            y_test,
            historical_probability,
        )
    )

    results_df = pl.DataFrame(
        results
    )

    # ------------------------------------------------------------
    # 8. Print comparison
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df
    )

    # ------------------------------------------------------------
    # 9. Historical model classification report
    # ------------------------------------------------------------

    historical_predictions = (
        historical_probability >= 0.5
    ).astype(int)

    print("\nHistorical-context model:")
    print(
        classification_report(
            y_test,
            historical_predictions,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.write_parquet(
        OUTPUT_PATH
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print("Risk model evaluation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
