from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "historical_repo_baselines.parquet"
)

RESULTS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "migration_risk_model_results.parquet"
)

COEFFICIENTS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "migration_risk_coefficients.parquet"
)


FEATURES = [
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


def main():

    print("=" * 70)
    print("MIGRO - Leakage-Safe Migration Risk Model")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    print(f"\nLoaded rows: {df.height:,}")
    print(f"Columns: {df.width}")

    required_columns = FEATURES + [
        "actual_duration_days",
        "migration_timestamp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ------------------------------------------------------------
    # 1. Parse timestamps and sort globally by time
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

    print(
        f"Rows with valid timestamps: {df.height:,}"
    )

    if df.height < 10:
        raise ValueError(
            "Too few timestamped migrations for model training."
        )

    # ------------------------------------------------------------
    # 2. Build high-duration target
    # ------------------------------------------------------------

    duration_p75 = (
        df.select(
            pl.col("actual_duration_days")
            .quantile(0.75)
        )
        .item()
    )

    print(
        f"\nHigh-duration threshold (P75): "
        f"{duration_p75:.2f} days"
    )

    df = df.with_columns(
        (
            pl.col("actual_duration_days")
            >= duration_p75
        )
        .cast(pl.Int8)
        .alias("high_duration")
    )

    print("\nTarget distribution:")

    print(
        df.group_by("high_duration")
        .len()
        .sort("high_duration")
    )

    # ------------------------------------------------------------
    # 3. Convert boolean feature to numeric
    # ------------------------------------------------------------

    if "is_first_observed_migration" in df.columns:

        df = df.with_columns(
            pl.col("is_first_observed_migration")
            .cast(pl.Int8)
            .alias("is_first_observed_migration")
        )

    # ------------------------------------------------------------
    # 4. Convert to pandas
    # ------------------------------------------------------------

    pdf = df.select(
        FEATURES
        + [
            "high_duration",
            "migration_timestamp",
        ]
    ).to_pandas()

    for column in FEATURES:

        pdf[column] = pd.to_numeric(
            pdf[column],
            errors="coerce",
        )

    # ------------------------------------------------------------
    # 5. Global chronological split
    # ------------------------------------------------------------

    split_index = int(len(pdf) * 0.75)

    if split_index <= 0 or split_index >= len(pdf):

        raise ValueError(
            "Invalid chronological train/test split."
        )

    train = pdf.iloc[:split_index].copy()
    test = pdf.iloc[split_index:].copy()

    X_train = train[FEATURES]
    y_train = train["high_duration"]

    X_test = test[FEATURES]
    y_test = test["high_duration"]

    print("\nChronological split:")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows:  {len(test):,}")

    print(
        f"\nTrain period: "
        f"{train['migration_timestamp'].min()} "
        f"to "
        f"{train['migration_timestamp'].max()}"
    )

    print(
        f"Test period:  "
        f"{test['migration_timestamp'].min()} "
        f"to "
        f"{test['migration_timestamp'].max()}"
    )

    # ------------------------------------------------------------
    # 6. Validate target classes
    # ------------------------------------------------------------

    train_classes = sorted(
        y_train.dropna().unique()
    )

    test_classes = sorted(
        y_test.dropna().unique()
    )

    print(f"\nTrain classes: {train_classes}")
    print(f"Test classes:  {test_classes}")

    if len(train_classes) < 2:

        raise ValueError(
            "Training data contains only one target class."
        )

    # ------------------------------------------------------------
    # 7. Preprocessing
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
                FEATURES,
            )
        ],
        remainder="drop",
    )

    # ------------------------------------------------------------
    # 8. Logistic regression
    # ------------------------------------------------------------

    model = Pipeline(
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

    print("\nTraining logistic regression...")

    model.fit(
        X_train,
        y_train,
    )

    # ------------------------------------------------------------
    # 9. Predictions
    # ------------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ------------------------------------------------------------
    # 10. Evaluation
    # ------------------------------------------------------------

    if len(np.unique(y_test)) >= 2:

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

    else:

        roc_auc = float("nan")

        print(
            "\nWarning: test set contains only one class."
        )

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # 11. Save predictions
    # ------------------------------------------------------------

    results = test[
        [
            "migration_timestamp",
        ]
    ].copy()

    results["actual_high_duration"] = (
        y_test.to_numpy()
    )

    results["predicted_high_duration"] = (
        predictions
    )

    results["estimated_probability"] = (
        probabilities
    )

    results_pl = pl.from_pandas(results)

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_pl.write_parquet(
        RESULTS_PATH
    )

    # ------------------------------------------------------------
    # 12. Save model coefficients
    # ------------------------------------------------------------

    classifier = (
        model
        .named_steps["classifier"]
    )

    coefficients = classifier.coef_[0]

    coefficient_rows = []

    for feature, coefficient in zip(
        FEATURES,
        coefficients,
    ):

        coefficient_rows.append(
            {
                "feature": feature,
                "coefficient": float(
                    coefficient
                ),
                "direction": (
                    "positive"
                    if coefficient > 0
                    else "negative"
                ),
                "absolute_coefficient": float(
                    abs(coefficient)
                ),
            }
        )

    coefficient_df = (
        pl.DataFrame(
            coefficient_rows
        )
        .sort(
            "absolute_coefficient",
            descending=True,
        )
    )

    coefficient_df.write_parquet(
        COEFFICIENTS_PATH
    )

    # ------------------------------------------------------------
    # 13. Final output
    # ------------------------------------------------------------

    print("\nSaved:")

    print(
        f"  {RESULTS_PATH}"
    )

    print(
        f"  {COEFFICIENTS_PATH}"
    )

    print("\nCoefficient interpretation:")

    print(coefficient_df)

    print("\n" + "=" * 70)
    print("Migration risk model completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
