import json
import os
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_DATA_PATH = "data/raw/telco_churn.csv"
PROCESSED_DATA_PATH = "data/processed"
ARTIFACT_PATH = "artifacts/preprocessing"


def preprocess_data(file_path: str):
    # --------------------------------------------------
    # 1. Load raw dataset
    # --------------------------------------------------
    df = pd.read_csv(file_path)

    print(f"Original dataset shape: {df.shape}")

    # --------------------------------------------------
    # 2. Remove unnecessary column
    # --------------------------------------------------
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # --------------------------------------------------
    # 3. Convert TotalCharges to numeric
    # --------------------------------------------------
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # --------------------------------------------------
    # 4. Remove rows with missing values
    # --------------------------------------------------
    missing_before = int(df.isnull().sum().sum())

    df = df.dropna().reset_index(drop=True)

    missing_after = int(df.isnull().sum().sum())

    # --------------------------------------------------
    # 5. Separate features and target
    # --------------------------------------------------
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # --------------------------------------------------
    # 6. Identify numerical and categorical columns
    # --------------------------------------------------
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["str"]
    ).columns.tolist()

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # --------------------------------------------------
    # 7. Train-test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # 8. Create preprocessing pipeline
    # --------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_columns
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            )
        ]
    )

    # --------------------------------------------------
    # 9. Fit ONLY on training data
    # --------------------------------------------------
    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    # --------------------------------------------------
    # 10. Get feature names
    # --------------------------------------------------
    feature_columns = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_columns
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_columns
    )

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    # --------------------------------------------------
    # 11. Create directories
    # --------------------------------------------------
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(ARTIFACT_PATH, exist_ok=True)

    # --------------------------------------------------
    # 12. Save processed datasets
    # --------------------------------------------------
    X_train_processed.to_csv(
        f"{PROCESSED_DATA_PATH}/X_train.csv",
        index=False
    )

    X_test_processed.to_csv(
        f"{PROCESSED_DATA_PATH}/X_test.csv",
        index=False
    )

    y_train.to_csv(
        f"{PROCESSED_DATA_PATH}/y_train.csv",
        index=False
    )

    y_test.to_csv(
        f"{PROCESSED_DATA_PATH}/y_test.csv",
        index=False
    )

    # --------------------------------------------------
    # 13. Save preprocessing artifacts
    # --------------------------------------------------
    with open(
        f"{ARTIFACT_PATH}/preprocessor.pkl",
        "wb"
    ) as file:
        pickle.dump(preprocessor, file)

    with open(
        f"{ARTIFACT_PATH}/feature_columns.pkl",
        "wb"
    ) as file:
        pickle.dump(list(feature_columns), file)

    # --------------------------------------------------
    # 14. Save preprocessing report
    # --------------------------------------------------
    report = {
        "original_rows": int(len(df) + missing_before),
        "final_rows": int(len(df)),
        "original_columns": int(len(pd.read_csv(file_path).columns)),
        "final_columns": int(df.shape[1]),
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
        "processed_feature_count": int(len(feature_columns))
    }

    with open(
        f"{ARTIFACT_PATH}/preprocessing_report.json",
        "w"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nPreprocessing completed successfully.")

    print(f"X_train shape: {X_train_processed.shape}")
    print(f"X_test shape: {X_test_processed.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    preprocess_data(RAW_DATA_PATH)
