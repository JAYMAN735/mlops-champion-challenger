from zenml import step
import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@step
def preprocess_data(
    df: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series
]:
    """
    Preprocess the Telco Churn dataset.

    Returns:
        X_train
        X_test
        y_train
        y_test
    """

    print("=" * 60)
    print("PREPROCESSING STEP")
    print("=" * 60)

    data = df.copy()

    # --------------------------------------------------------
    # 1. Convert TotalCharges to numeric
    # --------------------------------------------------------

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # 2. Remove rows with missing values
    # --------------------------------------------------------

    data = data.dropna().reset_index(drop=True)

    # --------------------------------------------------------
    # 3. Separate target
    # --------------------------------------------------------

    y = data["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    X = data.drop(
        columns=["Churn"]
    )

    # --------------------------------------------------------
    # 4. Remove customer ID
    # --------------------------------------------------------

    if "customerID" in X.columns:
        X = X.drop(
            columns=["customerID"]
        )

    # --------------------------------------------------------
    # 5. Identify numerical/categorical columns
    # --------------------------------------------------------

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print(
        f"Numerical columns   : {len(numerical_columns)}"
    )

    print(
        f"Categorical columns : {len(categorical_columns)}"
    )

    # --------------------------------------------------------
    # 6. Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # 7. Create preprocessor
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 8. Fit ONLY on training data
    # --------------------------------------------------------

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # --------------------------------------------------------
    # 9. Convert back to DataFrame
    # --------------------------------------------------------

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    # --------------------------------------------------------
    # 10. Save preprocessor artifact
    # --------------------------------------------------------

    os.makedirs(
        "artifacts/preprocessing",
        exist_ok=True
    )

    with open(
        "artifacts/preprocessing/preprocessor.pkl",
        "wb"
    ) as f:
        pickle.dump(
            preprocessor,
            f
        )

    # --------------------------------------------------------
    # 11. Save processed datasets
    # --------------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    X_train_processed.to_csv(
        "data/processed/X_train.csv",
        index=False
    )

    X_test_processed.to_csv(
        "data/processed/X_test.csv",
        index=False
    )

    y_train.to_csv(
        "data/processed/y_train.csv",
        index=False
    )

    y_test.to_csv(
        "data/processed/y_test.csv",
        index=False
    )

    # --------------------------------------------------------
    # 12. Print information
    # --------------------------------------------------------

    print(
        f"\nOriginal dataset : {data.shape}"
    )

    print(
        f"X_train          : {X_train_processed.shape}"
    )

    print(
        f"X_test           : {X_test_processed.shape}"
    )

    print(
        f"y_train          : {y_train.shape}"
    )

    print(
        f"y_test           : {y_test.shape}"
    )

    print("\nPreprocessor saved:")
    print(
        "artifacts/preprocessing/preprocessor.pkl"
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test
    )
