from zenml import step
import pandas as pd


@step
def ingest_data() -> pd.DataFrame:
    """
    Load the raw Telco Churn dataset.

    Returns:
        pd.DataFrame: Raw Telco Churn dataset.
    """

    data_path = "data/raw/telco_churn.csv"

    df = pd.read_csv(data_path)

    print("=" * 60)
    print("DATA INGESTION STEP")
    print("=" * 60)

    print(f"Dataset path : {data_path}")
    print(f"Dataset shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    return df
