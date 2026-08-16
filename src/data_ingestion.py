import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw Telco Churn dataset.
    """
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    file_path = "data/raw/telco_churn.csv"

    df = load_data(file_path)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())
