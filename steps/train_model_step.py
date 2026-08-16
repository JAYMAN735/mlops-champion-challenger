from zenml import step

import pandas as pd
import os
import pickle

from sklearn.ensemble import RandomForestClassifier


@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> str:
    """
    Train the selected champion model.

    Champion strategy:
        Random Forest + class_weight='balanced'
    """

    print("=" * 60)
    print("MODEL TRAINING STEP")
    print("=" * 60)

    print(f"Training data shape: {X_train.shape}")
    print(f"Target shape       : {y_train.shape}")

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("\nModel training completed.")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        "models/zenml",
        exist_ok=True
    )

    model_path = "models/zenml/champion_model.pkl"

    with open(
        model_path,
        "wb"
    ) as f:
        pickle.dump(
            model,
            f
        )

    print("\nModel saved:")
    print(model_path)

    return model_path
