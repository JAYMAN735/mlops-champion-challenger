import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# -----------------------------
# 1. Load processed data
# -----------------------------

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Data loaded successfully")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# -----------------------------
# 2. Create models
# -----------------------------

models = {

    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "xgboost": XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss"
    )
}


# -----------------------------
# 3. Create model directory
# -----------------------------

os.makedirs("models", exist_ok=True)


# -----------------------------
# 4. Train models
# -----------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    # Save model
    model_path = f"models/{name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"{name} trained successfully")
    print(f"Saved: {model_path}")


print("\nAll baseline models trained successfully!")
