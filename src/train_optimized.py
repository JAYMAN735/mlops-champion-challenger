import os
import json
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================================
# 1. Load training data
# ============================================================

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

print("Training data loaded")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# ============================================================
# 2. Load Optuna best parameters
# ============================================================

with open("artifacts/optuna/best_params.json", "r") as f:
    best_params = json.load(f)

print("\nOptuna parameters loaded")


# ============================================================
# 3. Create model directory
# ============================================================

os.makedirs("models/optimized", exist_ok=True)


# ============================================================
# 4. Create optimized models
# ============================================================

models = {

    "logistic_regression_optimized": LogisticRegression(
        **best_params["logistic_regression"],
        max_iter=2000,
        random_state=42
    ),

    "random_forest_optimized": RandomForestClassifier(
        **best_params["random_forest"],
        random_state=42,
        n_jobs=-1
    ),

    "xgboost_optimized": XGBClassifier(
        **best_params["xgboost"],
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 5. Train optimized models
# ============================================================

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    model.fit(X_train, y_train)

    model_path = f"models/optimized/{name}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained successfully")
    print(f"Saved: {model_path}")


print("\n" + "=" * 60)
print("ALL OPTIMIZED MODELS TRAINED")
print("=" * 60)
