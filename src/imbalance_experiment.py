import os
import json
import pickle

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. Load data
# ============================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).squeeze()


print("=" * 70)
print("IMBALANCE EXPERIMENT")
print("=" * 70)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTraining class percentage:")
print(
    y_train.value_counts(normalize=True) * 100
)


# ============================================================
# 2. Load Optuna best parameters
# ============================================================

with open(
    "artifacts/optuna/best_params.json",
    "r"
) as f:
    best_params = json.load(f)


# ============================================================
# 3. Calculate imbalance ratio
# ============================================================

class_counts = y_train.value_counts()

negative_count = class_counts[0]
positive_count = class_counts[1]

scale_pos_weight = negative_count / positive_count

print("\nScale Pos Weight:")
print(scale_pos_weight)


# ============================================================
# 4. Function to create models
# ============================================================

def create_models(strategy):

    models = {}

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    lr_params = best_params[
        "logistic_regression"
    ].copy()

    if strategy == "weighted":
        lr_params["class_weight"] = "balanced"

    elif strategy == "smote":
        lr_params["class_weight"] = None

    models["logistic_regression"] = (
        LogisticRegression(
            **lr_params,
            max_iter=2000,
            random_state=42
        )
    )


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf_params = best_params[
        "random_forest"
    ].copy()

    if strategy == "weighted":
        rf_params["class_weight"] = "balanced"

    elif strategy == "smote":
        rf_params["class_weight"] = None

    models["random_forest"] = (
        RandomForestClassifier(
            **rf_params,
            random_state=42,
            n_jobs=-1
        )
    )


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgb_params = best_params[
        "xgboost"
    ].copy()

    if strategy == "weighted":
        xgb_params["scale_pos_weight"] = scale_pos_weight

    elif strategy == "smote":
        xgb_params["scale_pos_weight"] = 1

    models["xgboost"] = (
        XGBClassifier(
            **xgb_params,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    )

    return models


# ============================================================
# 5. Run experiment
# ============================================================

strategies = [
    "none",
    "weighted",
    "smote"
]

results = []

os.makedirs(
    "artifacts/imbalance",
    exist_ok=True
)

os.makedirs(
    "models/imbalance",
    exist_ok=True
)


for strategy in strategies:

    print("\n")
    print("=" * 70)
    print(f"STRATEGY: {strategy.upper()}")
    print("=" * 70)

    models = create_models(strategy)

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name} "
            f"with {strategy}"
        )

        # ----------------------------------------------------
        # SMOTE pipeline
        # ----------------------------------------------------

        if strategy == "smote":

            pipeline = Pipeline(
                steps=[
                    (
                        "smote",
                        SMOTE(
                            random_state=42
                        )
                    ),
                    (
                        "model",
                        model
                    )
                ]
            )

            pipeline.fit(
                X_train,
                y_train
            )

            final_model = pipeline

        else:

            model.fit(
                X_train,
                y_train
            )

            final_model = model


        # ----------------------------------------------------
        # Test prediction
        # ----------------------------------------------------

        y_pred = final_model.predict(
            X_test
        )

        y_prob = final_model.predict_proba(
            X_test
        )[:, 1]


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )


        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            confusion_matrix(
                y_test,
                y_pred
            )
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "strategy": strategy,

            "model": model_name,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1,

            "roc_auc": roc_auc
        })


        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        model_path = (
            f"models/imbalance/"
            f"{model_name}_{strategy}.pkl"
        )

        with open(
            model_path,
            "wb"
        ) as f:

            pickle.dump(
                final_model,
                f
            )


# ============================================================
# 6. Create results dataframe
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
)


# ============================================================
# 7. Save results
# ============================================================

results_df.to_csv(
    "artifacts/imbalance/"
    "imbalance_results.csv",
    index=False
)


# ============================================================
# 8. Print final comparison
# ============================================================

print("\n")
print("=" * 80)
print("IMBALANCE STRATEGY COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 9. Best strategy
# ============================================================

best = results_df.iloc[0]

print("\n")
print("=" * 80)
print("CURRENT BEST MODEL")
print("=" * 80)

print(
    f"Model    : {best['model']}"
)

print(
    f"Strategy : {best['strategy']}"
)

print(
    f"F1       : {best['f1_score']:.4f}"
)

print(
    f"ROC-AUC  : {best['roc_auc']:.4f}"
)

print(
    f"Recall   : {best['recall']:.4f}"
)

print(
    f"Accuracy : {best['accuracy']:.4f}"
)


print("\nSaved:")
print(
    "artifacts/imbalance/imbalance_results.csv"
)
