from zenml import step

import pandas as pd
import pickle
import json
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


@step
def evaluate_model(
    model_path: str,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate the trained champion model on the
    untouched test dataset.
    """

    print("=" * 60)
    print("MODEL EVALUATION STEP")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load model
    # --------------------------------------------------------

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # 2. Predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # 3. Metrics
    # --------------------------------------------------------

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

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # 4. Create results
    # --------------------------------------------------------

    results = {
        "model": "Random Forest",
        "strategy": "class_weight_balanced",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist()
    }

    # --------------------------------------------------------
    # 5. Print results
    # --------------------------------------------------------

    print("\nEvaluation Results")
    print("-" * 40)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # 6. Save evaluation artifact
    # --------------------------------------------------------

    os.makedirs(
        "artifacts/evaluation",
        exist_ok=True
    )

    with open(
        "artifacts/evaluation/"
        "zenml_evaluation.json",
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        "\nEvaluation artifact saved:"
    )

    print(
        "artifacts/evaluation/"
        "zenml_evaluation.json"
    )

    return results
