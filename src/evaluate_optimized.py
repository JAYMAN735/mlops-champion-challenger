import pandas as pd
import pickle
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. Load test data
# ============================================================

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()


# ============================================================
# 2. Optimized models
# ============================================================

model_names = [
    "logistic_regression_optimized",
    "random_forest_optimized",
    "xgboost_optimized"
]


results = []


# ============================================================
# 3. Evaluate models
# ============================================================

for name in model_names:

    print("\n" + "=" * 60)
    print(f"EVALUATING: {name}")
    print("=" * 60)

    with open(
        f"models/optimized/{name}.pkl",
        "rb"
    ) as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })


# ============================================================
# 4. Results table
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("OPTIMIZED MODEL COMPARISON")
print("=" * 80)

print(results_df.to_string(index=False))


# ============================================================
# 5. Save results
# ============================================================

os.makedirs("artifacts/evaluation", exist_ok=True)

results_df.to_csv(
    "artifacts/evaluation/optimized_results.csv",
    index=False
)

print("\nSaved:")
print("artifacts/evaluation/optimized_results.csv")
