import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# -----------------------------
# 1. Load test data
# -----------------------------

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Test data loaded")
print("X_test shape:", X_test.shape)


# -----------------------------
# 2. Models to evaluate
# -----------------------------

model_names = [
    "logistic_regression",
    "random_forest",
    "xgboost"
]


results = []


# -----------------------------
# 3. Evaluate each model
# -----------------------------

for name in model_names:

    print("\n" + "=" * 50)
    print(f"Evaluating: {name}")
    print("=" * 50)

    # Load model
    with open(f"models/{name}.pkl", "rb") as f:
        model = pickle.load(f)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability for ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Print metrics
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Store results
    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })


# -----------------------------
# 4. Create comparison table
# -----------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("BASELINE MODEL COMPARISON")
print("=" * 70)

print(results_df)


# -----------------------------
# 5. Save evaluation results
# -----------------------------

results_df.to_csv(
    "artifacts/baseline_results.csv",
    index=False
)

print("\nSaved:")
print("artifacts/baseline_results.csv")
