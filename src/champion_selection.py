import os
import json
import pickle
import pandas as pd


# ============================================================
# 1. Load results
# ============================================================

baseline = pd.read_csv(
    "artifacts/baseline_results.csv"
)

optimized = pd.read_csv(
    "artifacts/evaluation/optimized_results.csv"
)

imbalance = pd.read_csv(
    "artifacts/imbalance/imbalance_results.csv"
)


# ============================================================
# 2. Standardize model names
# ============================================================

baseline["experiment"] = "baseline"
optimized["experiment"] = "optimized"
imbalance["experiment"] = "imbalance"


# ============================================================
# 3. Combine results
# ============================================================

all_results = pd.concat(
    [
        baseline,
        optimized,
        imbalance
    ],
    ignore_index=True
)


# ============================================================
# 4. Sort by primary metric
# ============================================================

all_results = all_results.sort_values(
    by=[
        "f1_score",
        "roc_auc",
        "recall"
    ],
    ascending=False
).reset_index(drop=True)


# ============================================================
# 5. Select champion
# ============================================================

champion = all_results.iloc[0]


# ============================================================
# 6. Add Champion / Challenger status
# ============================================================

all_results["status"] = "challenger"

all_results.loc[
    all_results.index == 0,
    "status"
] = "champion"


# ============================================================
# 7. Print complete comparison
# ============================================================

print("\n" + "=" * 100)
print("CHAMPION - CHALLENGER COMPARISON")
print("=" * 100)

columns_to_show = [
    "experiment",
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "status"
]

# Some baseline/optimized rows don't have strategy
print(
    all_results[
        [c for c in columns_to_show if c in all_results.columns]
    ].to_string(index=False)
)


# ============================================================
# 8. Print champion
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHAMPION")
print("=" * 70)

print(
    f"Experiment : {champion['experiment']}"
)

print(
    f"Model      : {champion['model']}"
)

print(
    f"F1 Score   : {champion['f1_score']:.4f}"
)

print(
    f"ROC-AUC    : {champion['roc_auc']:.4f}"
)

print(
    f"Recall     : {champion['recall']:.4f}"
)

print(
    f"Precision  : {champion['precision']:.4f}"
)

print(
    f"Accuracy   : {champion['accuracy']:.4f}"
)


# ============================================================
# 9. Save final comparison artifact
# ============================================================

os.makedirs(
    "artifacts/champion",
    exist_ok=True
)

all_results.to_csv(
    "artifacts/champion/"
    "champion_challenger_results.csv",
    index=False
)


# ============================================================
# 10. Save champion metadata
# ============================================================

champion_metadata = {
    "experiment": champion["experiment"],
    "model": champion["model"],
    "strategy": champion.get("strategy", None),
    "f1_score": float(champion["f1_score"]),
    "roc_auc": float(champion["roc_auc"]),
    "recall": float(champion["recall"]),
    "precision": float(champion["precision"]),
    "accuracy": float(champion["accuracy"]),
    "selection_metric": "f1_score",
    "selection_rule": (
        "Highest F1-score; "
        "ROC-AUC and recall used as supporting metrics"
    )
}


with open(
    "artifacts/champion/"
    "champion_metadata.json",
    "w"
) as f:

    json.dump(
        champion_metadata,
        f,
        indent=4
    )


print("\nArtifacts saved:")
print(
    "artifacts/champion/"
    "champion_challenger_results.csv"
)

print(
    "artifacts/champion/"
    "champion_metadata.json"
)
