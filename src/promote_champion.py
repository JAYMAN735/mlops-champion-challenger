import os
import json
import pickle
import shutil


# ============================================================
# 1. Paths
# ============================================================

champion_metadata_path = (
    "artifacts/champion/champion_metadata.json"
)

final_model_path = "models/final_model.pkl"

final_metadata_path = (
    "artifacts/champion/final_model_metadata.json"
)


# ============================================================
# 2. Load Champion metadata
# ============================================================

if not os.path.exists(champion_metadata_path):
    raise FileNotFoundError(
        f"Champion metadata not found: "
        f"{champion_metadata_path}"
    )


with open(
    champion_metadata_path,
    "r"
) as f:
    champion = json.load(f)


experiment = champion["experiment"]
model = champion["model"]
strategy = champion.get("strategy")


# ============================================================
# 3. Determine source model
# ============================================================

if experiment == "imbalance":

    if strategy is None:
        raise ValueError(
            "Champion strategy is missing for imbalance experiment."
        )

    source_model = (
        f"models/imbalance/"
        f"{model}_{strategy}.pkl"
    )

elif experiment == "optimized":

    source_model = (
        f"models/optimized/"
        f"{model}_optimized.pkl"
    )

elif experiment == "baseline":

    source_model = (
        f"models/{model}.pkl"
    )

else:

    raise ValueError(
        f"Unknown experiment: {experiment}"
    )


# ============================================================
# 4. Check source model
# ============================================================

if not os.path.exists(source_model):
    raise FileNotFoundError(
        f"Champion model not found: {source_model}"
    )


# ============================================================
# 5. Create directories
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "artifacts/champion",
    exist_ok=True
)


# ============================================================
# 6. Promote Champion
# ============================================================

shutil.copy2(
    source_model,
    final_model_path
)


# ============================================================
# 7. Verify promoted model
# ============================================================

with open(
    final_model_path,
    "rb"
) as f:
    final_model = pickle.load(f)


# ============================================================
# 8. Create final metadata
# ============================================================

metadata = {
    "model_name": model,
    "experiment": experiment,
    "strategy": strategy,
    "model_type": type(final_model).__name__,
    "selection_metric": champion["selection_metric"],
    "f1_score": champion["f1_score"],
    "roc_auc": champion["roc_auc"],
    "recall": champion["recall"],
    "precision": champion["precision"],
    "accuracy": champion["accuracy"],
    "source_model": source_model,
    "final_model": final_model_path
}


# ============================================================
# 9. Save metadata
# ============================================================

with open(
    final_metadata_path,
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# 10. Print result
# ============================================================

print("=" * 70)
print("CHAMPION MODEL PROMOTED SUCCESSFULLY")
print("=" * 70)

print(f"Model       : {model}")
print(f"Experiment  : {experiment}")
print(f"Strategy    : {strategy}")
print(f"Model type  : {type(final_model).__name__}")

print(f"F1 Score    : {champion['f1_score']:.4f}")
print(f"ROC-AUC     : {champion['roc_auc']:.4f}")
print(f"Recall      : {champion['recall']:.4f}")
print(f"Precision   : {champion['precision']:.4f}")
print(f"Accuracy    : {champion['accuracy']:.4f}")

print("\nSource model:")
print(source_model)

print("\nFinal model:")
print(final_model_path)

print("\nMetadata:")
print(final_metadata_path)
