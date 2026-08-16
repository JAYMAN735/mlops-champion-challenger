import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn


# ============================================================
# MLflow configuration
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "Telco-Churn-Champion-Challenger"

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Load evaluation results
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
# Add experiment labels
# ============================================================

baseline["experiment"] = "baseline"
optimized["experiment"] = "optimized"
imbalance["experiment"] = "imbalance"


# ============================================================
# Combine results
# ============================================================

results = pd.concat(
    [
        baseline,
        optimized,
        imbalance
    ],
    ignore_index=True
)


# ============================================================
# Load Champion metadata
# ============================================================

champion_path = (
    "artifacts/champion/champion_metadata.json"
)

import json

with open(champion_path, "r") as f:
    champion = json.load(f)


champion_model = champion["model"]
champion_experiment = champion["experiment"]
champion_strategy = champion.get("strategy")


# ============================================================
# Remove duplicate candidate rows
# ============================================================

results = results.drop_duplicates(
    subset=[
        "experiment",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]
)


# ============================================================
# Track every candidate
# ============================================================

for _, row in results.iterrows():

    experiment = row["experiment"]
    model_name = row["model"]

    strategy = row.get("strategy")

    # --------------------------------------------------------
    # Determine model path
    # --------------------------------------------------------

    if experiment == "baseline":

        model_path = (
            f"models/{model_name}.pkl"
        )

    elif experiment == "optimized":

        model_path = (
            f"models/optimized/{model_name}.pkl"
        )

    elif experiment == "imbalance":

        if pd.isna(strategy):

            strategy = "none"

        model_path = (
            f"models/imbalance/"
            f"{model_name}_{strategy}.pkl"
        )

    else:
        continue

    # --------------------------------------------------------
    # Verify model
    # --------------------------------------------------------

    if not os.path.exists(model_path):

        print(
            f"WARNING: Model not found: "
            f"{model_path}"
        )

        continue

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # --------------------------------------------------------
    # Determine Champion status
    # --------------------------------------------------------

    is_champion = (
        experiment == champion_experiment
        and model_name == champion_model
        and (
            strategy == champion_strategy
            or champion_strategy is None
        )
        and abs(
            float(row["f1_score"])
            - float(champion["f1_score"])
        ) < 1e-10
    )

    role = (
        "champion"
        if is_champion
        else "challenger"
    )

    # --------------------------------------------------------
    # Run name
    # --------------------------------------------------------

    run_name = (
        f"{role.upper()}-"
        f"{experiment}-"
        f"{model_name}"
    )

    if strategy and not pd.isna(strategy):

        run_name += f"-{strategy}"

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=run_name
    ):

        # Parameters
        mlflow.log_param(
            "experiment",
            experiment
        )

        mlflow.log_param(
            "model",
            model_name
        )

        if strategy and not pd.isna(strategy):

            mlflow.log_param(
                "strategy",
                strategy
            )

        # Metrics
        mlflow.log_metric(
            "accuracy",
            float(row["accuracy"])
        )

        mlflow.log_metric(
            "precision",
            float(row["precision"])
        )

        mlflow.log_metric(
            "recall",
            float(row["recall"])
        )

        mlflow.log_metric(
            "f1_score",
            float(row["f1_score"])
        )

        mlflow.log_metric(
            "roc_auc",
            float(row["roc_auc"])
        )

        # Tags
        mlflow.set_tag(
            "model_role",
            role
        )

        mlflow.set_tag(
            "project",
            "telco_churn"
        )

        mlflow.set_tag(
            "pipeline",
            "champion_challenger"
        )

        # --------------------------------------------------------
	# Log model
	# --------------------------------------------------------

if model_name.startswith("xgboost"):
	mlflow.sklearn.log_model(
        	model,
  	      	name="model",
        	skops_trusted_types=[
	            "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier"
        	]
    	)

else:
	mlflow.sklearn.log_model(
       		model,
       		name="model"
	)

print(
	f"Tracked: {run_name}"
)


print("\n" + "=" * 70)
print("ALL CANDIDATE MODELS TRACKED IN MLFLOW")
print("=" * 70)
