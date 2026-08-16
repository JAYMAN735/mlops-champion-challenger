import os
import json
import pickle
import mlflow
import mlflow.sklearn


# ============================================================
# MLflow configuration
# ============================================================

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "Telco-Churn-Champion-Challenger"

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Load Champion metadata
# ============================================================

metadata_path = "artifacts/champion/champion_metadata.json"

if not os.path.exists(metadata_path):
    raise FileNotFoundError(
        f"Champion metadata not found: {metadata_path}"
    )

with open(metadata_path, "r") as f:
    champion = json.load(f)


# ============================================================
# Determine Champion model path
# ============================================================

experiment = champion["experiment"]
model_name = champion["model"]
strategy = champion.get("strategy")


if experiment == "imbalance":

    model_path = (
        f"models/imbalance/"
        f"{model_name}_{strategy}.pkl"
    )

elif experiment == "optimized":

    model_path = (
        f"models/optimized/"
        f"{model_name}_optimized.pkl"
    )

elif experiment == "baseline":

    model_path = (
        f"models/{model_name}.pkl"
    )

else:
    raise ValueError(
        f"Unknown experiment: {experiment}"
    )


# ============================================================
# Load Champion model
# ============================================================

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model not found: {model_path}"
    )

with open(model_path, "rb") as f:
    model = pickle.load(f)


# ============================================================
# Start MLflow run
# ============================================================

with mlflow.start_run(
    run_name="Champion-RandomForest-Weighted"
) as run:

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    mlflow.log_param(
        "experiment",
        experiment
    )

    mlflow.log_param(
        "model",
        model_name
    )

    mlflow.log_param(
        "strategy",
        strategy
    )

    mlflow.log_param(
        "selection_metric",
        champion["selection_metric"]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mlflow.log_metric(
        "accuracy",
        champion["accuracy"]
    )

    mlflow.log_metric(
        "precision",
        champion["precision"]
    )

    mlflow.log_metric(
        "recall",
        champion["recall"]
    )

    mlflow.log_metric(
        "f1_score",
        champion["f1_score"]
    )

    mlflow.log_metric(
        "roc_auc",
        champion["roc_auc"]
    )

    # --------------------------------------------------------
    # Tags
    # --------------------------------------------------------

    mlflow.set_tag(
        "model_role",
        "champion"
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

    mlflow.sklearn.log_model(
        model,
        name="champion_model"
    )

    # --------------------------------------------------------
    # Log artifacts
    # --------------------------------------------------------

    mlflow.log_artifact(
        "artifacts/champion/champion_metadata.json"
    )

    mlflow.log_artifact(
        "artifacts/champion/champion_challenger_results.csv"
    )

    mlflow.log_artifact(
        "artifacts/optuna/best_params.json"
    )

    print("=" * 70)
    print("MLFLOW TRACKING COMPLETED")
    print("=" * 70)

    print(f"Run ID      : {run.info.run_id}")
    print(f"Experiment  : {experiment}")
    print(f"Model       : {model_name}")
    print(f"Strategy    : {strategy}")
    print(f"F1 Score    : {champion['f1_score']:.4f}")
    print(f"ROC-AUC     : {champion['roc_auc']:.4f}")
    print(f"Model path  : {model_path}")
