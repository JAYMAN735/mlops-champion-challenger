import os
import mlflow
import mlflow.sklearn
import pickle


# ============================================================
# MLflow Configuration
# ============================================================

mlflow.set_tracking_uri("http://localhost:5000")

EXPERIMENT_NAME = "Telco-Churn-Champion-Challenger"

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Model Paths
# ============================================================

CHAMPION_MODEL = "models/imbalance/random_forest_weighted.pkl"
CHALLENGER_MODEL = "models/imbalance/xgboost_weighted.pkl"


# ============================================================
# Verify Model Files
# ============================================================

if not os.path.exists(CHAMPION_MODEL):
    raise FileNotFoundError(
        f"Champion model not found: {CHAMPION_MODEL}"
    )

if not os.path.exists(CHALLENGER_MODEL):
    raise FileNotFoundError(
        f"Challenger model not found: {CHALLENGER_MODEL}"
    )


# ============================================================
# Load Models
# ============================================================

print("=" * 60)
print("LOADING CHAMPION MODEL")
print("=" * 60)

with open(CHAMPION_MODEL, "rb") as f:
    champion_model = pickle.load(f)


print("=" * 60)
print("LOADING CHALLENGER MODEL")
print("=" * 60)

with open(CHALLENGER_MODEL, "rb") as f:
    challenger_model = pickle.load(f)


# ============================================================
# Champion Metrics
# ============================================================

champion_metrics = {
    "f1_score": 0.6215644820295984,
    "roc_auc": 0.8329084075767066,
    "recall": 0.786096256684492,
    "precision": 0.513986013986014,
    "accuracy": 0.7455579246624022,
}


# ============================================================
# Challenger Metrics
# ============================================================

challenger_metrics = {
    "f1_score": 0.6187961985216474,
    "roc_auc": 0.8339968214690611,
    "recall": 0.7834224598930482,
    "precision": 0.5113438045375218,
    "accuracy": 0.7434257285003554,
}


# ============================================================
# Register Champion
# ============================================================

print("=" * 60)
print("LOGGING CHAMPION MODEL TO MLFLOW")
print("=" * 60)

with mlflow.start_run(
    run_name="REGISTRY-CHAMPION-RandomForest"
) as run:

    mlflow.log_param("model_role", "champion")
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("strategy", "weighted")

    for metric_name, metric_value in champion_metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    mlflow.sklearn.log_model(
        champion_model,
        artifact_path="model",
    )

    champion_run_id = run.info.run_id

    print("Champion Run ID:", champion_run_id)


# ============================================================
# Register Challenger
# ============================================================

print("=" * 60)
print("LOGGING CHALLENGER MODEL TO MLFLOW")
print("=" * 60)

with mlflow.start_run(
    run_name="REGISTRY-CHALLENGER-XGBoost"
) as run:

    mlflow.log_param("model_role", "challenger")
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("strategy", "weighted")

    for metric_name, metric_value in challenger_metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    # XGBoost requires trusted types for MLflow/skops validation
    mlflow.sklearn.log_model(
        challenger_model,
        artifact_path="model",
        skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )

    challenger_run_id = run.info.run_id

    print("Challenger Run ID:", challenger_run_id)


# ============================================================
# Completed
# ============================================================

print("=" * 60)
print("MODELS LOGGED SUCCESSFULLY")
print("=" * 60)

print("Champion  :", champion_run_id)
print("Challenger:", challenger_run_id)

print("=" * 60)
print("NEXT STEP: REGISTER MODELS IN MLFLOW MODEL REGISTRY")
print("=" * 60)
