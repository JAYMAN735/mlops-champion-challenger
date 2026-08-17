import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# MLflow Configuration
# ============================================================

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

EXPERIMENT_NAME = "Telco-Churn-Champion-Challenger"

CHAMPION_RUN_ID = "008908fbaabc47e781c6946a6b8ca889"
CHALLENGER_RUN_ID = "8f361d6587d94fa4b6f776c2a04bd6f0"


# ============================================================
# Model Registry Names
# ============================================================

CHAMPION_MODEL_NAME = "Telco-Churn-Champion"

CHALLENGER_MODEL_NAME = "Telco-Churn-Challenger"


# ============================================================
# Register Champion
# ============================================================

print("=" * 60)
print("REGISTERING CHAMPION MODEL")
print("=" * 60)

champion_model_uri = (
    f"runs:/{CHAMPION_RUN_ID}/model"
)

champion_result = mlflow.register_model(
    model_uri=champion_model_uri,
    name=CHAMPION_MODEL_NAME,
)

print("Champion registered")
print("Name   :", CHAMPION_MODEL_NAME)
print("Version:", champion_result.version)


# ============================================================
# Register Challenger
# ============================================================

print("=" * 60)
print("REGISTERING CHALLENGER MODEL")
print("=" * 60)

challenger_model_uri = (
    f"runs:/{CHALLENGER_RUN_ID}/model"
)

challenger_result = mlflow.register_model(
    model_uri=challenger_model_uri,
    name=CHALLENGER_MODEL_NAME,
)

print("Challenger registered")
print("Name   :", CHALLENGER_MODEL_NAME)
print("Version:", challenger_result.version)


# ============================================================
# Add Model Aliases
# ============================================================

print("=" * 60)
print("SETTING MODEL ALIASES")
print("=" * 60)

client.set_registered_model_alias(
    CHAMPION_MODEL_NAME,
    "champion",
    champion_result.version,
)

client.set_registered_model_alias(
    CHALLENGER_MODEL_NAME,
    "challenger",
    challenger_result.version,
)


print("Champion alias   : champion")
print("Challenger alias : challenger")


# ============================================================
# Add Tags
# ============================================================

client.set_model_version_tag(
    CHAMPION_MODEL_NAME,
    champion_result.version,
    "role",
    "champion",
)

client.set_model_version_tag(
    CHALLENGER_MODEL_NAME,
    challenger_result.version,
    "role",
    "challenger",
)


# ============================================================
# Completed
# ============================================================

print("=" * 60)
print("MLFLOW MODEL REGISTRY COMPLETED")
print("=" * 60)

print(
    f"Champion : models:/{CHAMPION_MODEL_NAME}@champion"
)

print(
    f"Challenger: models:/{CHALLENGER_MODEL_NAME}@challenger"
)

print("=" * 60)
