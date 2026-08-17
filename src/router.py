from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import mlflow
import time
import random

from src.model_service import ModelService


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Telco Churn Champion-Challenger Router",
    description="A/B testing router for Champion and Challenger models",
    version="1.0.0",
)


# ============================================================
# MLflow Configuration
# ============================================================

mlflow.set_tracking_uri("http://localhost:5000")

CHAMPION_MODEL_URI = "models:/Telco-Churn-Champion@champion"
CHALLENGER_MODEL_URI = "models:/Telco-Churn-Challenger@challenger"


# ============================================================
# A/B Testing Configuration
# ============================================================

CHAMPION_TRAFFIC_PERCENT = 80
CHALLENGER_TRAFFIC_PERCENT = 20


# ============================================================
# Prometheus Metrics
# ============================================================

ROUTER_REQUESTS = Counter(
    "router_requests_total",
    "Total requests received by the router",
    ["model", "status"],
)

MODEL_REQUESTS = Counter(
    "router_model_requests_total",
    "Total requests routed to each model",
    ["model"],
)

MODEL_LATENCY = Histogram(
    "router_model_latency_seconds",
    "Model prediction latency",
    ["model"],
)

ROUTER_ERRORS = Counter(
    "router_errors_total",
    "Total router prediction errors",
    ["model"],
)


# ============================================================
# Load Models from MLflow Registry
# ============================================================

print("=" * 60)
print("LOADING MODELS FROM MLFLOW REGISTRY")
print("=" * 60)

print("Champion:", CHAMPION_MODEL_URI)
print("Challenger:", CHALLENGER_MODEL_URI)


try:
    champion_mlflow_model = mlflow.pyfunc.load_model(
        CHAMPION_MODEL_URI
    )

    champion_model = ModelService(
        champion_mlflow_model
    )

    print("Champion model loaded successfully")

except Exception as e:
    print("Champion model loading failed:")
    print(e)
    raise


try:
    challenger_mlflow_model = mlflow.pyfunc.load_model(
        CHALLENGER_MODEL_URI
    )

    challenger_model = ModelService(
        challenger_mlflow_model
    )

    print("Challenger model loaded successfully")

except Exception as e:
    print("Challenger model loading failed:")
    print(e)
    raise


print("=" * 60)
print("MODEL LOADING COMPLETE")
print("=" * 60)


# ============================================================
# Choose Model
# ============================================================

def choose_model():

    value = random.uniform(0, 100)

    if value < CHAMPION_TRAFFIC_PERCENT:
        return "champion"

    return "challenger"


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Telco Churn Champion-Challenger Router",
        "status": "running",
        "champion_traffic": f"{CHAMPION_TRAFFIC_PERCENT}%",
        "challenger_traffic": f"{CHALLENGER_TRAFFIC_PERCENT}%",
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "champion_model": "loaded",
        "challenger_model": "loaded",
        "mlflow": "connected",
    }


# ============================================================
# Prometheus Metrics
# ============================================================

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
def predict(data: dict):

    model_name = choose_model()

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # Route request
        # ----------------------------------------------------

        if model_name == "champion":

            result = champion_model.predict(data)

        else:

            result = challenger_model.predict(data)

        prediction = result["prediction"]
        probability = result["probability"]

        elapsed_time = time.time() - start_time

        # ----------------------------------------------------
        # Prometheus metrics
        # ----------------------------------------------------

        MODEL_REQUESTS.labels(
            model=model_name
        ).inc()

        ROUTER_REQUESTS.labels(
            model=model_name,
            status="200"
        ).inc()

        MODEL_LATENCY.labels(
            model=model_name
        ).observe(elapsed_time)

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "model": model_name,
            "model_role": (
                "Champion"
                if model_name == "champion"
                else "Challenger"
            ),
            "prediction": prediction,
            "churn": (
                "Yes"
                if prediction == 1
                else "No"
            ),
            "churn_probability": probability,
            "latency_seconds": round(
                elapsed_time,
                6
            ),
        }

    except Exception as e:

        ROUTER_ERRORS.labels(
            model=model_name
        ).inc()

        ROUTER_REQUESTS.labels(
            model=model_name,
            status="500"
        ).inc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
def startup_event():

    print("=" * 60)
    print("TELCO CHURN CHAMPION-CHALLENGER ROUTER")
    print("=" * 60)

    print(
        "Champion traffic   :",
        CHAMPION_TRAFFIC_PERCENT,
        "%"
    )

    print(
        "Challenger traffic :",
        CHALLENGER_TRAFFIC_PERCENT,
        "%"
    )

    print(
        "MLflow Champion    :",
        CHAMPION_MODEL_URI
    )

    print(
        "MLflow Challenger  :",
        CHALLENGER_MODEL_URI
    )

    print("Router status      : READY")

    print("=" * 60)
