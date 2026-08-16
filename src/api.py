from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import pandas as pd
import pickle
import os
import time


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Telco Churn Prediction API",
    description="MLOps Champion Model Prediction Service",
    version="1.0.0",
)


# ============================================================
# Prometheus Metrics
# ============================================================

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["prediction"],
)

PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total number of prediction errors",
)


# ============================================================
# Model and Preprocessor Paths
# ============================================================

MODEL_PATH = "models/final_model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessing/preprocessor.pkl"


# ============================================================
# Check Required Files
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(
        f"Preprocessor not found: {PREPROCESSOR_PATH}"
    )


# ============================================================
# Load Model
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ============================================================
# Load Preprocessor
# ============================================================

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Telco Churn Prediction API",
        "status": "running",
        "model": "Random Forest Champion",
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


# ============================================================
# Prometheus Metrics Endpoint
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

    start_time = time.time()

    try:

        # ----------------------------------------------------
        # Convert JSON input to DataFrame
        # ----------------------------------------------------

        df = pd.DataFrame([data])

        # ----------------------------------------------------
        # Remove customer ID
        # ----------------------------------------------------

        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # ----------------------------------------------------
        # Convert TotalCharges to numeric
        # ----------------------------------------------------

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce",
            )

        # ----------------------------------------------------
        # Apply training preprocessor
        # ----------------------------------------------------

        X = preprocessor.transform(df)

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = int(
            model.predict(X)[0]
        )

        # ----------------------------------------------------
        # Record prediction metric
        # ----------------------------------------------------

        PREDICTION_COUNT.labels(
            prediction=str(prediction)
        ).inc()

        # ----------------------------------------------------
        # Calculate probability
        # ----------------------------------------------------

        probability = float(
            model.predict_proba(X)[0][1]
        )

        # ----------------------------------------------------
        # Record request metrics
        # ----------------------------------------------------

        elapsed_time = time.time() - start_time

        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            status="200",
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint="/predict"
        ).observe(elapsed_time)

        # ----------------------------------------------------
        # Return prediction
        # ----------------------------------------------------

        return {
            "prediction": prediction,
            "churn": "Yes" if prediction == 1 else "No",
            "churn_probability": round(
                probability,
                4,
            ),
        }

    except Exception as e:

        # ----------------------------------------------------
        # Record prediction error
        # ----------------------------------------------------

        PREDICTION_ERRORS.inc()

        elapsed_time = time.time() - start_time

        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            status="400",
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint="/predict"
        ).observe(elapsed_time)

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# ============================================================
# Startup Message
# ============================================================

@app.on_event("startup")
def startup_event():

    print("=" * 60)
    print("TELCO CHURN PREDICTION API")
    print("=" * 60)
    print("Model loaded       :", MODEL_PATH)
    print("Preprocessor loaded:", PREPROCESSOR_PATH)
    print("API status         : READY")
    print("Prometheus         : ENABLED")
    print("=" * 60)
