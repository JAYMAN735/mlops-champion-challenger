from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
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
# Model Configuration
# ============================================================

MODEL_PATH = "models/final_model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessing/preprocessor.pkl"

MODEL_NAME = "Random Forest"
MODEL_ROLE = "champion"


# ============================================================
# Prometheus Metrics
# ============================================================

# 1. API request count
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

# 2. API request latency
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
)

# 3. Total predictions
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["prediction"],
)

# 4. Prediction errors
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total number of prediction errors",
)

# 5. Model information
MODEL_INFO = Gauge(
    "model_info",
    "Information about the model currently serving predictions",
    ["model", "model_role"],
)

# 6. Latest prediction
CHURN_PREDICTION = Gauge(
    "churn_prediction",
    "Latest churn prediction: 1=Yes, 0=No",
)

# 7. Latest probability
CHURN_PROBABILITY = Gauge(
    "churn_probability",
    "Latest predicted probability of customer churn",
)

# 8. Latest prediction latency
PREDICTION_LATENCY = Gauge(
    "prediction_latency_seconds",
    "Latency of the latest prediction request in seconds",
)

# 9. Latest model confidence
MODEL_CONFIDENCE = Gauge(
    "model_confidence",
    "Confidence of the latest model prediction",
)

# 10. Successful predictions
PREDICTION_SUCCESS = Counter(
    "prediction_success_total",
    "Total number of successful predictions",
)

# 11. Prediction failures
PREDICTION_FAILURES = Counter(
    "prediction_failure_total",
    "Total number of failed predictions",
)

# 12. API uptime
API_UP = Gauge(
    "api_up",
    "API availability status: 1=up, 0=down",
)

# 13. Model loaded status
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the model is loaded: 1=yes, 0=no",
)

# 14. Preprocessor loaded status
PREPROCESSOR_LOADED = Gauge(
    "preprocessor_loaded",
    "Whether the preprocessor is loaded: 1=yes, 0=no",
)

# 15. Number of input features
INPUT_FEATURE_COUNT = Gauge(
    "input_feature_count",
    "Number of features received for prediction",
)


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
# Initialize Model Metrics
# ============================================================

MODEL_INFO.labels(
    model=MODEL_NAME,
    model_role=MODEL_ROLE,
).set(1)

MODEL_LOADED.set(1)
PREPROCESSOR_LOADED.set(1)
API_UP.set(1)


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Telco Churn Prediction API",
        "status": "running",
        "model": MODEL_NAME,
        "model_role": MODEL_ROLE,
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
        "model": MODEL_NAME,
        "model_role": MODEL_ROLE,
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

        # --------------------------------------------------------
        # Convert input JSON to DataFrame
        # --------------------------------------------------------

        df = pd.DataFrame([data])

        # --------------------------------------------------------
        # Remove customer ID if provided
        # --------------------------------------------------------

        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # --------------------------------------------------------
        # Convert TotalCharges
        # --------------------------------------------------------

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce",
            )

        # --------------------------------------------------------
        # Record number of input features
        # --------------------------------------------------------

        INPUT_FEATURE_COUNT.set(len(df.columns))

        # --------------------------------------------------------
        # Preprocessing
        # --------------------------------------------------------

        X = preprocessor.transform(df)

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        prediction = int(
            model.predict(X)[0]
        )

        # --------------------------------------------------------
        # Probability
        # --------------------------------------------------------

        probability = float(
            model.predict_proba(X)[0][1]
        )

        # --------------------------------------------------------
        # Confidence
        # --------------------------------------------------------

        confidence = max(
            probability,
            1 - probability
        )

        # --------------------------------------------------------
        # Calculate latency
        # --------------------------------------------------------

        elapsed_time = time.time() - start_time

        # ========================================================
        # Update Prometheus Metrics
        # ========================================================

        PREDICTION_COUNT.labels(
            prediction=str(prediction)
        ).inc()

        PREDICTION_SUCCESS.inc()

        CHURN_PREDICTION.set(prediction)

        CHURN_PROBABILITY.set(probability)

        PREDICTION_LATENCY.set(elapsed_time)

        MODEL_CONFIDENCE.set(confidence)

        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            status="200",
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint="/predict"
        ).observe(elapsed_time)

        # --------------------------------------------------------
        # Return response
        # --------------------------------------------------------

        return {
            "prediction": prediction,
            "churn": "Yes" if prediction == 1 else "No",
            "churn_probability": round(
                probability,
                4,
            ),
            "model": MODEL_NAME,
            "model_role": MODEL_ROLE,
            "latency_seconds": round(
                elapsed_time,
                6,
            ),
        }

    except Exception as e:

        # --------------------------------------------------------
        # Error metrics
        # --------------------------------------------------------

        PREDICTION_ERRORS.inc()
        PREDICTION_FAILURES.inc()

        elapsed_time = time.time() - start_time

        PREDICTION_LATENCY.set(elapsed_time)

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
# Startup
# ============================================================

@app.on_event("startup")
def startup_event():

    API_UP.set(1)

    print("=" * 60)
    print("TELCO CHURN PREDICTION API")
    print("=" * 60)
    print("Model loaded       :", MODEL_PATH)
    print("Preprocessor loaded:", PREPROCESSOR_PATH)
    print("Model              :", MODEL_NAME)
    print("Model role         :", MODEL_ROLE)
    print("API status         : READY")
    print("Prometheus         : ENABLED")
    print("=" * 60)
