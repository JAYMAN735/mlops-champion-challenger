# 🚀 MLOps Champion–Challenger Platform for Telco Customer Churn Prediction

## 📌 Project Overview

This project implements an end-to-end **MLOps pipeline for Telco Customer Churn Prediction**.

The system goes beyond traditional machine learning model training by implementing the complete machine learning lifecycle:

**Data Ingestion → Preprocessing → Model Training → Optuna Optimization → Evaluation → Champion–Challenger Selection → MLflow Tracking → FastAPI Serving → Docker → Prometheus Monitoring → Grafana Dashboard → Render Deployment**

The project trains and evaluates multiple machine learning models and identifies a **Champion model** while maintaining alternative **Challenger models**.

The final model is exposed through a production-style FastAPI REST API and deployed publicly using Render.

---

# 🎯 Objectives

The main objectives of this project are:

* Build a complete reproducible MLOps pipeline.
* Predict customer churn using machine learning.
* Train multiple candidate models.
* Perform hyperparameter optimization using Optuna.
* Compare models using multiple evaluation metrics.
* Implement Champion–Challenger model selection.
* Track experiments using MLflow.
* Orchestrate the ML workflow using ZenML.
* Serialize models and preprocessing artifacts using Pickle.
* Build a REST API using FastAPI.
* Containerize the application using Docker.
* Monitor API and model metrics using Prometheus.
* Visualize monitoring metrics using Grafana.
* Deploy the prediction API using Render.
* Maintain the complete project using Git and GitHub.

---

# 🧠 Problem Statement

Customer churn is an important business problem for telecommunications companies.

A customer may discontinue a service because of factors such as:

* Contract type
* Monthly charges
* Tenure
* Internet service
* Payment method
* Technical support
* Security services
* Streaming services

The objective of this project is to develop a machine learning system that predicts whether a customer is likely to churn.

The system produces:

1. A binary churn prediction.
2. A churn probability.
3. Model confidence.
4. Model information.
5. Operational monitoring metrics.

---

# 📊 Dataset

The project uses the **Telco Customer Churn dataset**.

Important features include:

| Feature          | Description                       |
| ---------------- | --------------------------------- |
| gender           | Customer gender                   |
| SeniorCitizen    | Senior citizen indicator          |
| Partner          | Partner status                    |
| Dependents       | Dependents status                 |
| tenure           | Number of months with the company |
| PhoneService     | Phone service subscription        |
| MultipleLines    | Multiple line subscription        |
| InternetService  | Internet service type             |
| OnlineSecurity   | Online security subscription      |
| OnlineBackup     | Online backup subscription        |
| DeviceProtection | Device protection subscription    |
| TechSupport      | Technical support subscription    |
| StreamingTV      | Streaming TV subscription         |
| StreamingMovies  | Streaming movie subscription      |
| Contract         | Contract type                     |
| PaperlessBilling | Paperless billing status          |
| PaymentMethod    | Payment method                    |
| MonthlyCharges   | Monthly charges                   |
| TotalCharges     | Total charges                     |

### Target Variable

`Churn`

* `Yes` → Customer churned
* `No` → Customer did not churn

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │   Telco Churn Data  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Data Ingestion   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Preprocessing     │
                         │ Encoding / Cleaning │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │       Candidate Models          │
                  │                                 │
                  │ Logistic Regression             │
                  │ Random Forest                   │
                  │ XGBoost                         │
                  └──────────────┬──────────────────┘
                                 │
                                 ▼
                         ┌─────────────────────┐
                         │       Optuna        │
                         │ Hyperparameter      │
                         │ Optimization        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Evaluation     │
                         │ Accuracy / F1 /    │
                         │ Recall / ROC-AUC   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌────────────────────────────┐
                    │ Champion–Challenger       │
                    │ Model Selection            │
                    └──────────────┬─────────────┘
                                   │
                                   ▼
                            ┌────────────┐
                            │   MLflow   │
                            │ Experiment │
                            │ Tracking   │
                            └──────┬─────┘
                                   │
                                   ▼
                            ┌────────────┐
                            │  FastAPI   │
                            │ Prediction │
                            │    API     │
                            └──────┬─────┘
                                   │
                             ┌─────┴──────┐
                             │            │
                             ▼            ▼
                         Prometheus    Render
                             │        Deployment
                             ▼
                          Grafana
                          Dashboard
```

---

# 🔄 MLOps Workflow

The project workflow consists of the following stages:

## 1. Data Ingestion

The raw Telco Churn dataset is stored in:

```text
data/raw/telco_churn.csv
```

The ingestion step loads the dataset into the pipeline.

---

## 2. Data Preprocessing

The preprocessing stage:

* Handles missing values.
* Converts numerical features.
* Encodes categorical features.
* Splits data into training and testing datasets.
* Saves the preprocessing pipeline.

The preprocessing artifact is stored as:

```text
artifacts/preprocessing/preprocessor.pkl
```

---

## 3. Model Training

Three primary candidate models were considered:

```text
Logistic Regression
Random Forest
XGBoost
```

Each model is trained and evaluated.

---

# 🔬 Optuna Hyperparameter Optimization

Optuna is used to automatically search for better model hyperparameters.

The optimization process searches through candidate parameter combinations and evaluates them using the selected objective metric.

Example parameters include:

```text
C
class_weight
max_depth
learning_rate
n_estimators
subsample
```

Optimized parameters are stored in:

```text
artifacts/optuna/best_params.json
```

---

# 🏆 Champion–Challenger Architecture

The project uses a Champion–Challenger approach.

### Champion

The currently selected production model.

### Challenger

Alternative candidate models that can be evaluated against the Champion.

The implemented workflow selected:

```text
Champion: Random Forest
Model Role: champion
```

The purpose of this architecture is to make future model replacement easier.

A new model can be evaluated against the existing Champion before being promoted.

---

# 📈 Model Evaluation

The optimized candidate models achieved the following results:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.7285 |    0.4934 | 0.7941 |   0.6086 |  0.8352 |
| Random Forest       |   0.7598 |    0.5349 | 0.7380 |   0.6202 |  0.8319 |
| XGBoost             |   0.7974 |    0.6422 | 0.5374 |   0.5852 |  0.8404 |

The project also includes experiments addressing class imbalance.

---

# 🧪 MLflow

MLflow is used for:

* Experiment tracking
* Parameter logging
* Metric logging
* Model tracking
* Champion/Challenger experiment comparison

Example tracked information includes:

```text
Model Name
Model Role
Hyperparameters
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

MLflow experiments include Champion and Challenger model runs.

---

# 🔗 ZenML

ZenML is used to orchestrate the machine learning pipeline.

The pipeline contains major stages such as:

```text
Ingestion
    ↓
Preprocessing
    ↓
Training
    ↓
Evaluation
```

The pipeline can be executed using:

```bash
python run_pipeline.py
```

ZenML provides reproducibility and allows individual pipeline steps to be tracked independently.

---

# 💾 Model Artifacts

The project stores serialized models using Pickle.

Important files include:

```text
models/
├── final_model.pkl
├── logistic_regression.pkl
├── random_forest.pkl
└── xgboost.pkl
```

The preprocessing object is stored as:

```text
artifacts/preprocessing/preprocessor.pkl
```

These artifacts allow the API to load the trained model and apply the same preprocessing used during training.

---

# ⚡ FastAPI Model Serving

FastAPI is used to expose the machine learning model through REST endpoints.

Main endpoints:

```text
GET  /health
POST /predict
GET  /metrics
GET  /docs
```

---

## Health Endpoint

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "model": "Random Forest",
  "model_role": "champion"
}
```

---

## Prediction Endpoint

```text
POST /predict
```

Example request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 65.5,
  "TotalCharges": 786.0
}
```

The API returns the predicted churn class and probability.

---

# 📊 Monitoring

The API exposes Prometheus-compatible metrics through:

```text
GET /metrics
```

Important metrics include:

```text
api_requests_total
api_request_latency_seconds
model_predictions_total
model_prediction_errors_total
model_info
churn_prediction
churn_probability
prediction_latency_seconds
model_confidence
prediction_success_total
prediction_failure_total
api_up
model_loaded
preprocessor_loaded
input_feature_count
```

---

# 🔥 Prometheus

Prometheus collects metrics from the FastAPI application.

Configuration:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "telco-churn-api"
    metrics_path: "/metrics"

    static_configs:
      - targets:
          - "telco-churn-api:8000"
```

Prometheus is used for monitoring the locally containerized API.

---

# 📉 Grafana

Grafana is connected to Prometheus and provides a monitoring dashboard.

The dashboard contains panels for metrics such as:

* API availability
* API requests
* Prediction count
* Churn probability
* Churn prediction
* Prediction latency
* Model confidence
* Model information
* Prediction success/failure

The dashboard is automatically provisioned from:

```text
grafana/dashboards/telco-churn-dashboard.json
```

Grafana provisioning files are located at:

```text
grafana/provisioning/
```

---

# 🐳 Docker

Docker is used to containerize the FastAPI application and monitoring services.

The project contains:

```text
Dockerfile
docker-compose.yml
```

Build the API container:

```bash
docker compose build --no-cache telco-churn-api
```

Start the services:

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

The local services are:

```text
FastAPI      → http://localhost:8000
Prometheus   → http://localhost:9090
Grafana      → http://localhost:3001
```

---

# ☁️ Render Deployment

The FastAPI model-serving application is deployed using Render.

Public deployment:

```text
https://mlops-champion-challenger.onrender.com
```

Available endpoints:

```text
https://mlops-champion-challenger.onrender.com/health

https://mlops-champion-challenger.onrender.com/docs

https://mlops-champion-challenger.onrender.com/metrics
```

The Render deployment makes the machine learning prediction API accessible from external machines.

### Important Architecture Note

The current deployment separates cloud serving from local monitoring.

```text
                 PUBLIC CLOUD
                     │
                     ▼
                  Render
                     │
                     ▼
                  FastAPI
                     │
                     ▼
               ML Model
                     
                     
              LOCAL MACHINE
                     
          ┌──────────┴──────────┐
          ▼                     ▼
     Prometheus              Grafana
          │                     │
          └──────────┬──────────┘
                     ▼
                Monitoring
```

Currently, Prometheus and Grafana run through Docker Compose locally, while the FastAPI application is deployed on Render.

---

# 📁 Project Structure

```text
mlops-champion-challenger/
│
├── artifacts/
│   ├── baseline_results.csv
│   ├── champion/
│   ├── evaluation/
│   ├── imbalance/
│   ├── optuna/
│   ├── predictions.csv
│   └── preprocessing/
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv
│   └── processed/
│
├── grafana/
│   ├── dashboards/
│   │   └── telco-churn-dashboard.json
│   │
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboard.yml
│       │
│       └── datasources/
│           └── prometheus.yml
│
├── models/
│   ├── final_model.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── pipelines/
│   └── training_pipeline.py
│
├── src/
│   ├── api.py
│   ├── champion_selection.py
│   ├── data_ingestion.py
│   ├── evaluate.py
│   ├── evaluate_optimized.py
│   ├── imbalance_experiment.py
│   ├── mlflow_tracking.py
│   ├── model_service.py
│   ├── optuna_tuning.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── promote_champion.py
│   ├── register_models.py
│   ├── router.py
│   ├── train.py
│   └── train_optimized.py
│
├── steps/
│   ├── evaluate_model_step.py
│   ├── ingest_data_step.py
│   ├── preprocess_data_step.py
│   └── train_model_step.py
│
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
├── requirements-docker.txt
├── run_pipeline.py
└── README.md
```

---

# 🛠️ Technologies Used

| Technology     | Purpose                       |
| -------------- | ----------------------------- |
| Python         | Programming language          |
| Pandas         | Data manipulation             |
| NumPy          | Numerical computation         |
| Scikit-learn   | Machine learning              |
| XGBoost        | Gradient boosting             |
| Optuna         | Hyperparameter optimization   |
| ZenML          | Pipeline orchestration        |
| MLflow         | Experiment/model tracking     |
| Pickle         | Model serialization           |
| FastAPI        | Model serving                 |
| Uvicorn        | API server                    |
| Docker         | Containerization              |
| Docker Compose | Multi-container orchestration |
| Prometheus     | Metrics collection            |
| Grafana        | Metrics visualization         |
| Git            | Version control               |
| GitHub         | Source-code hosting           |
| Render         | Cloud deployment              |

---

# ⚙️ Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/JAYMAN735/mlops-champion-challenger.git
```

Move into the project:

```bash
cd mlops-champion-challenger
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For Docker-specific dependencies:

```bash
pip install -r requirements-docker.txt
```

---

# ▶️ Running the ZenML Pipeline

Run:

```bash
python run_pipeline.py
```

The pipeline performs:

```text
Data Ingestion
      ↓
Preprocessing
      ↓
Training
      ↓
Evaluation
```

---

# ▶️ Running FastAPI Locally

Start the API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Metrics:

```bash
curl http://localhost:8000/metrics
```

---

# 🐳 Running the Complete Docker Stack

Build the services:

```bash
docker compose build
```

Start:

```bash
docker compose up -d
```

Check:

```bash
docker compose ps
```

Stop:

```bash
docker compose down
```

---

# 📊 Monitoring URLs

## FastAPI

```text
http://localhost:8000
```

## Swagger

```text
http://localhost:8000/docs
```

## Prometheus

```text
http://localhost:9090
```

## Grafana

```text
http://localhost:3001
```

---

# 🔍 Useful Prometheus Queries

The following PromQL queries can be used to analyze the API.

### API Availability

```promql
api_up
```

### Total API Requests

```promql
api_requests_total
```

### Request Rate

```promql
rate(api_requests_total[5m])
```

### Prediction Count

```promql
model_predictions_total
```

### Prediction Error Count

```promql
model_prediction_errors_total
```

### Churn Probability

```promql
churn_probability
```

### Churn Prediction

```promql
churn_prediction
```

### Prediction Latency

```promql
prediction_latency_seconds
```

### Model Confidence

```promql
model_confidence
```

### Successful Predictions

```promql
prediction_success_total
```

### Failed Predictions

```promql
prediction_failure_total
```

### Model Loaded

```promql
model_loaded
```

### Preprocessor Loaded

```promql
preprocessor_loaded
```

---

# 🔐 Reproducibility

The project is designed to make the machine learning workflow reproducible.

Important reproducibility components include:

* Version-controlled source code
* Requirements files
* ZenML pipeline
* Serialized preprocessing artifact
* Serialized model artifacts
* Optuna optimization results
* MLflow experiment tracking
* Docker configuration
* Prometheus configuration
* Grafana dashboard provisioning

---

# 📌 Results

The final system successfully demonstrates:

* Multiple model training
* Hyperparameter optimization
* Model comparison
* Champion–Challenger selection
* MLflow experiment tracking
* ZenML pipeline orchestration
* FastAPI model serving
* Docker containerization
* Prometheus monitoring
* Grafana visualization
* Cloud deployment using Render
* GitHub-based version control

---

# 🚧 Challenges

During development, several challenges were addressed:

### ZenML Configuration

ZenML version and configuration compatibility issues required environment and configuration management.

### Dataset Imbalance

The target variable contains class imbalance, requiring additional experiments and class-weight strategies.

### Docker Dependencies

The Docker environment required the correct dependencies and model artifacts to be included in the container.

### Grafana Provisioning

Grafana datasource provisioning initially caused startup failures because the datasource configuration and dashboard datasource references needed to match.

### Prometheus Networking

Prometheus communicates with the FastAPI service through the Docker Compose service name:

```text
telco-churn-api:8000
```

instead of using `localhost`.

### Cloud Deployment

Render deployment required the API to listen on the externally accessible interface:

```text
0.0.0.0
```

and use the deployment environment's assigned port.

---

# 🔮 Future Scope

Future improvements could include:

* Automated CI/CD using GitHub Actions
* Automated model retraining
* Automated Champion promotion
* Model drift detection
* Data drift monitoring
* Cloud-hosted Prometheus
* Cloud-hosted Grafana
* Alerting for model/API failures
* Authentication and authorization
* API rate limiting
* Automated rollback
* Multiple production model versions
* Canary deployment
* A/B testing between Champion and Challenger models

---

# 👨‍💻 Author

**Jayman Prajapati**

M.Sc. Data Science

---

# 🔗 Project Links

### GitHub Repository

https://github.com/JAYMAN735/mlops-champion-challenger

### Render Deployment

https://mlops-champion-challenger.onrender.com

### FastAPI Documentation

https://mlops-champion-challenger.onrender.com/docs

### Health Check

https://mlops-champion-challenger.onrender.com/health

### Metrics

https://mlops-champion-challenger.onrender.com/metrics

---


