import pandas as pd
import pickle
import os


# ============================================================
# 1. Paths
# ============================================================

PREPROCESSOR_PATH = "artifacts/preprocessing/preprocessor.pkl"
MODEL_PATH = "models/final_model.pkl"

INPUT_PATH = "data/raw/telco_churn.csv"
OUTPUT_PATH = "artifacts/predictions.csv"


# ============================================================
# 2. Load preprocessor
# ============================================================

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)


# ============================================================
# 3. Load final champion model
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


print("Preprocessor loaded successfully")
print("Final champion model loaded successfully")


# ============================================================
# 4. Load raw data
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("\nRaw data shape:", df.shape)


# ============================================================
# 5. Keep customer IDs for output
# ============================================================

if "customerID" in df.columns:
    customer_ids = df["customerID"].copy()
else:
    customer_ids = pd.Series(
        range(len(df)),
        name="customerID"
    )


# ============================================================
# 6. Prepare data exactly as during training
# ============================================================

prediction_data = df.copy()

# Convert TotalCharges to numeric
if "TotalCharges" in prediction_data.columns:

    prediction_data["TotalCharges"] = pd.to_numeric(
        prediction_data["TotalCharges"],
        errors="coerce"
    )

# Remove missing values
prediction_data = prediction_data.dropna()

# Remove ID
if "customerID" in prediction_data.columns:
    prediction_data = prediction_data.drop(
        "customerID",
        axis=1
    )

# Remove actual target if it exists
if "Churn" in prediction_data.columns:
    prediction_data = prediction_data.drop(
        "Churn",
        axis=1
    )


# ============================================================
# 7. Preprocess
# ============================================================

X_processed = preprocessor.transform(
    prediction_data
)

print(
    "Processed prediction data shape:",
    X_processed.shape
)


# ============================================================
# 8. Make predictions
# ============================================================

predictions = model.predict(
    X_processed
)

probabilities = model.predict_proba(
    X_processed
)[:, 1]


# ============================================================
# 9. Convert predictions
# ============================================================

prediction_labels = [
    "Yes" if prediction == 1 else "No"
    for prediction in predictions
]


# ============================================================
# 10. Create output
# ============================================================

result = pd.DataFrame({
    "prediction": prediction_labels,
    "churn_probability": probabilities
})


# ============================================================
# 11. Add customer IDs
# ============================================================

result.insert(
    0,
    "customerID",
    customer_ids.iloc[
        prediction_data.index
    ].values
)


# ============================================================
# 12. Save predictions
# ============================================================

os.makedirs(
    "artifacts",
    exist_ok=True
)

result.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 13. Display results
# ============================================================

print("\nPrediction completed successfully!")

print("\nPrediction counts:")
print(result["prediction"].value_counts())

print("\nFirst 10 predictions:")
print(result.head(10))

print("\nSaved:")
print(OUTPUT_PATH)
