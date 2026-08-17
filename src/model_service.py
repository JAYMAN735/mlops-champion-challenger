import pickle
import pandas as pd


PREPROCESSOR_PATH = "artifacts/preprocessing/preprocessor.pkl"


class ModelService:

    def __init__(self, model):
        self.model = model

        with open(PREPROCESSOR_PATH, "rb") as f:
            self.preprocessor = pickle.load(f)

        # MLflow PyFunc hides predict_proba().
        # Access the underlying sklearn model.
        self.raw_model = None

        try:
            self.raw_model = model._model_impl.sklearn_model
        except Exception:
            self.raw_model = model

    def predict(self, data: dict):

        df = pd.DataFrame([data])

        # Remove customer ID
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # Convert TotalCharges to numeric
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )

        # Apply training preprocessing
        X = self.preprocessor.transform(df)

        # Prediction
        prediction = self.model.predict(X)[0]

        # Probability from underlying sklearn model
        probability = None

        if hasattr(self.raw_model, "predict_proba"):
            probability = self.raw_model.predict_proba(X)[0][1]

        return {
            "prediction": int(prediction),
            "probability": (
                round(float(probability), 4)
                if probability is not None
                else None
            )
        }
