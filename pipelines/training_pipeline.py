from zenml import pipeline

from steps.ingest_data_step import ingest_data
from steps.preprocess_data_step import preprocess_data
from steps.train_model_step import train_model
from steps.evaluate_model_step import evaluate_model


@pipeline(enable_cache=False)
def training_pipeline():

    # --------------------------------------------------------
    # Step 1: Data ingestion
    # --------------------------------------------------------

    data = ingest_data()

    # --------------------------------------------------------
    # Step 2: Preprocessing
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = preprocess_data(
        data
    )

    # --------------------------------------------------------
    # Step 3: Model training
    # --------------------------------------------------------

    model_path = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Step 4: Model evaluation
    # --------------------------------------------------------

    evaluate_model(
        model_path,
        X_test,
        y_test
    )
