from pipelines.training_pipeline import training_pipeline


if __name__ == "__main__":

    print("=" * 60)
    print("STARTING ZENML TRAINING PIPELINE")
    print("=" * 60)

    training_pipeline()

    print("\nPipeline execution completed.")
