import os
import json
import pickle
import optuna
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================================
# 1. Load training data
# ============================================================

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

print("Training data loaded")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# ============================================================
# 2. Create artifact directory
# ============================================================

os.makedirs("artifacts/optuna", exist_ok=True)
os.makedirs("models/optimized", exist_ok=True)


# ============================================================
# 3. Cross-validation
# ============================================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ============================================================
# 4. Logistic Regression Objective
# ============================================================

def logistic_objective(trial):

    C = trial.suggest_float(
        "C",
        0.001,
        10.0,
        log=True
    )

    class_weight = trial.suggest_categorical(
        "class_weight",
        [None, "balanced"]
    )

    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=2000,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    return scores.mean()


# ============================================================
# 5. Random Forest Objective
# ============================================================

def random_forest_objective(trial):

    n_estimators = trial.suggest_int(
        "n_estimators",
        100,
        500
    )

    max_depth = trial.suggest_int(
        "max_depth",
        3,
        20
    )

    min_samples_split = trial.suggest_int(
        "min_samples_split",
        2,
        10
    )

    min_samples_leaf = trial.suggest_int(
        "min_samples_leaf",
        1,
        5
    )

    max_features = trial.suggest_categorical(
        "max_features",
        ["sqrt", "log2"]
    )

    class_weight = trial.suggest_categorical(
        "class_weight",
        [None, "balanced", "balanced_subsample"]
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    return scores.mean()


# ============================================================
# 6. XGBoost Objective
# ============================================================

def xgboost_objective(trial):

    n_estimators = trial.suggest_int(
        "n_estimators",
        100,
        500
    )

    max_depth = trial.suggest_int(
        "max_depth",
        3,
        10
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        0.01,
        0.3,
        log=True
    )

    subsample = trial.suggest_float(
        "subsample",
        0.6,
        1.0
    )

    colsample_bytree = trial.suggest_float(
        "colsample_bytree",
        0.6,
        1.0
    )

    min_child_weight = trial.suggest_int(
        "min_child_weight",
        1,
        10
    )

    gamma = trial.suggest_float(
        "gamma",
        0.0,
        5.0
    )

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    return scores.mean()


# ============================================================
# 7. Run Optuna studies
# ============================================================

studies = {}

print("\n" + "=" * 60)
print("OPTUNA: LOGISTIC REGRESSION")
print("=" * 60)

logistic_study = optuna.create_study(
    direction="maximize",
    study_name="logistic_regression_optimization"
)

logistic_study.optimize(
    logistic_objective,
    n_trials=30
)

studies["logistic_regression"] = logistic_study


print("\n" + "=" * 60)
print("OPTUNA: RANDOM FOREST")
print("=" * 60)

rf_study = optuna.create_study(
    direction="maximize",
    study_name="random_forest_optimization"
)

rf_study.optimize(
    random_forest_objective,
    n_trials=30
)

studies["random_forest"] = rf_study


print("\n" + "=" * 60)
print("OPTUNA: XGBOOST")
print("=" * 60)

xgb_study = optuna.create_study(
    direction="maximize",
    study_name="xgboost_optimization"
)

xgb_study.optimize(
    xgboost_objective,
    n_trials=30
)

studies["xgboost"] = xgb_study


# ============================================================
# 8. Print best results
# ============================================================

results = []

for model_name, study in studies.items():

    print("\n" + "=" * 60)
    print(f"BEST RESULT: {model_name}")
    print("=" * 60)

    print("Best F1:", study.best_value)
    print("Best parameters:")
    
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    results.append({
        "model": model_name,
        "best_f1": study.best_value,
        "best_params": str(study.best_params)
    })


# ============================================================
# 9. Save Optuna results
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "artifacts/optuna/optuna_results.csv",
    index=False
)


# ============================================================
# 10. Save best parameters as JSON
# ============================================================

best_params = {
    model_name: study.best_params
    for model_name, study in studies.items()
}

with open(
    "artifacts/optuna/best_params.json",
    "w"
) as f:
    json.dump(
        best_params,
        f,
        indent=4
    )


# ============================================================
# 11. Save Optuna studies
# ============================================================

for model_name, study in studies.items():

    with open(
        f"artifacts/optuna/{model_name}_study.pkl",
        "wb"
    ) as f:

        pickle.dump(
            study,
            f
        )


print("\n" + "=" * 60)
print("OPTUNA OPTIMIZATION COMPLETED")
print("=" * 60)

print("\nArtifacts created:")
print("artifacts/optuna/optuna_results.csv")
print("artifacts/optuna/best_params.json")
print("artifacts/optuna/logistic_regression_study.pkl")
print("artifacts/optuna/random_forest_study.pkl")
print("artifacts/optuna/xgboost_study.pkl")
