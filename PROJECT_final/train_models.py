import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# ============================================================
# AILAF - MODEL TRAINING AND COMPARISON
# ============================================================
# Test 1 and Quiz 1 are always required.
# Test 2 and Quiz 2 are optional, so one model is trained for
# each combination of available marks (4 scenarios).
#
# Decision Tree, Random Forest and XGBoost are all tuned the
# same way (small grid, student-grouped 3-fold CV, macro F1),
# compared on the same test set, and the best one (highest
# 5-fold CV macro F1 with all four marks) becomes AILAF's model.
# ============================================================


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = Path("data/processed/ailaf_ml_dataset.csv")
RESULTS_DIR = Path("data/processed/model_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Beginner", "Intermediate", "Advanced"]

SCENARIOS = {
    "t1q1": ["test1", "quiz1"],
    "t1q1t2": ["test1", "quiz1", "test2"],
    "t1q1q2": ["test1", "quiz1", "quiz2"],
    "t1q1t2q2": ["test1", "quiz1", "test2", "quiz2"]
}

MAIN_SCENARIO = "t1q1t2q2"

GRIDS = {
    "Decision Tree": {
        "max_depth": [3, 5, 8],
        "min_samples_leaf": [10, 30]
    },
    "Random Forest": {
        "n_estimators": [200],
        "max_depth": [5, 8],
        "min_samples_leaf": [10, 30]
    },
    "XGBoost": {
        "n_estimators": [200, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1]
    }
}


def make_model(name, params):

    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, **params)

    if name == "Random Forest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)

    return XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        **params
    )


def make_weights(y):
    # square-root balanced weights: helps Beginner and Advanced
    # without pushing average students into Beginner
    return np.sqrt(compute_sample_weight("balanced", y))


def cross_validate(name, params, X, y, groups, folds):

    # student-grouped cross-validation, returns macro F1 and accuracy per fold
    f1_scores = []
    accuracies = []

    for train_index, valid_index in GroupKFold(n_splits=folds).split(X, y, groups):

        model = make_model(name, params)

        model.fit(
            X.iloc[train_index],
            y.iloc[train_index],
            sample_weight=make_weights(y.iloc[train_index])
        )

        predictions = model.predict(X.iloc[valid_index])

        f1_scores.append(f1_score(y.iloc[valid_index], predictions, average="macro"))
        accuracies.append(accuracy_score(y.iloc[valid_index], predictions))

    return f1_scores, accuracies


# ============================================================
# 2. LOAD DATA AND SPLIT BY STUDENT
# ============================================================

data = pd.read_csv(DATA_FILE)

print("Dataset:", data.shape)
print(data["competency_level"].value_counts())

y_all = data["competency_level"].map({c: i for i, c in enumerate(CLASSES)})
groups_all = data["id_student"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

train_index, test_index = next(splitter.split(data, y_all, groups_all))

train_data = data.iloc[train_index].reset_index(drop=True)
test_data = data.iloc[test_index].reset_index(drop=True)

y_train = y_all.iloc[train_index].reset_index(drop=True)
y_test = y_all.iloc[test_index].reset_index(drop=True)

groups_train = train_data["id_student"]

print("\nTrain rows:", len(train_data), "| Test rows:", len(test_data))


# ============================================================
# 3. TUNE AND EVALUATE EVERY ALGORITHM IN EVERY SCENARIO
# ============================================================

trained = {}
rows = []

for scenario, features in SCENARIOS.items():

    X_train = train_data[features]
    X_test = test_data[features]

    for name, grid in GRIDS.items():

        # tuning: student-grouped 3-fold CV on the training set
        best_params = None
        best_score = -1

        for params in ParameterGrid(grid):

            f1_scores, _ = cross_validate(
                name, params, X_train, y_train, groups_train, folds=3
            )

            if np.mean(f1_scores) > best_score:
                best_score = np.mean(f1_scores)
                best_params = params

        # final fit on the training set with the best parameters
        model = make_model(name, best_params)

        model.fit(X_train, y_train, sample_weight=make_weights(y_train))

        predictions = model.predict(X_test)

        trained[(name, scenario)] = {
            "model": model,
            "params": best_params,
            "predictions": predictions
        }

        rows.append({
            "Scenario": scenario,
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "F1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "Macro_F1": f1_score(y_test, predictions, average="macro", zero_division=0),
            "Best_Params": json.dumps({k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in best_params.items()})
        })

        print(f"{scenario:9s} {name:14s} accuracy {rows[-1]['Accuracy']:.3f}  macro F1 {rows[-1]['Macro_F1']:.3f}")

by_scenario = pd.DataFrame(rows)

by_scenario.to_csv(RESULTS_DIR / "model_comparison_by_scenario.csv", index=False)


# ============================================================
# 4. 5-FOLD CROSS-VALIDATION (ALL FOUR MARKS)
# ============================================================

print("\n5-fold student-grouped cross-validation (all four marks)")

X_main = data[SCENARIOS[MAIN_SCENARIO]]

cv_rows = []

for name in GRIDS:

    params = trained[(name, MAIN_SCENARIO)]["params"]

    f1_scores, accuracies = cross_validate(
        name, params, X_main, y_all, groups_all, folds=5
    )

    cv_rows.append({
        "Model": name,
        "CV_Accuracy_Mean": np.mean(accuracies),
        "CV_Accuracy_Std": np.std(accuracies),
        "CV_Macro_F1_Mean": np.mean(f1_scores),
        "CV_Macro_F1_Std": np.std(f1_scores)
    })

    print(f"{name:14s} CV macro F1 {np.mean(f1_scores):.3f} +/- {np.std(f1_scores):.3f}")

cv_table = pd.DataFrame(cv_rows)

comparison = by_scenario[by_scenario["Scenario"] == MAIN_SCENARIO].drop(
    columns=["Scenario"]
).merge(cv_table, on="Model")

comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)


# ============================================================
# 5. CHOOSE THE FINAL ALGORITHM
# ============================================================

best_name = comparison.sort_values(
    "CV_Macro_F1_Mean", ascending=False
).iloc[0]["Model"]

print("\nBest algorithm (highest CV macro F1):", best_name)


# ============================================================
# 6. SAVE FINAL MODELS (ONE PER SCENARIO)
# ============================================================

for scenario in SCENARIOS:

    joblib.dump(
        trained[(best_name, scenario)]["model"],
        RESULTS_DIR / f"final_model_{scenario}.joblib"
    )

main_predictions = trained[(best_name, MAIN_SCENARIO)]["predictions"]

report = classification_report(
    y_test,
    main_predictions,
    labels=[0, 1, 2],
    target_names=CLASSES,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(y_test, main_predictions, labels=[0, 1, 2])

scenario_summary = {}

for scenario in SCENARIOS:

    row = by_scenario[
        (by_scenario["Scenario"] == scenario) & (by_scenario["Model"] == best_name)
    ].iloc[0]

    scenario_summary[scenario] = {
        "features": SCENARIOS[scenario],
        "accuracy": float(row["Accuracy"]),
        "macro_f1": float(row["Macro_F1"])
    }

info = {
    "algorithm": best_name,
    "classes": CLASSES,
    "main_scenario": MAIN_SCENARIO,
    "scenarios": scenario_summary,
    "rows": int(len(data)),
    "train_rows": int(len(train_data)),
    "test_rows": int(len(test_data)),
    "class_counts": data["competency_level"].value_counts().to_dict(),
    "report": report,
    "confusion_matrix": matrix.tolist(),
    "cv": cv_rows,
    "label_rule": "average of later assessments: below 60 Beginner, below 80 Intermediate, otherwise Advanced",
    "weights": "square-root balanced class weights",
    "split": "student-grouped 80/20 split"
}

with open(RESULTS_DIR / "final_model_info.json", "w") as file:
    json.dump(info, file, indent=2)

with open(RESULTS_DIR / "final_model_classification_report.txt", "w") as file:
    file.write(f"Final model: {best_name}\n")
    file.write("Student-grouped 80/20 split, all four marks\n\n")
    file.write(classification_report(
        y_test, main_predictions, labels=[0, 1, 2],
        target_names=CLASSES, zero_division=0
    ))


# ============================================================
# 7. FIGURES
# ============================================================

for name in GRIDS:

    figure, axis = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, trained[(name, MAIN_SCENARIO)]["predictions"], labels=[0, 1, 2]),
        display_labels=CLASSES
    ).plot(ax=axis, cmap="Greens", colorbar=False)

    axis.set_title(f"{name} - Confusion Matrix")

    figure.tight_layout()
    figure.savefig(RESULTS_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150)
    plt.close(figure)

final_main = trained[(best_name, MAIN_SCENARIO)]["model"]

importance = pd.DataFrame({
    "Feature": SCENARIOS[MAIN_SCENARIO],
    "Importance": final_main.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

figure, axis = plt.subplots(figsize=(5, 3.5))
axis.barh(importance["Feature"][::-1], importance["Importance"][::-1], color="#247452")
axis.set_title(f"{best_name} - Feature Importance")
figure.tight_layout()
figure.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
plt.close(figure)

print("\nFinal model:", best_name)
print(comparison[["Model", "Accuracy", "Precision", "Recall", "F1", "Macro_F1", "CV_Macro_F1_Mean"]].round(3).to_string(index=False))
print("\nPer-class recall:", {c: round(report[c]["recall"], 3) for c in CLASSES})
print("Saved to:", RESULTS_DIR)
