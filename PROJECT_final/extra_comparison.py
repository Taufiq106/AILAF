import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# AILAF - EXTRA COMPARISON (Report Table 4.10)
# ============================================================
# Same four marks, same student-grouped split, two models:
#   Random Forest (settings of the final all-four-marks model)
#   Logistic Regression (simple baseline)
# Tasks:
#   1. AILAF level (3 classes)
#   2. OULAD final result, 4 classes
#   3. OULAD final result, 3 classes (withdrawn removed)
# Run from the project folder:  python extra_comparison.py
# Needs data/oulad/studentInfo.csv and data/processed/ailaf_ml_dataset.csv
# ============================================================

data = pd.read_csv("data/processed/ailaf_ml_dataset.csv")

student_info = pd.read_csv("data/oulad/studentInfo.csv")

key = ["id_student", "code_module", "code_presentation"]

data = data.merge(student_info[key + ["final_result"]], on=key, how="left")

# share of withdrawn students (data used vs all registered students)
modules = data[["code_module", "code_presentation"]].drop_duplicates()
registered = student_info.merge(modules, on=["code_module", "code_presentation"])

print("AILAF records:", len(data))
print("Withdrawn share in AILAF records: %.1f%%" % ((data["final_result"] == "Withdrawn").mean() * 100))
print("Registered students in the same module presentations:", len(registered))
print("Withdrawn share among them: %.1f%%" % ((registered["final_result"] == "Withdrawn").mean() * 100))

FOUR = ["test1", "quiz1", "test2", "quiz2"]
TWO = ["test1", "quiz1"]


def forest():
    return RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def weights(y):
    return np.sqrt(compute_sample_weight("balanced", y))


def evaluate(df, features, target, model):

    train_index, test_index = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        .split(df, groups=df["id_student"])
    )

    X = df[features]
    y = df[target]

    if hasattr(model, "steps"):
        fit_name = model.steps[-1][0] + "__sample_weight"
    else:
        fit_name = "sample_weight"

    model.fit(X.iloc[train_index], y.iloc[train_index],
              **{fit_name: weights(y.iloc[train_index])})

    predictions = model.predict(X.iloc[test_index])
    actual = y.iloc[test_index]

    return (
        accuracy_score(actual, predictions) * 100,
        f1_score(actual, predictions, average="macro"),
        actual.value_counts(normalize=True).max() * 100
    )


tasks = [
    ("AILAF level, 3 classes", data, "competency_level"),
    ("Final result, 4 classes", data.dropna(subset=["final_result"]).reset_index(drop=True), "final_result"),
    ("Final result, 3 classes", data[data["final_result"].isin(["Pass", "Fail", "Distinction"])].reset_index(drop=True), "final_result")
]

print("\nTask | Marks | Random Forest (acc / macro F1) | Logistic Regression (acc / macro F1) | Biggest class")

for name, df, target in tasks:

    for features, label in [(FOUR, "Four marks"), (TWO, "Test 1, Quiz 1")]:

        rf = evaluate(df, features, target, forest())
        lr = evaluate(df, features, target, logistic())

        print(f"{name} | {label} | {rf[0]:.1f}% / {rf[1]:.3f} | {lr[0]:.1f}% / {lr[1]:.3f} | {rf[2]:.1f}%")
