import json
import joblib
import pandas as pd
from pathlib import Path

# ============================================================
# AILAF - COMPETENCY PREDICTOR
# ============================================================
# Test 1 and Quiz 1 are required.
# Test 2 and Quiz 2 are optional.
# The model trained for the available marks is used.
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_results"
)

INFO_PATH = RESULTS_DIR / "final_model_info.json"


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

REQUIRED_FEATURES = ["test1", "quiz1"]

OPTIONAL_FEATURES = ["test2", "quiz2"]

FEATURE_COLUMNS = REQUIRED_FEATURES + OPTIONAL_FEATURES

SCENARIO_FILES = {
    "t1q1": ["test1", "quiz1"],
    "t1q1t2": ["test1", "quiz1", "test2"],
    "t1q1q2": ["test1", "quiz1", "quiz2"],
    "t1q1t2q2": ["test1", "quiz1", "test2", "quiz2"]
}

FEATURE_LABELS = {
    "test1": "Test 1",
    "quiz1": "Quiz 1",
    "test2": "Test 2",
    "quiz2": "Quiz 2"
}


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

print("Loading AILAF prediction models...")

if not INFO_PATH.exists():
    raise FileNotFoundError(
        f"Model information not found: {INFO_PATH}. Run train_models.py first."
    )

with open(INFO_PATH) as file:
    MODEL_INFO = json.load(file)

CLASSES = MODEL_INFO["classes"]

models = {}

for scenario in SCENARIO_FILES:

    path = RESULTS_DIR / f"final_model_{scenario}.joblib"

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    models[scenario] = joblib.load(path)

print(f"{MODEL_INFO['algorithm']} models loaded successfully.")


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def is_missing(value):

    return value is None or pd.isna(value)


def choose_scenario(test2, quiz2):

    if not is_missing(test2) and not is_missing(quiz2):
        return "t1q1t2q2"

    if not is_missing(test2):
        return "t1q1t2"

    if not is_missing(quiz2):
        return "t1q1q2"

    return "t1q1"


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_competency(student_data):
    """
    Predict a student's competency level.

    Input:
        Dictionary or one-row DataFrame with test1 and quiz1.
        test2 and quiz2 are optional (leave out or use None).

    Output:
        Dictionary with the predicted competency, class
        probabilities and which marks were used.
    """

    if isinstance(student_data, pd.DataFrame):

        if len(student_data) != 1:
            raise ValueError("Please provide exactly one student record.")

        student_data = student_data.iloc[0].to_dict()

    elif not isinstance(student_data, dict):

        raise TypeError("Input must be a dictionary or pandas DataFrame.")

    for feature in REQUIRED_FEATURES:

        if is_missing(student_data.get(feature)):
            raise ValueError(f"Missing required feature: {feature}")

    scenario = choose_scenario(
        student_data.get("test2"),
        student_data.get("quiz2")
    )

    features = SCENARIO_FILES[scenario]

    row = pd.DataFrame(
        [{feature: float(student_data[feature]) for feature in features}]
    )[features]

    probabilities = models[scenario].predict_proba(row)[0]

    best_index = int(probabilities.argmax())

    return {
        "competency_level": CLASSES[best_index],
        "probabilities": {
            CLASSES[i]: float(probabilities[i])
            for i in range(len(CLASSES))
        },
        "scenario": scenario,
        "features_used": features,
        "model_accuracy": MODEL_INFO["scenarios"][scenario]["accuracy"],
        "algorithm": MODEL_INFO["algorithm"]
    }


# ============================================================
# 6. QUICK TEST
# ============================================================

if __name__ == "__main__":

    examples = [
        {"test1": 85, "quiz1": 90},
        {"test1": 85, "quiz1": 90, "test2": 80, "quiz2": 88},
        {"test1": 45, "quiz1": 55, "test2": 40},
        {"test1": 60, "quiz1": 70, "quiz2": 65}
    ]

    for example in examples:

        result = predict_competency(example)

        print(example)
        print("  ->", result["competency_level"], "| marks used:", result["features_used"])
