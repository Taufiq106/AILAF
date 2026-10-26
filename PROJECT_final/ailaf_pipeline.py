import pandas as pd
from pathlib import Path

from predictor import predict_competency
from pdm import PedagogicalDecisionModule
from recommendation_engine import (
    RecommendationEngine,
    display_recommendations
)

# ============================================================
# AILAF PIPELINE (command line)
# ============================================================
# marks -> competency prediction -> Pedagogical Decision Module
# -> recommendation engine -> learning pathway
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

KNOWLEDGE_BASE_PATH = (
    PROJECT_DIR
    / "data"
    / "knowledge_base"
    / "learning_resources.csv"
)

PILOT_COURSE = "TFB1013"

# Example student (test2 and quiz2 are optional: remove them or set None)
student = {
    "test1": 62,
    "quiz1": 78,
    "test2": 55,
    "quiz2": None
}

# ============================================================
# 2. PREDICT COMPETENCY
# ============================================================

prediction = predict_competency(student)

print("\nMarks used:", prediction["features_used"])
print("Predicted competency (model):", prediction["competency_level"])
print("Probabilities:", {k: round(v, 3) for k, v in prediction["probabilities"].items()})

# ============================================================
# 3. PEDAGOGICAL DECISION
# ============================================================

marks = {}

names = {"test1": "Test 1", "quiz1": "Quiz 1", "test2": "Test 2", "quiz2": "Quiz 2"}

for key, name in names.items():

    if student.get(key) is not None:
        marks[name] = student[key]

pdm = PedagogicalDecisionModule()

decision = pdm.decide(prediction["competency_level"], marks)

print("\nCompetency used:", decision.competency_level)

for line in decision.explanation:
    print(" -", line)

# ============================================================
# 4. RECOMMENDATIONS
# ============================================================

engine = RecommendationEngine(KNOWLEDGE_BASE_PATH)

recommendations = engine.recommend(
    competency_level=decision.competency_level,
    learning_strategy=decision.learning_strategy,
    difficulty_level=decision.difficulty_level,
    course=PILOT_COURSE,
    resource_types=decision.resource_formats,
    priority_assessment=decision.weakest_assessment
)

display_recommendations(recommendations.head(10))

print(f"({len(recommendations)} resources in total)")
