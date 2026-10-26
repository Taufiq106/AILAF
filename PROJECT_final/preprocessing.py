import pandas as pd
from pathlib import Path

# ============================================================
# AILAF - OULAD TO TEST/QUIZ FORMAT
# ============================================================
# OULAD assessments are mapped to the university format:
#   TMA (tutor marked)    -> test1, test2
#   CMA (computer marked) -> quiz1, quiz2
#
# Features : test1, quiz1, test2, quiz2
# Label    : average of the student's LATER assessments
#            (everything after the first two tests and quizzes)
#            below 60 = Beginner, below 80 = Intermediate, else Advanced
# ============================================================

DATA_PATH = Path("data/oulad")
OUTPUT_PATH = Path("data/processed")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_PATH / "ailaf_ml_dataset.csv"
BACKUP_FILE = OUTPUT_PATH / "ailaf_ml_dataset_previous.csv"


# ============================================================
# 1. BACKUP THE PREVIOUS DATASET
# ============================================================

if OUTPUT_FILE.exists() and not BACKUP_FILE.exists():
    OUTPUT_FILE.rename(BACKUP_FILE)
    print("Previous dataset backed up to:", BACKUP_FILE)


# ============================================================
# 2. LOAD DATA
# ============================================================

assessments = pd.read_csv(DATA_PATH / "assessments.csv")
student_assessment = pd.read_csv(DATA_PATH / "studentAssessment.csv")

print("assessments:", assessments.shape)
print("studentAssessment:", student_assessment.shape)


# ============================================================
# 3. KEEP TMA AND CMA ONLY
# ============================================================

assessments = assessments[
    assessments["assessment_type"].isin(["TMA", "CMA"])
]

# Keep only module presentations that have BOTH TMA and CMA
types_per_course = (
    assessments
    .groupby(["code_module", "code_presentation"])["assessment_type"]
    .nunique()
)

valid_courses = types_per_course[types_per_course == 2].index

assessments = assessments.set_index(
    ["code_module", "code_presentation"]
).loc[valid_courses].reset_index()

print("\nModule presentations with both TMA and CMA:")
print(valid_courses.tolist())


# ============================================================
# 4. MERGE SCORES AND REMOVE MISSING VALUES
# ============================================================

data = student_assessment.merge(
    assessments[
        [
            "id_assessment",
            "code_module",
            "code_presentation",
            "assessment_type",
            "date"
        ]
    ],
    on="id_assessment",
    how="inner"
)

data["score"] = pd.to_numeric(data["score"], errors="coerce")
data = data.dropna(subset=["score", "date"])

# Order by assessment date, then number each assessment per student
data = data.sort_values(
    ["id_student", "code_module", "code_presentation", "date"]
)

data["order"] = (
    data
    .groupby(
        ["id_student", "code_module", "code_presentation", "assessment_type"]
    )
    .cumcount() + 1
)


# ============================================================
# 5. FIRST TWO TESTS AND QUIZZES -> test1, quiz1, test2, quiz2
# ============================================================

early = data[data["order"] <= 2].copy()

early["column"] = early["assessment_type"].map(
    {"TMA": "test", "CMA": "quiz"}
) + early["order"].astype(str)

wide = early.pivot_table(
    index=["id_student", "code_module", "code_presentation"],
    columns="column",
    values="score",
    aggfunc="first"
).reset_index()

wide.columns.name = None

print("\nStudents before dropping incomplete rows:", len(wide))

wide = wide.dropna(subset=["test1", "test2", "quiz1", "quiz2"])

print("Students with all four scores:", len(wide))


# ============================================================
# 6. LATER ASSESSMENTS -> COMPETENCY LABEL
# ============================================================

later = (
    data[data["order"] > 2]
    .groupby(["id_student", "code_module", "code_presentation"])
    .agg(
        later_average_score=("score", "mean"),
        later_assessment_count=("score", "size")
    )
    .reset_index()
)

wide = wide.merge(
    later,
    on=["id_student", "code_module", "code_presentation"],
    how="inner"
)

print("Students with later assessments:", len(wide))


def assign_competency(score):
    if score < 60:
        return "Beginner"
    elif score < 80:
        return "Intermediate"
    else:
        return "Advanced"


wide["competency_level"] = wide["later_average_score"].apply(
    assign_competency
)


# ============================================================
# 7. REMOVE DUPLICATES AND SAVE
# ============================================================

wide = wide.drop_duplicates()

wide = wide[
    [
        "id_student",
        "code_module",
        "code_presentation",
        "test1",
        "quiz1",
        "test2",
        "quiz2",
        "later_average_score",
        "later_assessment_count",
        "competency_level"
    ]
]

wide.to_csv(OUTPUT_FILE, index=False)

print("\nFinal dataset shape:", wide.shape)
print("\nCompetency distribution:")
print(wide["competency_level"].value_counts())
print("\nSaved to:", OUTPUT_FILE)
