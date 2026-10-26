import sys
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# AILAF - TEMPORARY FAKE DATA (for testing and demonstration only)
# ============================================================
# All fake rows use IDs that start with FAKE, so they can be removed.
#
#   python fake_data.py add 40      create fake data for 40 students
#   python fake_data.py remove      remove every fake row
#
# WARNING: this data is invented. Never report it as real results,
# and run "remove" before you collect real data.
# ============================================================

EVAL = Path("data/evaluation")
FAKE_DIR = Path("data/fake")
EVAL.mkdir(parents=True, exist_ok=True)

FILES = {
    "prepost": (EVAL / "prepost_results.csv", "student_id"),
    "student": (EVAL / "student_feedback.csv", "respondent"),
    "expert": (EVAL / "expert_feedback.csv", "respondent"),
    "review": (EVAL / "question_review.csv", "reviewer")
}

STUDENT_COLUMNS = [
    "Usefulness 1", "Usefulness 2", "Usefulness 3",
    "Pathway clarity 1", "Pathway clarity 2",
    "Ease of use 1", "Ease of use 2",
    "Satisfaction 1", "Satisfaction 2"
]

EXPERT_COLUMNS = [
    "Clarity 1", "Clarity 2",
    "AI-pedagogy integration 1", "AI-pedagogy integration 2",
    "Educational relevance 1", "Educational relevance 2",
    "Practical applicability 1", "Practical applicability 2",
    "Overall quality 1"
]

rng = np.random.default_rng(2026)


def clip(value, low, high):
    return int(max(low, min(high, round(value))))


def level_from_average(average):
    if average < 60:
        return "Beginner"
    elif average < 80:
        return "Intermediate"
    return "Advanced"


def stamp(base, minutes):
    return (base + timedelta(minutes=int(minutes))).strftime("%Y-%m-%d %H:%M:%S")


def append(path, new_rows):
    new = pd.DataFrame(new_rows)

    if path.exists():
        old_columns = pd.read_csv(path, nrows=0).columns
        new = new.reindex(columns=old_columns)
        new.to_csv(path, mode="a", header=False, index=False)
    else:
        new.to_csv(path, index=False)


def add(n):

    base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) - timedelta(days=2)

    # ---------- fake students: mostly Advanced ----------
    levels = rng.choice(["Advanced", "Intermediate", "Beginner"], size=n, p=[0.70, 0.22, 0.08])

    centre = {"Advanced": (86, 93, 88, 94), "Intermediate": (70, 80, 72, 82), "Beginner": (52, 62, 50, 60)}

    students = []
    questions = pd.read_csv(EVAL / "prepost_questions.csv")
    total = len(questions)

    for i, level in enumerate(levels, start=1):

        marks = [clip(rng.normal(c, 6), 0, 100) for c in centre[level]]
        average = sum(marks) / 4

        # test scores: stronger students score higher, and everyone improves
        pre_mean = {"Advanced": 14, "Intermediate": 11, "Beginner": 8}[level]
        gain = {"Advanced": 3, "Intermediate": 4.5, "Beginner": 5}[level]

        pre = clip(rng.normal(pre_mean, 2.5), 0, total)
        post = clip(pre + rng.normal(gain, 2), 0, total)

        students.append({
            "student_id": f"FAKE{i:03d}",
            "test1": marks[0], "quiz1": marks[1], "test2": marks[2], "quiz2": marks[3],
            "average": round(average, 1),
            "level": level_from_average(average),
            "pre": pre, "post": post
        })

    fake = pd.DataFrame(students)

    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    fake[["student_id", "test1", "quiz1", "test2", "quiz2", "average", "level"]].to_csv(
        FAKE_DIR / "fake_marks.csv", index=False
    )

    # ---------- pre-test and post-test ----------
    rows = []
    for k, s in fake.iterrows():
        rows.append({"timestamp": stamp(base, k * 3), "student_id": s.student_id,
                     "phase": "Pre-test", "score": s.pre, "total": total})
        rows.append({"timestamp": stamp(base, 600 + k * 3), "student_id": s.student_id,
                     "phase": "Post-test", "score": s.post, "total": total})
    append(FILES["prepost"][0], rows)

    # ---------- student questionnaire ----------
    quality = {"Advanced": 4.4, "Intermediate": 4.0, "Beginner": 3.6}
    rows = []
    for k, s in fake.iterrows():
        row = {"timestamp": stamp(base, 700 + k * 2), "respondent": s.student_id, "role": ""}
        for col in STUDENT_COLUMNS:
            row[col] = clip(rng.normal(quality[s.level], 0.7), 1, 5)
        row["comment"] = "(fake) " + str(rng.choice(["Clear pathway.", "Good videos.", "More exercises please.", "", "Easy to use."]))
        rows.append(row)
    append(FILES["student"][0], rows)

    # ---------- expert questionnaire (5 fake experts) ----------
    roles = ["Lecturer", "Lecturer", "Educational technology expert", "Lecturer", "Other academic"]
    rows = []
    for e in range(5):
        row = {"timestamp": stamp(base, 800 + e * 20), "respondent": f"FAKE-Expert{e + 1}",
               "role": roles[e], "experience": str(int(rng.integers(4, 20)))}
        for col in EXPERT_COLUMNS:
            row[col] = clip(rng.normal(4.3, 0.6), 1, 5)
        row["comment"] = "(fake) " + str(rng.choice(["Well structured.", "Add more videos.", "Check the Beginner results.", ""]))
        rows.append(row)
    append(FILES["expert"][0], rows)

    # ---------- question review (2 fake reviewers) ----------
    rows = []
    for r in range(2):
        for k, q in questions.iterrows():
            flagged = (k in (3, 11)) and r == 0
            rows.append({
                "timestamp": stamp(base, 900 + r * 30), "reviewer": f"FAKE-Reviewer{r + 1}",
                "question_id": q.question_id, "topic": q.topic,
                "answer_correct": "Yes", "matches_course": "No" if flagged else "Yes",
                "comment": "(fake) Check wording." if flagged else "",
                "overall": "Approved with the changes above" if r == 0 else "Approved as written"
            })
    append(FILES["review"][0], rows)

    print(f"Fake data added for {n} students, 5 experts and 2 reviewers.")
    print("Fake student levels:", fake["level"].value_counts().to_dict())
    print(f"Mean pre-test {fake.pre.mean():.1f} / {total}, mean post-test {fake.post.mean():.1f} / {total}")
    print("Marks saved in data/fake/fake_marks.csv")
    print("REMEMBER: this is invented data. Run  python fake_data.py remove  before real data collection.")


def remove():

    for name, (path, column) in FILES.items():

        if not path.exists():
            continue

        data = pd.read_csv(path)
        keep = data[~data[column].astype(str).str.startswith("FAKE")]

        removed = len(data) - len(keep)

        if len(keep) == 0:
            path.unlink()
        else:
            keep.to_csv(path, index=False)

        print(f"{path.name}: removed {removed} fake rows")

    if FAKE_DIR.exists():
        shutil.rmtree(FAKE_DIR)
        print("Removed data/fake/")


if __name__ == "__main__":

    command = sys.argv[1] if len(sys.argv) > 1 else ""

    if command == "add":
        add(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif command == "remove":
        remove()
    else:
        print("Use:  python fake_data.py add 40   or   python fake_data.py remove")
