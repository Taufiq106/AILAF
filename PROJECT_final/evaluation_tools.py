import json
import pandas as pd
import streamlit as st

from pathlib import Path
from datetime import datetime

from security import safe_text, clean_id, WRITE_LOCK

# ============================================================
# AILAF - EVALUATION TOOLS (Chapter 3.5)
# ============================================================
# Model Performance : AI performance evaluation
# Pre/Post Test     : student evaluation (learning gain)
# Student Feedback  : student evaluation (questionnaire)
# Expert Review     : expert validation (questionnaire)
# Results Summary   : combined results
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

EVAL_DIR = BASE_DIR / "data" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_DIR / "data" / "processed" / "model_results"

PREPOST_QUESTIONS = EVAL_DIR / "prepost_questions.csv"
PREPOST_RESULTS = EVAL_DIR / "prepost_results.csv"
STUDENT_FEEDBACK = EVAL_DIR / "student_feedback.csv"
EXPERT_FEEDBACK = EVAL_DIR / "expert_feedback.csv"
QUESTION_REVIEW = EVAL_DIR / "question_review.csv"

CONSENT_TEXT = (
    "Participation is voluntary. Your responses are used only for this "
    "research (AILAF Final Year Project) and are reported as group results."
)

LIKERT = [1, 2, 3, 4, 5]
LIKERT_HELP = "1 = Strongly disagree, 5 = Strongly agree"

STUDENT_ITEMS = {
    "Usefulness": [
        "The learning pathway helped me know what to study next.",
        "The recommended resources matched my level.",
        "I would use this system to prepare for tests and quizzes."
    ],
    "Pathway clarity": [
        "I understood why these resources were recommended.",
        "The order of the resources was easy to follow."
    ],
    "Ease of use": [
        "The system was easy to use.",
        "Entering my marks was simple."
    ],
    "Satisfaction": [
        "I am satisfied with the system overall.",
        "I would recommend this system to other students."
    ]
}

EXPERT_ITEMS = {
    "Clarity": [
        "The framework components and their roles are clearly defined.",
        "The overall architecture is easy to understand."
    ],
    "AI-pedagogy integration": [
        "The framework combines AI prediction with CA, CBL and UDL in a meaningful way.",
        "The pedagogical decision rules are justified by the educational theories."
    ],
    "Educational relevance": [
        "The recommendations are relevant to the course topics and assessments.",
        "The framework addresses real needs in higher education."
    ],
    "Practical applicability": [
        "The framework could be used in a real course.",
        "Lecturers could maintain the knowledge base themselves."
    ],
    "Overall quality": [
        "Overall, the framework is of good quality."
    ]
}


def save_row(path, row):

    # text is cleaned before saving, and only one write happens at a time
    row = {
        key: (safe_text(value) if isinstance(value, str) else value)
        for key, value in row.items()
    }

    new_row = pd.DataFrame([row])

    with WRITE_LOCK:

        if path.exists():
            # keep the same columns as the existing file
            header = pd.read_csv(path, nrows=0).columns
            new_row = new_row.reindex(columns=header)
            new_row.to_csv(path, mode="a", header=False, index=False)
        else:
            new_row.to_csv(path, index=False)


def read_file(path):

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


# ============================================================
# 1. MODEL PERFORMANCE (AI PERFORMANCE EVALUATION)
# ============================================================

def render_model_performance():

    info_path = RESULTS_DIR / "final_model_info.json"
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    scenario_path = RESULTS_DIR / "model_comparison_by_scenario.csv"

    if not info_path.exists():
        st.warning("Model results not found. Run train_models.py first.")
        return

    with open(info_path) as file:
        info = json.load(file)

    main = info["scenarios"][info["main_scenario"]]

    a, b, c, d = st.columns(4)
    a.metric("Final model", info["algorithm"])
    b.metric("Students", f"{info['rows']:,}")
    c.metric("Accuracy (all four marks)", f"{main['accuracy'] * 100:.1f}%")
    d.metric("Macro F1", f"{main['macro_f1']:.3f}")

    st.caption(
        f"Label: {info['label_rule']}. {info['split']}; {info['weights']}."
    )

    if comparison_path.exists():

        st.subheader("Algorithm comparison (all four marks)")

        comparison = pd.read_csv(comparison_path)

        show = comparison[[
            "Model", "Accuracy", "Precision", "Recall", "F1", "Macro_F1",
            "CV_Macro_F1_Mean", "CV_Macro_F1_Std"
        ]].round(3)

        st.dataframe(show, hide_index=True)

        st.caption(
            "Precision, Recall and F1 are weighted averages on the test set. "
            "CV = 5-fold student-grouped cross-validation."
        )

    st.subheader("Accuracy by available marks (final model)")

    scenario_names = {
        "t1q1": "Test 1 and Quiz 1 only",
        "t1q1t2": "Test 1, Quiz 1 and Test 2",
        "t1q1q2": "Test 1, Quiz 1 and Quiz 2",
        "t1q1t2q2": "All four marks"
    }

    st.dataframe(
        pd.DataFrame([
            {
                "Marks available": scenario_names[key],
                "Accuracy": f"{value['accuracy'] * 100:.1f}%",
                "Macro F1": round(value["macro_f1"], 3)
            }
            for key, value in info["scenarios"].items()
        ]),
        hide_index=True
    )

    st.subheader("Per-class results (all four marks)")

    report = info["report"]

    st.dataframe(
        pd.DataFrame(
            {
                name: {
                    "Precision": round(report[name]["precision"], 3),
                    "Recall": round(report[name]["recall"], 3),
                    "F1-score": round(report[name]["f1-score"], 3),
                    "Support": int(report[name]["support"])
                }
                for name in info["classes"]
            }
        ).T
    )

    st.subheader("Confusion matrix (rows = actual, columns = predicted)")

    st.dataframe(
        pd.DataFrame(
            info["confusion_matrix"],
            index=["Actual " + c for c in info["classes"]],
            columns=["Pred " + c for c in info["classes"]]
        )
    )

    figure = RESULTS_DIR / "feature_importance.png"

    if figure.exists():
        st.image(str(figure), width=420)


# ============================================================
# 2. PRE-TEST AND POST-TEST
# ============================================================

def render_prepost():

    st.write(
        "Take the pre-test before using AILAF and the post-test after. "
        "The same questions are used both times so the scores can be compared."
    )

    if not PREPOST_QUESTIONS.exists():
        st.warning("Question file not found: " + str(PREPOST_QUESTIONS))
        return

    questions = pd.read_csv(PREPOST_QUESTIONS)

    with st.form("prepost_form"):

        student_id = st.text_input("Student ID (required)")

        phase = st.radio("Test", ["Pre-test", "Post-test"], horizontal=True)

        answers = []

        for number, q in questions.iterrows():

            options = [
                f"A. {q['option_a']}",
                f"B. {q['option_b']}",
                f"C. {q['option_c']}",
                f"D. {q['option_d']}"
            ]

            answers.append(
                st.radio(
                    f"Q{number + 1}. {q['question']}",
                    options,
                    index=None,
                    key=f"prepost_q_{number}"
                )
            )

        consent = st.checkbox(CONSENT_TEXT + " I agree.")

        submitted = st.form_submit_button("Submit Test", type="primary")

    if submitted:

        student_id = clean_id(student_id)

        if not student_id:
            st.error("Please enter a valid Student ID (3 to 20 letters, numbers, - or _).")
            return

        if not consent:
            st.error("Please tick the consent box.")
            return

        score = 0

        for number, q in questions.iterrows():

            choice = answers[number]

            if choice is not None and choice[0] == q["correct_option"]:
                score += 1

        save_row(PREPOST_RESULTS, {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_id": student_id,
            "phase": phase,
            "score": score,
            "total": len(questions)
        })

        st.success(f"{phase} submitted. Score: {score} / {len(questions)}")


# ============================================================
# 3 AND 4. QUESTIONNAIRES
# ============================================================

def render_questionnaire(title, items, path, key_prefix, who_label, who_options=None, extra_label=None):

    st.write(f"Rate each statement. {LIKERT_HELP}.")

    with st.form(f"{key_prefix}_form"):

        if who_options:
            who = st.selectbox(who_label, who_options)
            name = st.text_input("Name (optional)")
        else:
            who = ""
            name = st.text_input(who_label)

        extra = st.text_input(extra_label) if extra_label else ""

        answers = {}

        for group, statements in items.items():

            st.markdown(f"**{group}**")

            for number, statement in enumerate(statements):

                answers[(group, number)] = st.radio(
                    statement,
                    LIKERT,
                    index=None,
                    horizontal=True,
                    key=f"{key_prefix}_{group}_{number}"
                )

        comment = st.text_area("Comments or suggestions (optional)")

        consent = st.checkbox(CONSENT_TEXT + " I agree.")

        submitted = st.form_submit_button("Submit", type="primary")

    if submitted:

        if not consent:
            st.error("Please tick the consent box.")
            return

        if any(value is None for value in answers.values()):
            st.error("Please answer every statement.")
            return

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "respondent": safe_text(name, 100),
            "role": who
        }

        if extra_label:
            row["experience"] = safe_text(extra, 50)

        for (group, number), value in answers.items():
            row[f"{group} {number + 1}"] = value

        row["comment"] = comment.strip()

        save_row(path, row)

        st.success("Thank you. Your response was saved.")


def render_student_feedback():

    render_questionnaire(
        "Student Feedback", STUDENT_ITEMS, STUDENT_FEEDBACK,
        "student", "Student ID (optional)"
    )


def render_expert_review():

    render_questionnaire(
        "Expert Review", EXPERT_ITEMS, EXPERT_FEEDBACK,
        "expert", "Your role",
        ["Lecturer", "Educational technology expert", "Other academic"],
        "Years of teaching experience (optional)"
    )


# ============================================================
# 4B. LECTURER REVIEW OF THE TEST QUESTIONS
# ============================================================
# Shows the correct answers, so this tab appears only in the
# lecturer version (app_ekb.py), never in the student version.
# ============================================================

def render_question_review():

    st.write(
        "Please check each pre-test and post-test question. "
        "The correct answer is shown for you."
    )

    st.warning("This page shows the answer key. It is for lecturers and the researcher only.")

    if not PREPOST_QUESTIONS.exists():
        st.warning("Question file not found: " + str(PREPOST_QUESTIONS))
        return

    questions = pd.read_csv(PREPOST_QUESTIONS)

    with st.form("question_review_form"):

        reviewer = st.text_input("Reviewer name")

        for number, q in questions.iterrows():

            correct = q["correct_option"]

            st.markdown(f"**Q{number + 1}.** {q['question']}")

            st.caption(
                f"A. {q['option_a']}   |   B. {q['option_b']}   |   "
                f"C. {q['option_c']}   |   D. {q['option_d']}"
            )

            st.caption(f"Correct answer: {correct}. {q['option_' + correct.lower()]}")

            col1, col2 = st.columns(2)

            with col1:
                st.radio(
                    "Answer correct?", ["Yes", "No"],
                    index=None, horizontal=True, key=f"review_ok_{number}"
                )

            with col2:
                st.radio(
                    "Matches the course?", ["Yes", "No"],
                    index=None, horizontal=True, key=f"review_match_{number}"
                )

            st.text_input(
                "Change needed or comment (optional)",
                key=f"review_note_{number}"
            )

            st.divider()

        overall = st.radio(
            "Overall",
            ["Approved as written", "Approved with the changes above", "Needs a major revision"],
            index=None
        )

        submitted = st.form_submit_button("Submit review", type="primary")

    if submitted:

        name = safe_text(reviewer, 100)

        if not name:
            st.error("Please enter the reviewer name.")
            return

        ok = [st.session_state.get(f"review_ok_{n}") for n in range(len(questions))]
        match = [st.session_state.get(f"review_match_{n}") for n in range(len(questions))]

        if any(v is None for v in ok + match) or overall is None:
            st.error("Please answer every question and choose the overall result.")
            return

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for number, q in questions.iterrows():

            save_row(QUESTION_REVIEW, {
                "timestamp": stamp,
                "reviewer": name,
                "question_id": q["question_id"],
                "topic": q["topic"],
                "answer_correct": ok[number],
                "matches_course": match[number],
                "comment": safe_text(st.session_state.get(f"review_note_{number}", ""), 500),
                "overall": overall
            })

        st.success("Thank you. Your review was saved.")


# ============================================================
# 5. RESULTS SUMMARY
# ============================================================

def group_means(data, items):

    rows = []

    for group, statements in items.items():

        columns = [f"{group} {n + 1}" for n in range(len(statements))]
        columns = [c for c in columns if c in data.columns]

        if columns:
            rows.append({
                "Category": group,
                "Mean (1-5)": round(data[columns].mean().mean(), 2)
            })

    return pd.DataFrame(rows)


def render_results_summary():

    prepost = read_file(PREPOST_RESULTS)
    students = read_file(STUDENT_FEEDBACK)
    experts = read_file(EXPERT_FEEDBACK)

    a, b, c = st.columns(3)
    a.metric("Test submissions", len(prepost))
    b.metric("Student questionnaires", len(students))
    c.metric("Expert questionnaires", len(experts))

    st.subheader("Learning gain (pre-test to post-test)")

    if prepost.empty:
        st.info("No test results yet.")
    else:

        # keep each student's latest score for each phase
        latest = prepost.sort_values("timestamp").drop_duplicates(
            ["student_id", "phase"], keep="last"
        )

        table = latest.pivot(index="student_id", columns="phase", values="score")

        if "Pre-test" in table.columns and "Post-test" in table.columns:

            paired = table.dropna(subset=["Pre-test", "Post-test"])

            if paired.empty:
                st.info("No student has both a pre-test and a post-test yet.")
            else:

                total = int(prepost["total"].iloc[0])

                x, y, z = st.columns(3)
                x.metric("Students with both tests", len(paired))
                y.metric("Mean pre-test", f"{paired['Pre-test'].mean():.1f} / {total}")
                z.metric(
                    "Mean post-test",
                    f"{paired['Post-test'].mean():.1f} / {total}",
                    f"{(paired['Post-test'] - paired['Pre-test']).mean():+.1f}"
                )

                st.dataframe(paired)

        else:
            st.info("Both a pre-test and a post-test are needed to show learning gain.")

    st.subheader("Student questionnaire")

    if students.empty:
        st.info("No student responses yet.")
    else:
        st.dataframe(group_means(students, STUDENT_ITEMS), hide_index=True)

    st.subheader("Expert questionnaire")

    if experts.empty:
        st.info("No expert responses yet.")
    else:
        st.dataframe(group_means(experts, EXPERT_ITEMS), hide_index=True)

    reviews = read_file(QUESTION_REVIEW)

    st.subheader("Lecturer review of test questions")

    if reviews.empty:
        st.info("No question reviews yet.")
    else:

        summary = reviews.groupby(["question_id", "topic"]).agg(
            Reviews=("reviewer", "count"),
            Not_correct=("answer_correct", lambda x: int((x == "No").sum())),
            Not_matching_course=("matches_course", lambda x: int((x == "No").sum()))
        ).reset_index()

        flagged = summary[(summary["Not_correct"] > 0) | (summary["Not_matching_course"] > 0)]

        st.write(f"{reviews['reviewer'].nunique()} reviewer(s). Questions flagged: {len(flagged)}.")

        st.dataframe(summary, hide_index=True)

        comments = reviews[reviews["comment"].fillna("").astype(str).str.strip() != ""]

        if not comments.empty:
            st.dataframe(
                comments[["reviewer", "question_id", "comment"]],
                hide_index=True
            )

    st.subheader("Download raw data")

    for label, data, name in [
        ("Test results", prepost, "prepost_results.csv"),
        ("Student feedback", students, "student_feedback.csv"),
        ("Expert feedback", experts, "expert_feedback.csv"),
        ("Question review", reviews, "question_review.csv")
    ]:
        if not data.empty:
            st.download_button(
                label,
                data=data.to_csv(index=False).encode("utf-8"),
                file_name=name,
                key="download_" + name
            )


# ============================================================
# EVALUATION PAGE
# ============================================================

def render_evaluation_page(include_review=False):

    st.title("Evaluation")

    st.write(
        "Tools used to validate AILAF: AI performance, student evaluation "
        "and expert validation."
    )

    names = [
        "Model Performance",
        "Pre/Post Test",
        "Student Feedback",
        "Expert Review"
    ]

    if include_review:
        names.append("Question Review")

    names.append("Results Summary")

    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Model Performance"]:
        render_model_performance()

    with tabs["Pre/Post Test"]:
        render_prepost()

    with tabs["Student Feedback"]:
        render_student_feedback()

    with tabs["Expert Review"]:
        render_expert_review()

    if include_review:
        with tabs["Question Review"]:
            render_question_review()

    with tabs["Results Summary"]:
        render_results_summary()
