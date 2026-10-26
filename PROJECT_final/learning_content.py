import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from security import safe_text, clean_id, WRITE_LOCK

QUIZ_PATH = Path("data/knowledge_base/quizzes.csv")
RESULTS_PATH = Path("data/quiz_results.csv")


def load_quizzes():
    if not QUIZ_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(QUIZ_PATH)


def save_quiz_result(course, topic, level, student_id, score, total):
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "student_id": student_id,
        "course": course,
        "topic": topic,
        "competency_level": level,
        "score": score,
        "total": total
    }])

    with WRITE_LOCK:

        if RESULTS_PATH.exists():
            row.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
        else:
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            row.to_csv(RESULTS_PATH, index=False)


def render_quiz(resource):
    key = str(resource["resource_id"])
    quizzes = load_quizzes()

    if quizzes.empty:
        st.info("Quiz file not found.")
        return

    questions = quizzes[
        (quizzes["course"] == resource["course"])
        & (quizzes["topic"] == resource["topic"])
        & (quizzes["competency_level"] == resource["competency_level"])
    ].reset_index(drop=True)

    if questions.empty:
        st.info("No quiz questions available for this resource yet.")
        return

    with st.form(f"quiz_form_{key}"):

        student_id = st.text_input(
            "Student ID (optional)",
            key=f"quiz_sid_{key}"
        )

        st.caption("Your score is saved for this research. Student ID is optional.")

        answers = []

        for number, q in questions.iterrows():

            options = [
                f"A. {q['option_a']}",
                f"B. {q['option_b']}",
                f"C. {q['option_c']}",
                f"D. {q['option_d']}"
            ]

            choice = st.radio(
                f"Q{number + 1}. {q['question']}",
                options,
                index=None,
                key=f"quiz_q_{key}_{number}"
            )

            answers.append(choice)

        submitted = st.form_submit_button("Submit Quiz")

    if submitted:

        score = 0

        for number, q in questions.iterrows():

            choice = answers[number]
            correct_letter = q["correct_option"]
            correct_text = q[f"option_{correct_letter.lower()}"]

            if choice is not None and choice[0] == correct_letter:
                score += 1
                st.success(f"Q{number + 1}: Correct. {q['explanation']}")
            elif choice is None:
                st.warning(
                    f"Q{number + 1}: Not answered. "
                    f"Correct answer: {correct_letter}. {correct_text}. "
                    f"{q['explanation']}"
                )
            else:
                st.error(
                    f"Q{number + 1}: Incorrect. "
                    f"Correct answer: {correct_letter}. {correct_text}. "
                    f"{q['explanation']}"
                )

        st.markdown(f"**Score: {score} / {len(questions)}**")

        checked_id = clean_id(student_id)

        if checked_id is None:
            st.warning("Student ID not saved: use 3 to 20 letters, numbers, - or _ only.")
            checked_id = ""

        save_quiz_result(
            resource["course"],
            resource["topic"],
            resource["competency_level"],
            checked_id,
            score,
            len(questions)
        )


def render_resource_content(resource):

    has_url = (
        "resource_url" in resource.index
        and pd.notna(resource["resource_url"])
        and str(resource["resource_url"]).strip() != ""
    )

    content = ""
    if "content" in resource.index and pd.notna(resource["content"]):
        content = str(resource["content"]).strip()

    if resource["resource_type"] == "Quiz" and not has_url:
        with st.expander("Take the Quiz"):
            render_quiz(resource)
    elif content:
        with st.expander("View Content"):
            st.markdown(content)
