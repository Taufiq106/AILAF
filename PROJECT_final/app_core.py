import streamlit as st
import pandas as pd

from pathlib import Path

from pdm import PedagogicalDecisionModule
from recommendation_engine import RecommendationEngine
from learning_content import render_resource_content
from evaluation_tools import render_evaluation_page
from security import valid_url

# ============================================================
# AILAF - SHARED APPLICATION CODE
# ============================================================
# app.py     : student app (Home, Learning Dashboard, Evaluation, About)
# app_ekb.py : same app + EKB Management page for lecturers
# ============================================================


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_BASE_PATH = (
    BASE_DIR
    / "data"
    / "knowledge_base"
    / "learning_resources.csv"
)

PILOT_COURSE = "TFB1013"

COURSE_NAMES = {
    "TFB1013": "Computer Programming (C++)",
    "TFB2063": "Data Science",
    "TFB2033": "Human Computer Interaction"
}

MARK_NAMES = {
    "test1": "Test 1",
    "quiz1": "Quiz 1",
    "test2": "Test 2",
    "quiz2": "Quiz 2"
}


# ============================================================
# 2. STYLING
# ============================================================

STYLE = """
<style>

.stApp {
    background-color: #F7F9F8;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background-color: #F0F5F2;
    border-right: 1px solid #E1EAE4;
}

h1, h2, h3 {
    color: #183D32;
}

div.stButton > button[kind="primary"] {
    background-color: #247452;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 600;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #195A3E;
    color: white;
}

</style>
"""


# ============================================================
# 3. LOAD SYSTEM COMPONENTS
# ============================================================

@st.cache_resource
def load_pdm():

    return PedagogicalDecisionModule()


@st.cache_resource
def load_recommendation_engine():

    return RecommendationEngine(
        KNOWLEDGE_BASE_PATH
    )


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def section_heading(title, description=None):

    st.subheader(title)

    if description:

        st.caption(description)


def get_competency_color(level):

    colors = {
        "Beginner": "#C28B24",
        "Intermediate": "#347DB5",
        "Advanced": "#247452"
    }

    return colors.get(level, "#247452")


# ============================================================
# 5. HOME PAGE
# ============================================================

def render_home():

    st.title("AILAF")

    st.subheader("Adaptive Intelligent Learning Alignment Framework")

    st.write(
        "AILAF predicts your competency from your Test and Quiz marks, then "
        "builds a learning pathway that follows three educational theories."
    )

    st.divider()

    st.markdown("### How it works")

    steps = [
        (
            "1. Enter your marks",
            "Test 1 and Quiz 1 are required. Test 2 and Quiz 2 are optional "
            "and make the prediction more accurate."
        ),
        (
            "2. Competency is decided",
            "A trained model predicts your later performance. Your own marks "
            "also show your current level. The higher one is used (CBL)."
        ),
        (
            "3. Pedagogical rules are applied",
            "The system chooses difficulty (CBL), puts topics linked to your "
            "weakest assessment first (CA) and offers several formats (UDL)."
        ),
        (
            "4. Follow your pathway",
            "Open the resources, read the notes, try the exercises and take "
            "the quizzes."
        )
    ]

    for title, description in steps:

        with st.container(border=True):

            st.markdown(f"**{title}**")
            st.write(description)

    st.divider()

    st.markdown("### Educational theories")

    a, b, c = st.columns(3)

    with a:
        with st.container(border=True):
            st.markdown("**Constructive Alignment (CA)**")
            st.write(
                "Resources are linked to the test or quiz they prepare you for."
            )

    with b:
        with st.container(border=True):
            st.markdown("**Competency-Based Learning (CBL)**")
            st.write(
                "Your competency level decides the difficulty of the resources."
            )

    with c:
        with st.container(border=True):
            st.markdown("**Universal Design for Learning (UDL)**")
            st.write(
                "Videos, notes, examples, exercises and quizzes. You can "
                "choose the formats you prefer."
            )

    st.info(
        "This is a research prototype. The model was trained on OULAD "
        "assessment data converted to test and quiz format. Results are "
        "predictions, not official grades."
    )

    st.caption(
        "Quiz scores, test results and questionnaires are saved for this "
        "research only. Giving a Student ID is optional unless stated."
    )


# ============================================================
# 6. LEARNING DASHBOARD
# ============================================================

def render_dashboard():

    import predictor

    pdm = load_pdm()
    engine = load_recommendation_engine()

    knowledge_base = pd.read_csv(KNOWLEDGE_BASE_PATH)

    st.title("Welcome to Your Learning Dashboard")

    st.write(
        "Enter your marks to see your predicted competency and a learning "
        "pathway aligned with your assessments."
    )

    st.divider()

    # --------------------------------------------------------
    # LEARNING PROFILE
    # --------------------------------------------------------

    section_heading(
        "Learning Profile",
        "Test 1 and Quiz 1 are required. Tick Test 2 or Quiz 2 if you have them."
    )

    with st.container(border=True):

        col1, col2 = st.columns(2)

        with col1:

            test1 = st.number_input("Test 1 Mark (%)", 0.0, 100.0, 70.0, 1.0)
            quiz1 = st.number_input("Quiz 1 Mark (%)", 0.0, 100.0, 70.0, 1.0)

        with col2:

            has_test2 = st.checkbox("I have a Test 2 mark")
            test2 = st.number_input(
                "Test 2 Mark (%)", 0.0, 100.0, 70.0, 1.0,
                disabled=not has_test2
            )

            has_quiz2 = st.checkbox("I have a Quiz 2 mark")
            quiz2 = st.number_input(
                "Quiz 2 Mark (%)", 0.0, 100.0, 70.0, 1.0,
                disabled=not has_quiz2
            )

        courses = sorted(knowledge_base["course"].dropna().unique().tolist())

        selected_course = st.selectbox(
            "Course",
            courses,
            index=courses.index(PILOT_COURSE) if PILOT_COURSE in courses else 0,
            format_func=lambda code: f"{code}: {COURSE_NAMES.get(code, code)}"
        )

        course_topics = sorted(
            knowledge_base[
                knowledge_base["course"] == selected_course
            ]["topic"].dropna().unique().tolist()
        )

        formats = [
            f for f in [
                "Video", "Lecture Notes", "Worked Example",
                "Practice Exercise", "Quiz", "Case Study", "Project"
            ]
            if f in knowledge_base["resource_type"].unique()
        ]

        topic_col, format_col = st.columns(2)

        with topic_col:

            selected_topic = st.selectbox(
                "Topic",
                ["All Topics"] + course_topics
            )

        with format_col:

            preferred_formats = st.multiselect(
                "Preferred formats (optional)",
                formats,
                help="Leave empty to see all formats suggested for your level."
            )

        generate = st.button("Generate Learning Pathway", type="primary")

    if generate:

        student = {"test1": test1, "quiz1": quiz1}
        marks = {"Test 1": test1, "Quiz 1": quiz1}

        if has_test2:
            student["test2"] = test2
            marks["Test 2"] = test2

        if has_quiz2:
            student["quiz2"] = quiz2
            marks["Quiz 2"] = quiz2

        prediction = predictor.predict_competency(student)

        decision = pdm.decide(prediction["competency_level"], marks)

        topic_filter = None if selected_topic == "All Topics" else selected_topic

        notes = []

        formats_to_use = preferred_formats if preferred_formats else decision.resource_formats

        recommendations = engine.recommend(
            competency_level=decision.competency_level,
            learning_strategy=decision.learning_strategy,
            difficulty_level=decision.difficulty_level,
            course=selected_course,
            topic=topic_filter,
            resource_types=formats_to_use,
            priority_assessment=decision.weakest_assessment
        )

        if recommendations.empty and preferred_formats:

            notes.append(
                "No resources in your preferred formats for this level and topic. "
                "Showing the other formats suggested for your level."
            )

            recommendations = engine.recommend(
                competency_level=decision.competency_level,
                learning_strategy=decision.learning_strategy,
                difficulty_level=decision.difficulty_level,
                course=selected_course,
                topic=topic_filter,
                resource_types=decision.resource_formats,
                priority_assessment=decision.weakest_assessment
            )

        if recommendations.empty:

            notes.append(
                "No exact match found. Showing other resources for this course."
            )

            recommendations = engine.recommend(
                competency_level=None,
                course=selected_course,
                topic=topic_filter,
                priority_assessment=decision.weakest_assessment
            )

        st.session_state["result"] = {
            "prediction": prediction,
            "decision": decision,
            "recommendations": recommendations,
            "notes": notes,
            "marks": marks
        }

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if "result" not in st.session_state:
        return

    result = st.session_state["result"]

    prediction = result["prediction"]
    decision = result["decision"]
    recommendations = result["recommendations"]

    st.divider()

    section_heading("Competency Result")

    color = get_competency_color(decision.competency_level)

    st.markdown(
        f"<h2 style='color:{color}'>{decision.competency_level}</h2>",
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Current level (from marks)",
        decision.current_level,
        f"average {decision.average_mark:.1f}%",
        delta_color="off"
    )
    m2.metric("Predicted later level (model)", decision.predicted_level)
    m3.metric("Used for recommendations", decision.competency_level)

    st.caption(
        f"Prediction by {prediction['algorithm']} using "
        + ", ".join(MARK_NAMES[f] for f in prediction["features_used"])
        + f". Model accuracy for this input type: "
        f"{prediction['model_accuracy'] * 100:.1f}%."
    )

    st.bar_chart(pd.Series(prediction["probabilities"], name="Probability"))

    section_heading("Why this pathway?")

    with st.container(border=True):

        for line in decision.explanation:
            st.write(line)

    st.divider()

    section_heading("Pedagogical Focus")

    for focus in decision.pedagogical_focus:
        st.write("- " + focus)

    st.divider()

    section_heading(
        "Recommended Learning Resources",
        f"{len(recommendations)} resources, in pathway order."
    )

    for note in result["notes"]:
        st.warning(note)

    if recommendations.empty:
        st.info("No resources found.")
        return

    current_topic = None

    for number, (_, resource) in enumerate(recommendations.iterrows(), start=1):

        if resource["topic"] != current_topic:

            current_topic = resource["topic"]

            st.markdown(f"#### {current_topic}")

            assessment = resource.get("assessment")

            if pd.notna(assessment) and str(assessment).strip():
                st.caption(f"Prepares you for: {assessment}")

        with st.container(border=True):

            st.markdown(f"**{number}. {resource['resource_title']}**")
            st.caption(
                f"{resource['resource_type']} | {resource['difficulty']} | "
                f"{resource['activity_type']}"
            )
            st.caption(engine.explain_match(resource, decision))

            resource_url = resource.get("resource_url")

            if pd.notna(resource_url) and valid_url(resource_url):

                st.link_button("Open Learning Resource", str(resource_url))

            render_resource_content(resource)


# ============================================================
# 7. EDUCATIONAL KNOWLEDGE BASE MANAGEMENT
# ============================================================

def render_ekb():


    st.title("Educational Knowledge Base Manager")
    st.write("Manage resources used by AILAF. Changes are saved to learning_resources.csv.")

    required_columns = [
        "course", "topic", "resource_title", "resource_type",
        "difficulty", "competency_level", "learning_strategy",
        "activity_type", "resource_url"
    ]

    if not KNOWLEDGE_BASE_PATH.exists():
        st.error(f"Knowledge base file not found: {KNOWLEDGE_BASE_PATH}")
        st.stop()

    try:
        ekb = pd.read_csv(KNOWLEDGE_BASE_PATH)
    except Exception as error:
        st.error(f"Unable to read knowledge base: {error}")
        st.stop()

    missing = [c for c in required_columns if c not in ekb.columns]
    if missing:
        st.error("CSV is missing required columns: " + ", ".join(missing))
        st.stop()

    st.caption(
        "Competency and pedagogical labels are AILAF prototype mappings; "
        "seek lecturer review before treating them as validated classifications."
    )

    a, b, c = st.columns(3)
    a.metric("Resources", len(ekb))
    b.metric("Courses", ekb["course"].nunique())
    c.metric(
        "No link or content",
        int((
            ekb["resource_url"].fillna("").astype(str).str.strip().eq("")
            & (ekb["content"].fillna("").astype(str).str.strip().eq("") if "content" in ekb.columns else True)
        ).sum())
    )

    with st.expander("Add a Resource", expanded=True):
        with st.form("ekb_add_form", clear_on_submit=True):
            left, right = st.columns(2)
            with left:
                course = st.text_input("Course Code", value=PILOT_COURSE)
                topic = st.text_input("Topic")
                title = st.text_input("Resource Title")
                url = st.text_input("Resource URL (optional)")
                assessment = st.selectbox(
                    "Aligned assessment (CA)",
                    ["", "Test 1; Quiz 1", "Test 2; Quiz 2", "Test 1", "Quiz 1", "Test 2", "Quiz 2"]
                )
            with right:
                resource_type = st.selectbox(
                    "Resource Type",
                    ["Lecture Notes", "Video", "Worked Example", "Practice Exercise",
                     "Quiz", "Case Study", "Project", "Other"]
                )
                difficulty = st.selectbox("Difficulty", ["Basic", "Intermediate", "Advanced"])
                competency = st.selectbox(
                    "Competency Level", ["Beginner", "Intermediate", "Advanced"]
                )
                strategy = st.selectbox(
                    "Learning Strategy",
                    ["Foundational Learning", "Reinforcement and Guided Practice",
                     "Extension and Challenging Application"]
                )
                activity = st.selectbox(
                    "Activity Type",
                    ["Concept Review", "Guided Practice", "Basic Quiz",
                     "Concept Reinforcement", "Practice Exercise", "Application Quiz",
                     "Case Study", "Problem Solving", "Project-Based Activity",
                     "Independent Learning"]
                )

            content = st.text_area("Content (task text or notes, optional)")
            add = st.form_submit_button("Add Resource", type="primary")
            if add:
                if not all(x.strip() for x in [course, topic, title]):
                    st.error("Course, topic, and title are required.")
                elif url.strip() and not valid_url(url):
                    st.error("The URL must start with http:// or https://.")
                else:
                    dupe = (
                        ekb["course"].fillna("").astype(str).str.casefold().eq(course.strip().casefold())
                        & ekb["resource_title"].fillna("").astype(str).str.casefold().eq(title.strip().casefold())
                        & ekb["topic"].fillna("").astype(str).str.casefold().eq(topic.strip().casefold())
                    ).any()
                    if dupe:
                        st.error("A resource with the same course, title, and topic already exists.")
                    else:
                        row = {col: "" for col in ekb.columns}
                        row.update({
                            "course": course.strip(),
                            "topic": topic.strip(), "resource_title": title.strip(),
                            "resource_type": resource_type, "difficulty": difficulty,
                            "competency_level": competency, "learning_strategy": strategy,
                            "activity_type": activity, "resource_url": url.strip(),
                            "content": content.strip(),
                            "assessment": assessment
                        })
                        numbers = pd.to_numeric(
                            ekb["resource_id"].astype(str).str.extract(r"R(\d+)$")[0],
                            errors="coerce"
                        )
                        next_number = int(numbers.max()) + 1 if numbers.notna().any() else 1
                        row["resource_id"] = f"{course.strip()}-R{next_number:03d}"
                        pd.concat([ekb, pd.DataFrame([row])], ignore_index=True).to_csv(
                            KNOWLEDGE_BASE_PATH, index=False
                        )
                        load_recommendation_engine.clear()
                        st.success("Resource added.")
                        st.rerun()

    st.divider()
    st.subheader("Browse and Edit Resources")
    f1, f2, f3 = st.columns(3)
    with f1:
        course_filter = st.selectbox(
            "Course filter", ["All"] + sorted(ekb["course"].dropna().astype(str).unique())
        )
    with f2:
        level_filter = st.selectbox(
            "Competency filter",
            ["All"] + sorted(ekb["competency_level"].dropna().astype(str).unique())
        )
    with f3:
        topic_view_filter = st.selectbox(
            "Topic filter", ["All"] + sorted(ekb["topic"].dropna().astype(str).unique())
        )

    view = ekb.copy()
    if course_filter != "All":
        view = view[view["course"].astype(str) == course_filter]
    if level_filter != "All":
        view = view[view["competency_level"].astype(str) == level_filter]
    if topic_view_filter != "All":
        view = view[view["topic"].astype(str) == topic_view_filter]

    st.caption(f"Displaying {len(view)} of {len(ekb)} resources.")
    edited = st.data_editor(
        view,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="ekb_editor",
        column_config={
            "competency_level": st.column_config.SelectboxColumn(
                "Competency Level", options=["Beginner", "Intermediate", "Advanced"]
            ),
            "difficulty": st.column_config.SelectboxColumn(
                "Difficulty", options=["Basic", "Intermediate", "Advanced"]
            )
        }
    )

    if st.button("Save Edited Resources", type="primary"):
        updated = ekb.copy()
        if len(edited) != len(view):
            st.error("Row count changed unexpectedly; no changes were saved.")
        else:
            for pos, original_idx in enumerate(view.index):
                for col in edited.columns:
                    updated.loc[original_idx, col] = edited.iloc[pos][col]

            required_nonblank = ["course", "topic", "resource_title"]
            blank = updated[required_nonblank].fillna("").astype(str).apply(
                lambda s: s.str.strip().eq("")
            ).any(axis=1)
            duplicates = updated.duplicated(
                subset=["course", "resource_title", "topic"], keep=False
            )
            if blank.any():
                st.error("Required fields cannot be blank.")
            elif duplicates.any():
                st.error("Duplicate course/title/topic combinations detected.")
            else:
                updated.to_csv(KNOWLEDGE_BASE_PATH, index=False)
                load_recommendation_engine.clear()
                st.success("Edits saved.")
                st.rerun()

    st.divider()
    st.subheader("Delete a Resource")
    if not ekb.empty:
        delete_idx = st.selectbox(
            "Select resource",
            options=list(ekb.index),
            format_func=lambda i: (
                f"{ekb.loc[i, 'course']} | {ekb.loc[i, 'resource_title']} "
                f"| {ekb.loc[i, 'competency_level']}"
            )
        )
        confirm = st.checkbox("Confirm deletion")
        if st.button("Delete Selected Resource", disabled=not confirm):
            ekb.drop(index=delete_idx).reset_index(drop=True).to_csv(
                KNOWLEDGE_BASE_PATH, index=False
            )
            load_recommendation_engine.clear()
            st.success("Resource deleted.")
            st.rerun()

    st.download_button(
        "Download Knowledge Base Backup",
        data=ekb.to_csv(index=False).encode("utf-8"),
        file_name="learning_resources_backup.csv",
        mime="text/csv"
    )


# ============================================================
# 8. ABOUT AILAF
# ============================================================

def render_about():

    import predictor

    st.title("About AILAF")

    st.write(
        "The Adaptive Intelligent Learning Alignment Framework (AILAF) "
        "combines machine learning with three educational theories to "
        "support personalised, competency-based learning."
    )

    st.divider()

    section_heading("Framework Components")

    components = [
        (
            "Data Preprocessing",
            "Converts OULAD tutor-marked and computer-marked assessments into "
            "Test 1, Quiz 1, Test 2 and Quiz 2. The label comes from the "
            "student's later assessment average."
        ),
        (
            "AI Prediction Model",
            f"{predictor.MODEL_INFO['algorithm']} was selected after comparing "
            "Decision Tree, Random Forest and XGBoost. One model is trained "
            "for each combination of available marks."
        ),
        (
            "Pedagogical Decision Module",
            "Applies CBL (level and difficulty), CA (assessment priority) and "
            "UDL (formats) and explains each decision."
        ),
        (
            "Educational Knowledge Base",
            "Stores resources with topic, aligned assessment, competency, "
            "difficulty, format, activity, link and content."
        ),
        (
            "Recommendation Engine",
            "Filters resources by the PDM decision and orders them as a "
            "pathway: understand, practise, check, extend."
        ),
        (
            "Student Interface and Evaluation",
            "Dashboard, quizzes, pre/post tests and questionnaires."
        )
    ]

    for index, (title, description) in enumerate(components, start=1):

        with st.container(border=True):

            st.markdown(f"**{index}. {title}**")
            st.write(description)

    st.divider()

    section_heading("Educational Theories")

    theories = [
        (
            "Constructive Alignment (CA)",
            "Each topic is linked to the test or quiz it prepares you for. "
            "Topics for your weakest assessment are listed first."
        ),
        (
            "Competency-Based Learning (CBL)",
            "Your competency level decides the difficulty. It is the higher "
            "of your current level (average of your marks) and the predicted "
            "level."
        ),
        (
            "Universal Design for Learning (UDL)",
            "Several formats are offered for each level and you can choose "
            "the formats you prefer."
        )
    ]

    for title, description in theories:

        with st.container(border=True):

            st.markdown(f"**{title}**")
            st.write(description)

    st.divider()

    st.caption(
        "Research prototype. Data: OULAD assessments converted to test and "
        "quiz format because university data was not available."
    )


# ============================================================
# 9. RUN THE APPLICATION
# ============================================================

def run(include_ekb=False):

    st.set_page_config(
        page_title="AILAF | Learning Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(STYLE, unsafe_allow_html=True)

    pages = ["Home", "Learning Dashboard", "Evaluation"]

    if include_ekb:
        pages.append("EKB Management")

    pages.append("About AILAF")

    with st.sidebar:

        st.title("AILAF")

        st.caption("Adaptive Intelligent Learning Alignment Framework")

        st.divider()

        page = st.radio("NAVIGATION", pages)

        st.divider()

        st.markdown("**PROJECT INFORMATION**")

        st.caption("Final Year Project Prototype")

        st.caption("Courses: " + ", ".join(COURSE_NAMES))

        st.caption("Data Source: OULAD (test/quiz format)")

    try:

        if page == "Home":
            render_home()

        elif page == "Learning Dashboard":
            render_dashboard()

        elif page == "Evaluation":
            render_evaluation_page(include_ekb)

        elif page == "EKB Management":
            render_ekb()

        else:
            render_about()

    except FileNotFoundError as error:

        st.error(f"Unable to start AILAF: {error}")
