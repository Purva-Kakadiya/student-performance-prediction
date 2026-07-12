import streamlit as st
import pandas as pd
import pickle
import os

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# ==========================================
# Title
# ==========================================

st.title("🎓 Student Performance Prediction")

st.markdown("""
Predict a student's **final grade (G3)** using a Machine Learning model
trained on the Student Performance Dataset.
""")

st.divider()

# ==========================================
# Load Trained Pipeline
# ==========================================

MODEL_PATH = "models/student_pipeline.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")

    st.info(
        "Run train.py first to generate the trained pipeline."
    )

    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.title("About")

st.sidebar.info(
"""
This application predicts the final student grade (G3)
using a Random Forest Regression model.

Model:
Random Forest Regressor

Dataset:
Student Performance Dataset

Built using:
- Python
- Scikit-learn
- Streamlit
"""
)

# ==========================================
# Input Section
# ==========================================

st.header("Enter Student Information")

# -------------------------
# Personal Information
# -------------------------

age = st.slider(
    "Age",
    min_value=15,
    max_value=22,
    value=17
)

# -------------------------
# Parent Education
# -------------------------

st.subheader("Parents Education")

Medu = st.selectbox(
    "Mother's Education",
    [
        0,
        1,
        2,
        3,
        4
    ],
    help="""
0 = None
1 = Primary Education
2 = 5th to 9th Grade
3 = Secondary Education
4 = Higher Education
"""
)

Fedu = st.selectbox(
    "Father's Education",
    [
        0,
        1,
        2,
        3,
        4
    ]
)

# -------------------------
# Travel Time
# -------------------------

traveltime = st.selectbox(
    "Travel Time",
    [
        1,
        2,
        3,
        4
    ],
    help="""
1 = <15 min

2 = 15-30 min

3 = 30-60 min

4 = >60 min
"""
)

# -------------------------
# Study Time
# -------------------------

studytime = st.selectbox(
    "Weekly Study Time",
    [
        1,
        2,
        3,
        4
    ],
    help="""
1 = <2 hours

2 = 2-5 hours

3 = 5-10 hours

4 = >10 hours
"""
)

# -------------------------
# Failures
# -------------------------

failures = st.selectbox(
    "Past Class Failures",
    [
        0,
        1,
        2,
        3
    ]
)

# -------------------------
# Internet Access
# -------------------------

internet = st.selectbox(
    "Internet Access at Home",
    [
        "yes",
        "no"
    ]
)

# -------------------------
# Higher Education
# -------------------------

higher = st.selectbox(
    "Wants Higher Education",
    [
        "yes",
        "no"
    ]
)

# -------------------------
# Family Relationship
# -------------------------

famrel = st.slider(
    "Family Relationship Quality",
    min_value=1,
    max_value=5,
    value=3
)

# -------------------------
# Absences
# -------------------------

absences = st.number_input(
    "Number of Absences",
    min_value=0,
    max_value=100,
    value=4
)

st.divider()

# ==========================================
# Create Prediction Data
# ==========================================

input_data = pd.DataFrame({
    "age": [age],
    "Medu": [Medu],
    "Fedu": [Fedu],
    "traveltime": [traveltime],
    "studytime": [studytime],
    "failures": [failures],
    "internet": [internet],
    "higher": [higher],
    "famrel": [famrel],
    "absences": [absences]
})

# ==========================================
# Predict Button
# ==========================================

if st.button("Predict Final Grade"):

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(20, prediction))

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        label="Predicted Final Grade (G3)",
        value=f"{prediction:.2f} / 20"
    )

    # --------------------------
    # Performance Category
    # --------------------------

    if prediction >= 16:
        st.success("Excellent Performance ⭐⭐⭐⭐⭐")

    elif prediction >= 14:
        st.success("Very Good Performance ⭐⭐⭐⭐")

    elif prediction >= 12:
        st.info("Good Performance ⭐⭐⭐")

    elif prediction >= 10:
        st.warning("Average Performance ⭐⭐")

    else:
        st.error("Needs Improvement ⭐")

    # --------------------------
    # Progress Bar
    # --------------------------

    st.subheader("Predicted Score")

    st.progress(prediction / 20)

    # --------------------------
    # Interpretation
    # --------------------------

    st.write("### Interpretation")

    if prediction >= 16:
        st.write(
            """
            The student is expected to perform exceptionally well
            in the final examination.
            """
        )

    elif prediction >= 12:
        st.write(
            """
            The student is expected to perform well.
            Continued study and attendance should help maintain
            this performance.
            """
        )

    elif prediction >= 10:
        st.write(
            """
            The student is likely to pass but there is room
            for improvement.
            """
        )

    else:
        st.write(
            """
            The student may require additional academic support
            before the final examination.
            """
        )

# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "Student Performance Prediction | "
    "Machine Learning Project | "
    "Random Forest Regressor | "
    "Streamlit"
)