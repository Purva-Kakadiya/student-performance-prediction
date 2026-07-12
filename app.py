import streamlit as st
import pickle
import pandas as pd

with open("models/student_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Student Performance Prediction")

st.write("Welcome to the Student Performance Prediction App")

studytime = st.slider(
    "Study Time",
    1,
    4,
    2
)

failures = st.slider(
    "Previous Failures",
    0,
    4,
    0
)

absences = st.number_input(
    "Absences",
    min_value=0,
    max_value=100,
    value=0
)