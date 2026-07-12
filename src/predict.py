# src/predict.py

import pickle
import pandas as pd


# Load model
with open("../models/student_model.pkl", "rb") as file:
    model = pickle.load(file)


# Example student
student = {
    "school": 0,
    "sex": 1,
    "age": 17,
    "address": 1,
    "famsize": 0,
    "Pstatus": 1,
    "Medu": 4,
    "Fedu": 4,
    "traveltime": 1,
    "studytime": 3,
    "failures": 0,
    "schoolsup": 0,
    "famsup": 1,
    "paid": 0,
    "activities": 1,
    "nursery": 1,
    "higher": 1,
    "internet": 1,
    "romantic": 0,
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 4,
    "G1": 15,
    "G2": 16
}

student_df = pd.DataFrame([student])

prediction = model.predict(student_df)

print("Predicted Final Grade:", prediction[0])