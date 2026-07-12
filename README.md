🎓 Student Performance Prediction

A Machine Learning project that predicts a student's final grade (G3) using academic, family, and personal factors from the Student Performance Dataset.

This project was developed using Python, Scikit-learn, and Streamlit as part of my Machine Learning learning journey and internship.

📌 Project Overview

Student performance is influenced by multiple factors such as:

Study time
Previous failures
Parents' education
Travel time
Family relationship
Internet access
School absences

This project uses a Random Forest Regressor to estimate the student's final grade (G3).

🚀 Features
Data preprocessing using Scikit-learn Pipeline
OneHotEncoder for categorical features
Random Forest Regression model
Model evaluation using MAE, RMSE, and R² Score
Interactive Streamlit web application
Predicts student final grade based on user input
🛠 Technologies Used
Python
Pandas
NumPy
Scikit-learn
Streamlit
Pickle
Git
GitHub
📂 Project Structure
student-performance-prediction/
│
├── app.py
├── data/
│   └── student-mat.csv
│
├── models/
│   └── student_pipeline.pkl
│
├── notebooks/
│   └── Student_Performance.ipynb
│
├── src/
│   └── train.py
│
├── requirements.txt
├── README.md
└── .gitignore
📊 Selected Features

The final model uses the following features:

Age
Mother's Education
Father's Education
Travel Time
Study Time
Previous Failures
Internet Access
Higher Education Aspiration
Family Relationship
Absences

Target Variable:

G3 (Final Grade)
⚙️ Machine Learning Pipeline

Dataset

    ↓

Feature Selection

    ↓

Preprocessing

    ↓

OneHotEncoder

    ↓

Random Forest Regressor

    ↓

Model Evaluation

    ↓

Pipeline Saved

    ↓

Streamlit Application

📈 Model Evaluation

Replace these values with your actual results after training.

Metric	Value
MAE	XX.XX
RMSE	XX.XX
R² Score	XX.XX
▶️ Installation

Clone the repository

Go to the project directory

cd student-performance-prediction

Create a virtual environment

python -m venv venv

Activate the environment

Windows

venv\Scripts\activate

Install dependencies

pip install -r requirements.txt
🧠 Train the Model

Go to the source folder

cd src

Run

python train.py

This generates:

models/student_pipeline.pkl
💻 Run the Application

Return to the project root folder

cd ..

Run

streamlit run app.py

The application will open automatically in your browser.

📷 Application Screenshot

Add screenshots of your application here after completing the project.

Example:

Home Page
Prediction Result
Input Form
📚 Future Improvements
Hyperparameter tuning using GridSearchCV
Feature importance visualization
Deploy online
Improve UI design
Compare multiple regression algorithms
👨‍💻 Author

Kakadiya Purva

Bachelor of Engineering (Information Technology)

Machine Learning & Python Enthusiast

📄 License

This project is developed for educational purposes.