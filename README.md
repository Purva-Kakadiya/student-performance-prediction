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

🛠 Technologies Used
Python
Pandas
NumPy
Scikit-learn
Streamlit
Pickle
Git
GitHub

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

📈 Model Evaluation

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