from preprocessing import load_data

# Load and preprocess data
df = load_data("../data/student-mat.csv")

print(df.head())