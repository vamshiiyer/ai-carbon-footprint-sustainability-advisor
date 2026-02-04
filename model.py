import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("data.csv")

# Encode diet column
encoder = LabelEncoder()
data["diet"] = encoder.fit_transform(data["diet"])

# Features and target
X = data[["transport_km", "electricity_units", "diet", "household_size"]]
y = data["impact"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
with open("carbon_model.pkl", "wb") as f:
    pickle.dump((model, encoder), f)

print("✅ AI model trained and saved as carbon_model.pkl")
