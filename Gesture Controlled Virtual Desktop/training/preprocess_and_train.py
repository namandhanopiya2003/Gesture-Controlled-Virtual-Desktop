# Importing necessary libraries
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Path to raw gesture landmark data
DATA_PATH = "data/raw_landmarks"

# Lists to store features (X) and labels (y)
X, y = [], []

# Loops through all .npy files in the data folder
for file in os.listdir(DATA_PATH):
    label = file.replace(".npy", "")                     # Extracts gesture label from filename
    data = np.load(os.path.join(DATA_PATH, file))        # Loads array of landmarks
    X.extend(data)                                       # Appends landmarks to feature list
    y.extend([label] * len(data))                        # Appends corresponding labels

# Converts lists to arrays
X = np.array(X)
y = np.array(y)

# Creates a DataFrame for easier manipulation and export
feature_names = [f"f{i}" for i in range(X.shape[1])]     # Feature column names
df = pd.DataFrame(X, columns=feature_names)
df["label"] = y                                          # Adds label column

# Saves combined training data to CSV for future reference
df.to_csv("training/gesture_data.csv", index=False)
print(f"<><> Loaded {len(X)} samples from {len(set(y))} gesture classes.")
print(">>>> Training data exported to training/gesture_data.csv")

# ---------- Trains Random Forest Classifier ----------
X = df.drop(columns=["label"])                                     # Features for training
y = df["label"]                                                    # Target labels
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize model
model.fit(X, y)                                                    # Train model

# Saves trained model for later use
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/gesture_classifier.pkl")
print("<><> Model trained and saved to models/gesture_classifier.pkl")

