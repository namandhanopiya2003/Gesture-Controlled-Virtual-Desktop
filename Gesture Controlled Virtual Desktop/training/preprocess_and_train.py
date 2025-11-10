import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = "data/raw_landmarks"
X, y = [], []

for file in os.listdir(DATA_PATH):
    label = file.replace(".npy", "")
    data = np.load(os.path.join(DATA_PATH, file))
    X.extend(data)
    y.extend([label] * len(data))

X = np.array(X)
y = np.array(y)

feature_names = [f"f{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["label"] = y
df.to_csv("training/gesture_data.csv", index=False)
print(f"<><> Loaded {len(X)} samples from {len(set(y))} gesture classes.")
print(">>>> Training data exported to training/gesture_data.csv")

X = df.drop(columns=["label"])
y = df["label"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/gesture_classifier.pkl")
print("<><> Model trained and saved to models/gesture_classifier.pkl")
