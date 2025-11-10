import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

data = np.load("data/processed/landmarks.npy")
labels = np.load("data/processed/labels.npy")

model = RandomForestClassifier(n_estimators=100)
model.fit(data, labels)

joblib.dump(model, "models/gesture_classifier.pkl")
