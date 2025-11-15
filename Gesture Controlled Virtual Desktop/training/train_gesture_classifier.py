# Importing necessary libraries
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# ---------- Loads preprocessed gesture data ----------
data = np.load("data/processed/landmarks.npy")            # Array of hand landmarks (features)
labels = np.load("data/processed/labels.npy")             # Corresponding gesture labels

# ---------- Trains Random Forest Classifier ----------
model = RandomForestClassifier(n_estimators=100)         # Initialize model with 100 trees
model.fit(data, labels)                                  # Train model on the features and labels

# ---------------- Saves trained model ----------------
joblib.dump(model, "models/gesture_classifier.pkl")      # Save model for later use
