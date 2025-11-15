# Importing necessary libraries
import numpy as np
import joblib

# Class to handle gesture classification using a pre-trained model
class GestureClassifier:
    def __init__(self, model_path="models/gesture_classifier.pkl"):
        # Loads the model
        self.model = joblib.load(model_path)

    def predict(self, landmarks: np.ndarray) -> str:
        # Flattens landmarks to 1D array if needed
        features = landmarks.flatten()

        # Predicts gesture using the loaded model
        return self.model.predict([features])[0]
