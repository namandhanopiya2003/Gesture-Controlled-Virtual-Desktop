import numpy as np
import joblib

class GestureClassifier:
    def __init__(self, model_path="models/gesture_classifier.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, landmarks: np.ndarray) -> str:
        features = landmarks.flatten()
        return self.model.predict([features])[0]
