import joblib
import numpy as np
import pandas as pd

class GestureClassifier:
    def __init__(self, model_path="models/gesture_classifier.pkl"):
        self.model = joblib.load(model_path)
        self.feature_names = [f"f{i}" for i in range(63)]

    def classify(self, features):
        df = pd.DataFrame([features], columns=self.feature_names)
        probs = self.model.predict_proba(df)[0]
        max_prob = max(probs)
        gesture = self.model.classes_[np.argmax(probs)]
        return gesture, max_prob
