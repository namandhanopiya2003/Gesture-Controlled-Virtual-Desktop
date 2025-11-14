# Importing necessary libraries
import joblib
import numpy as np
import pandas as pd

# Class to handle gesture classification using a pre-trained model
class GestureClassifier:
    def __init__(self, model_path="models/gesture_classifier.pkl"):

        # Loads pre-trained Random Forest classifier
        self.model = joblib.load(model_path)
        
        # Feature names for the 63 hand landmarks (x, y, z for 21 points)
        self.feature_names = [f"f{i}" for i in range(63)]

    def classify(self, features):

        # Converts features into a DataFrame for model compatibility
        df = pd.DataFrame([features], columns=self.feature_names)

        # Gets probability estimates for all classes
        probs = self.model.predict_proba(df)[0]

        # Finds maximum probability and corresponding gesture label
        max_prob = max(probs)
        gesture = self.model.classes_[np.argmax(probs)]

        # Returns predicted gesture and its confidence
        return gesture, max_prob
