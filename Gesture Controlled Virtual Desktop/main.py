# Importing all necessary libraries
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from ui_manager import GestureDesktopWindow
import mediapipe as mp
import threading
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import subprocess

from ml.gesture_classifier import GestureClassifier
from logger.low_confidence_logger import log_low_confidence

# Loads the pre-trained gesture classifier
classifier = GestureClassifier("models/gesture_classifier.pkl")

# Initializes hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils


# Main loop to capture video and detect gestures
def gesture_loop(ui):
    cap = cv2.VideoCapture(0)
    current_gesture = ""
    all_confidences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        h, w, _ = frame.shape

        # Checks if any hand is detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draws landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extracts x, y, z coordinates of each hand landmark
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # If all landmarks detected, classify gesture
                if len(landmarks) == 63:
                    gesture, confidence = classifier.classify(landmarks)
                    # Tracks confidence for analysis
                    all_confidences.append(confidence)

                    # Logs low-confidence gestures for retraining
                    if confidence < 0.8:
                        log_low_confidence(landmarks, confidence)

                    # Triggers action only if gesture changes
                    if gesture and gesture != current_gesture:
                        current_gesture = gesture
                        ui.trigger_action(gesture)

                # Draws a point in the UI following the index finger tip
                index_tip = hand_landmarks.landmark[8]
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)

                # Maps webcam coordinates to UI coordinates
                qt_x = int(cx * ui.width() / w)
                qt_y = int(cy * ui.height() / h)
                ui.draw_point(qt_x, qt_y)

        # Shows webcam feed with hand landmarks
        cv2.imshow("Gesture-Controlled Virtual Desktop - Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Releases webcam
    cap.release()
    cv2.destroyAllWindows()

    # Saves all gesture confidences before retraining
    if all_confidences:
        pd.Series(all_confidences).to_csv("logs/confidence_before.csv", index=False)

    # Retrains the model using low-confidence gestures
    auto_retrain_on_exit()


# Function to retrain gesture classifier using low-confidence data
def auto_retrain_on_exit():
    log_path = "logs/low_confidence_data.csv"
    if not os.path.exists(log_path):
        print("<!> No low-confidence data found for retraining.")
        return

    df = pd.read_csv(log_path)
    if df.empty:
        print("<!> Log file is empty. Skipping retraining.")
        return

    print(">>> Using low-confidence logs to improve model...")

    # Feature column names for hand landmarks
    feature_cols = [f"f{i}" for i in range(63)]
    df_features = df[feature_cols]

    # Loads existing model to predict labels for low-confidence data
    model = joblib.load("models/gesture_classifier.pkl")
    df["label"] = model.predict(df_features)

    # Combines new low-confidence data with base training data
    df_base = pd.read_csv("training/gesture_data.csv")
    df_combined = pd.concat([df_base, df[["label"] + feature_cols]], ignore_index=True)
    df_combined = df_combined.sample(frac=1).reset_index(drop=True)

    # Trains new Random Forest model
    X = df_combined[feature_cols]
    y = df_combined["label"]
    new_model = RandomForestClassifier(n_estimators=150, random_state=42)
    new_model.fit(X, y)
    joblib.dump(new_model, "models/gesture_classifier.pkl")
    print("<><> Model improved and saved successfully.")

    # Clears low-confidence log for next session
    open(log_path, "w").write("timestamp,confidence," + ",".join(feature_cols) + "\n")
    print(">>> Cleared low-confidence log for next session.")

    # Plots confidence before vs after retraining
    try:
        before_conf = pd.read_csv("logs/confidence_before.csv", header=None)[0]
        after_conf = new_model.predict_proba(X).max(axis=1)

        plt.figure(figsize=(10, 5))
        plt.hist(before_conf, bins=20, alpha=0.6, label='Before Retraining')
        plt.hist(after_conf, bins=20, alpha=0.6, label='After Retraining')
        plt.axvline(np.mean(before_conf), color='red', linestyle='--', label=f'Avg Before: {np.mean(before_conf):.2f}')
        plt.axvline(np.mean(after_conf), color='green', linestyle='--', label=f'Avg After: {np.mean(after_conf):.2f}')
        plt.title("Model Confidence Distribution Before vs After Retraining")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        os.makedirs("logs", exist_ok=True)
        plot_path = "logs/confidence_comparison.png"
        plt.savefig(plot_path)
        print(f"<!> Saved confidence plot to {plot_path}")

        # Opens plot automatically
        if sys.platform == "win32":
            subprocess.Popen(["start", "", plot_path], shell=True)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", plot_path])
        else:
            subprocess.Popen(["xdg-open", plot_path])

        # Waits for user to close plot before deleting
        input(">>> Press Enter after closing the plot to auto-delete it...")

        os.remove(plot_path)                      # Deletes plot after viewing
        print("<!> Deleted confidence plot image.")

    except Exception as e:
        print(f"<!> Visualization failed: {e}")


# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = GestureDesktopWindow()
    t = threading.Thread(target=gesture_loop, args=(ui,))
    t.start()
    sys.exit(app.exec_())

