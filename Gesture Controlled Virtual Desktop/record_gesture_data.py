# Importing necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import os

# Initializes hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Sets path to save recorded landmark data
SAVE_PATH = "data/raw_landmarks"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Prompts user to enter gesture label
gesture_label = input("Enter gesture label (e.g., swipe, pinch, draw): ")
samples = []

# Starts webcam for recording gestures
cap = cv2.VideoCapture(0)
print("Recording started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()                              # Captures frame from webcam
    if not ret:
        break
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # Converts frame to RGB
    result = hands.process(frame_rgb)                    # Detects hand landmarks

    # Checks if any hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draws hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extracts x, y, z coordinates for each landmark
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            # Stores the landmark data for current frame
            samples.append(landmark_list)

    # Displays live recording with landmarks
    cv2.imshow("Recording Gesture", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Releases webcam and closes window
cap.release()
cv2.destroyAllWindows()

# Saves recorded landmark data
np.save(f"{SAVE_PATH}/{gesture_label}.npy", np.array(samples))
print(f"Saved {len(samples)} samples under label '{gesture_label}'")
