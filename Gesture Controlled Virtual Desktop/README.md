## ✋ Real-Time Gesture-Controlled Virtual Desktop

## 🧠 ABOUT THIS PROJECT ==>
- An AI-powered real-time gesture-controlled virtual desktop environment that allows users to interact with a simulated UI using hand gestures—without touching the screen. It uses MediaPipe for landmark detection, Random Forest for gesture classification, and PyQt5 for UI interaction.

- It continuously improves over time by retraining itself using low-confidence predictions and visualizes the improvement using confidence comparison plots. Ideal for data science demonstrations, human-computer interaction prototypes, or futuristic control interfaces.

---

## ⚙ TECHNOLOGIES USED ==>

- **Python**
- **MediaPipe** (hand landmarks)
- **OpenCV** (frame capture & display)
- **NumPy, Pandas**
- **Scikit-learn** (Random Forest classifier)
- **Matplotlib** (confidence visualization)
- **PyQt5** (virtual desktop UI)
- **Joblib** (model persistence)

---

## 📁 PROJECT FOLDER STRUCTURE ==>

main_folder/<br>
│<br>
├── main.py                         # Main entry point to run the app<br>
├── requirements.txt                # All dependencies<br>
├── README.md                       # Project overview and setup guide<br>
├── record_gesture_data.py<br>
├── ui_manager.py<br>
│<br>
├── models/<br>
│   ├── gesture_classifier.pkl      # Trained gesture classification model<br>
│   └── depth_model/                # Pretrained MonoDepth2 model (optional download)<br>
│<br>
├── modules/<br>
│   ├── gesture_detection.py        # Detects gestures using landmarks<br>
│   ├── depth_estimation.py         # Gets hand depth using MonoDepth2<br>
│   ├── hand_tracking.py            # MediaPipe-based real-time hand tracking<br>
│   ├── analytics_logger.py         # Logs gestures, app usage, session stats<br>
│   └── gpt_assistant.py            # Optional: GPT integration (OpenAI API)<br>
│<br>
├── utils/<br>
│   ├── data_preprocessing.py       # Prepares data for training/classification<br>
│   ├── plot_utils.py               # Functions for session visualizations<br>
│   └── config.py                   # Config variables (paths, thresholds, etc.)<br>
│<br>
├── data/<br>
│   ├── raw_landmarks/              # Raw landmark data from training<br>
│   └── processed/                  # Processed data for model training<br>
│<br>
├── logger/<br>
│   └── low_confidence_logger.py<br>
│<br>
├── logs/<br>
│   ├── confidence_before.csv<br>
│   └── low_confidence_data.csv<br>
│<br>
├── ml/<br>
│   └──gesture_classifier.py<br>
│<br>
└── training/<br>
    ├── train_gesture_classifier.py # Trains scikit-learn model on gestures<br>
    ├── cluster_gestures.py         # Optional: KMeans or DBSCAN for gestures<br>
    └── analyze_sessions.py         # Creates EDA reports/heatmaps from logs

---

## 📝 WHAT EACH FILE DOES ==>

**main.py**:
- Runs the webcam, tracks hand landmarks, classifies gestures, interacts with the PyQt5 UI, logs low-confidence gestures, and retrains the model post-session.

**ml/gesture_classifier.py**:
- Loads the trained model and provides classification with probability.

**ui_manager.py**:
- Contains PyQt5 code to display a desktop-like interface, draws points, and triggers mock actions.

**logger/low_confidence_logger.py**:
- Stores low-confidence gesture data to logs/ for improving the model.

**training/preprocess_and_train.py**:
- Trains the initial Random Forest model using .npy landmark data and exports gesture_data.csv.

**logs/**:
- Stores confidence logs, retraining plots, and gesture logs.

---

## 🚀 HOW TO RUN ==>

# Step 1: Move to project folder
cd "D:\Gesture-Controlled Virtual Desktop"
D:

# Step 2: Activate virtual environment
python -m venv venv
venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the project
python main.py
[[ Use hand gestures in front of webcam to interact ]]
[[ Press 'Q' on the OpenCV window to quit ]]

# Step 5: After exit:
- The model is retrained if low-confidence data is available
- A graph will show how confidence improved
- Image will auto-delete after closing

# SOME ADDITIONAL STEPS:
- python record_gesture_data.py                          << To record hand movements for training
- python training/preprocess_and_train.py                << To train model
[[ RUN THESE STEPS AFTER INSTALL DEPENDENCIES ]]

---

## ✅ FEATURES & IMPROVEMENTS ==>
- Gesture prediction with real-time confidence thresholding
- Auto-logging and retraining with low-confidence gestures
- Smart visualization showing model improvement over time
- Real desktop interaction with gestures using PyQt5
- Modular structure: ML, UI, logging, training separated cleanly
- Confidence plot is deleted after user closes it to avoid clutter

---

## 📌 TO DO / FUTURE ENHANCEMENTS ==>
- Add 3D interaction using depth estimation (MonoDepth)
- Enable drag & drop, window snapping gestures
- Voice-assistant integration using NLP
- Full-fledged gesture training GUI for custom classes
- Save interaction analytics across sessions
- Export retrained models version-wise

---

## ✨ SAMPLE OUTPUT ==>

🎥 Camera: ON
🤖 Gesture: "SwipeRight" with 92.4% confidence
⚠️ Logged gesture with 65.3% confidence (used for retraining)
✅ Model improved and saved successfully.
📊 Saved confidence plot to logs/confidence_comparison.png
🧽 Deleted confidence plot image.

---

## 📬 CONTACT ==>
For questions or feedback, feel free to reach out!


---
