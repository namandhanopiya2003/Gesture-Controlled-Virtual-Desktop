# Importing necessary modules
import os
import csv
from datetime import datetime

# Path to store low-confidence gesture logs
LOG_FILE = "logs/low_confidence_data.csv"

# Function to log gestures with low classification confidence
def log_low_confidence(landmarks, confidence):

    # Checks if log file already exists
    file_exists = os.path.isfile(LOG_FILE)

    # Determines if header needs to be written
    write_header = not file_exists or os.stat(LOG_FILE).st_size == 0

    # Opens CSV in append mode
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Writes header if file is new or empty
        if write_header:
            header = ["timestamp", "confidence"] + [f"f{i}" for i in range(63)]
            writer.writerow(header)

        # Prepares row: timestamp, confidence, all 63 landmarks
        row = [datetime.now().isoformat(), confidence] + landmarks
        writer.writerow(row)                   # Writes row to CSV
