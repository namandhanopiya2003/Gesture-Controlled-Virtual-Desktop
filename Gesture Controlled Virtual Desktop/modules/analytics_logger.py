# Importing libraries
import csv
from datetime import datetime

# Class to log gesture session data
class SessionLogger:
    def __init__(self, filepath="data/session_logs.csv"):
        self.filepath = filepath

        # Opens file in append mode and writes header if file is empty
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gesture", "depth", "app_used"])

    def log(self, gesture, depth, app_name="desktop"):
        # Opens CSV file in append mode
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)

            # Writes timestamp, gesture, depth (rounded to 2 decimals), and app name
            writer.writerow([datetime.now(), gesture, round(depth, 2), app_name])
