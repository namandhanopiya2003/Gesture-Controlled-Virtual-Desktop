import csv
from datetime import datetime

class SessionLogger:
    def __init__(self, filepath="data/session_logs.csv"):
        self.filepath = filepath
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gesture", "depth", "app_used"])

    def log(self, gesture, depth, app_name="desktop"):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), gesture, round(depth, 2), app_name])
