import os
import csv
from datetime import datetime

LOG_FILE = "logs/low_confidence_data.csv"

def log_low_confidence(landmarks, confidence):
    file_exists = os.path.isfile(LOG_FILE)
    write_header = not file_exists or os.stat(LOG_FILE).st_size == 0

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        if write_header:
            header = ["timestamp", "confidence"] + [f"f{i}" for i in range(63)]
            writer.writerow(header)

        row = [datetime.now().isoformat(), confidence] + landmarks
        writer.writerow(row)
