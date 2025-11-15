# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loads session logs containing gesture data
df = pd.read_csv("data/session_logs.csv")

# ---------- Plot 1: Frequency of each gesture ----------
plt.figure(figsize=(8, 5))                              # Sets figure size
sns.countplot(x="gesture", data=df)                     # Counts occurrences of each gesture
plt.title("Gesture Frequency")                          # Adds title to the plot
plt.show()                                              # Displays the plot

# ------ Prepare timestamp for time-based analysis ------
df["timestamp"] = pd.to_datetime(df["timestamp"])       # Converts timestamp to datetime
df["hour"] = df["timestamp"].dt.hour                    # Extracts hour from timestamp

# ------------ Plot 2: Gesture usage by hour ------------
plt.figure(figsize=(8, 5))                              # Sets figure size
sns.histplot(data=df, x="hour", hue="gesture", multiple="stack")
# Plots stacked histogram showing how gestures are distributed over hours
plt.title("Gesture Usage by Hour")                      # Adds title
plt.show()                                              # Displays the plot
