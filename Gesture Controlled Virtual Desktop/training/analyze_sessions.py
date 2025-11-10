import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/session_logs.csv")

plt.figure(figsize=(8, 5))
sns.countplot(x="gesture", data=df)
plt.title("Gesture Frequency")
plt.show()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="hour", hue="gesture", multiple="stack")
plt.title("Gesture Usage by Hour")
plt.show()
