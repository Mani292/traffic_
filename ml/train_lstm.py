import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Banglore_traffic_Dataset.csv")

# Feature engineering
df["Date"] = pd.to_datetime(df["Date"])
df["hour"] = df["Date"].dt.hour

df.rename(columns={
    "Traffic Volume": "vehicle_count",
    "Average Speed": "avg_speed",
    "Congestion Level": "congestion"
}, inplace=True)

# Normalize congestion to 3 classes
df["congestion"] = pd.cut(
    df["congestion"],
    bins=[0, 40, 70, 200],
    labels=[0,1,2]
).astype(int)

df = df.dropna()

# Build LSTM sequences
X, y = [], []
for i in range(len(df)-10):
    X.append(df[["vehicle_count","avg_speed","hour"]].values[i:i+10])
    y.append(df["congestion"].values[i+10])

X, y = np.array(X), np.array(y)

# Build model
model = Sequential([
    LSTM(64, input_shape=(10,3)),
    Dense(3, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=15, batch_size=32)

# Save model
model.save("../backend/lstm_model.h5")
print("LSTM model saved successfully.")
