import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

lifestyle = ["alcohol_consumption_per_week", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day"]

scaler = StandardScaler()
train[lifestyle] = scaler.fit_transform(train[lifestyle])
test[lifestyle] = scaler.transform(test[lifestyle])

encoder = LabelEncoder()
train["diabetes_stage"] = encoder.fit_transform(train["diabetes_stage"])
test["diabetes_stage"] = encoder.transform(test["diabetes_stage"])