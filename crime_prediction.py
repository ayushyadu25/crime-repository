import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("crime_data.csv")

# Select features and target
df['Hour'] = pd.to_datetime(df['Date']).dt.hour
features = df[['Latitude', 'Longitude', 'Hour']]
target = df['Crime_Type']  # Replace with actual crime type column

# Convert crime type to numeric labels
target = target.astype('category').cat.codes

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")