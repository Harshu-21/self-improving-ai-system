import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load original training data

original_data = pd.read_csv(
    r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\data\initial_train.csv"
)

# Load failure cases (with features)
failure_cases = pd.read_csv(
    r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\data\failure_cases.csv"
)

# Reconstruct failure dataset

failure_features = failure_cases.drop(["y_true", "y_pred"], axis=1)
failure_features["income"] = failure_cases["y_true"]

# Combine datasets

augmented_data = pd.concat(
    [original_data, failure_features],
    ignore_index=True
)

# Clean data

augmented_data.replace(" ?", np.nan, inplace=True)
augmented_data.dropna(inplace=True)

# Encode categorical variables

label_encoders = {}
for col in augmented_data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    augmented_data[col] = le.fit_transform(augmented_data[col])
    label_encoders[col] = le

# Split features and target

X = augmented_data.drop("income", axis=1)
y = augmented_data["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Retrain model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

joblib.dump(model, os.path.join(RESULTS_DIR, "improved_model.pkl"))

with open(os.path.join(RESULTS_DIR, "after_retraining.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

print("✅ Model retrained using failure-driven learning")
print("📈 New accuracy:", accuracy)
