import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
data = pd.read_csv(r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\data/initial_train.csv")

# Replace missing values
data.replace(" ?", np.nan, inplace=True)
data.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split features and target
X = data.drop("income", axis=1)
y = data["income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save outputs
os.makedirs("results", exist_ok=True)
joblib.dump(model, r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\results/base_model.pkl")

test_results = X_test.copy()
test_results["y_true"] = y_test.values
test_results["y_pred"] = y_pred

test_results.to_csv(r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\results/base_predictions.csv", index=False)


with open(r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\results/before_retraining.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

print("✅ Base model trained successfully")
print("Accuracy:", accuracy)
