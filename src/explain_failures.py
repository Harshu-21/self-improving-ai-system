import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

# Load trained model

model = joblib.load(
    r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\results\base_model.pkl"
)

# Load failure cases
failures = pd.read_csv(
    r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\data\failure_cases.csv"
)

# Prepare features
X_failures = failures.drop(["y_true", "y_pred"], axis=1)

# VERY SMALL sample (fast + safe)
X_sample = X_failures.sample(
    n=min(50, len(X_failures)),
    random_state=42
)

# SHAP explainer
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_sample,
    approximate=True
)

# HANDLE SHAP OUTPUT SAFELY
# Case 1: list (most common)
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]

# Case 2: numpy array with class axis
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_values_class1 = shap_values[:, :, 1]

else:
    raise ValueError("Unexpected SHAP output format")

# Save plot
output_dir = r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\results\shap_plots"
os.makedirs(output_dir, exist_ok=True)

plt.figure()
shap.summary_plot(
    shap_values_class1,
    X_sample,
    show=False
)
plt.savefig(
    os.path.join(output_dir, "failure_summary_class1.png"),
    bbox_inches="tight"
)
plt.close()

print("✅ SHAP failure explanation completed successfully")
print("📁 results/shap_plots/failure_summary_class1.png")
