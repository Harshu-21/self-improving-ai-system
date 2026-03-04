import pandas as pd

# Load predictions
preds = pd.read_csv(r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\results/base_predictions.csv")

# Identify failures
failures = preds[preds["y_true"] != preds["y_pred"]]

# Save failure cases
failures.to_csv(r"C:\Users\patil\OneDrive\Documents\self-improving-ai-system\data/failure_cases.csv", index=False)

print("❌ Total failures detected:", len(failures))
print("📁 Saved to data/failure_cases.csv")
