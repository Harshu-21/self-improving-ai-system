import os

# Resolve project root safely

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

before_path = os.path.join(RESULTS_DIR, "before_retraining.txt")
after_path = os.path.join(RESULTS_DIR, "after_retraining.txt")

# Read results

with open(before_path, "r") as f:
    before = f.read()

with open(after_path, "r") as f:
    after = f.read()

# Display comparison

print("= BEFORE RETRAINING =")
print(before)

print("\n= AFTER RETRAINING =")
print(after)
