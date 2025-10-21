import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path

# Set save directory to the current folder (apps/explainai)
save_dir = Path(__file__).resolve().parent

# Load dataset
data = load_breast_cancer(as_frame=True)
df = data.frame
X = df.drop(columns=["target"])
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {acc:.3f}")

# Save dataset and model in the same folder as this script
csv_path = save_dir / "breast_cancer.csv"
pkl_path = save_dir / "rf_model.pkl"

df.to_csv(csv_path, index=False)
with open(pkl_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Files saved in {save_dir}")
print(f"   - {csv_path.name}")
print(f"   - {pkl_path.name}")
