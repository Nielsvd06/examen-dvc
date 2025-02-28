import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import json

# Load data
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Load trained model
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions
y_pred = model.predict(X_test)

# Save predictions
pd.DataFrame(y_pred, columns=["predictions"]).to_csv("data/processed/predictions.csv", index=False)

# Compute metrics
metrics = {
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
}

# Save metrics
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f)
