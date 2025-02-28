import pandas as pd
import pickle
from sklearn.linear_model import Ridge

# Load data
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Load best parameters
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Train model
model = Ridge(**best_params)
model.fit(X_train, y_train)

# Save trained model
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)
