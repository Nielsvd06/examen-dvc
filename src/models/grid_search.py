import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Load data
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Define hyperparameters
param_grid = {"alpha": [0.1, 1, 10, 100]}

# Run GridSearch
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring="r2")
grid_search.fit(X_train, y_train)

# Save best parameters
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)
