import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Convert 'date' column to datetime if it exists
if 'date' in X_train.columns:
    X_train['date'] = pd.to_datetime(X_train['date'])
    X_test['date'] = pd.to_datetime(X_test['date'])

    # Extract useful datetime features
    X_train['year'] = X_train['date'].dt.year
    X_train['month'] = X_train['date'].dt.month
    X_train['day'] = X_train['date'].dt.day
    X_train['hour'] = X_train['date'].dt.hour

    X_test['year'] = X_test['date'].dt.year
    X_test['month'] = X_test['date'].dt.month
    X_test['day'] = X_test['date'].dt.day
    X_test['hour'] = X_test['date'].dt.hour

    # Drop the original 'date' column
    X_train.drop(columns=['date'], inplace=True)
    X_test.drop(columns=['date'], inplace=True)

# Ensure feature consistency: Keep only numeric columns
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler as training

# Convert back to DataFrame to maintain column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save processed data
X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)
