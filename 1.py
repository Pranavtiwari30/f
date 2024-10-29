import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the training and testing datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Define feature set and target variable for training data
X = train_data.drop(columns=['total_fare'])
y = train_data['total_fare']

# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize and train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions on validation set
y_val_pred = model.predict(X_val_scaled)

# Manually calculate Mean Squared Error and RMSE
mse = np.mean((y_val - y_val_pred) ** 2)
rmse = np.sqrt(mse)
r2 = 1 - (np.sum((y_val - y_val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

print(f"Validation MSE: {mse}")
print(f"Validation RMSE: {rmse}")
print(f"Validation R^2 Score: {r2}")

# Visualization 1: Actual vs. Predicted Total Fare
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.3, color='blue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel("Actual Total Fare")
plt.ylabel("Predicted Total Fare")
plt.title("Actual vs. Predicted Total Fare on Validation Set")
plt.show()

# Visualization 2: Residuals Plot with Reduced Sample and No KDE
residuals = y_val - y_val_pred
sample_residuals = residuals.sample(n=1000, random_state=42)  # Sample 1,000 points
plt.figure(figsize=(10, 6))
plt.hist(sample_residuals, bins=30, color='purple', alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residuals Distribution on Validation Set (Sampled)")
plt.show()

# Prepare the test data by scaling with the previously fitted scaler
X_test = test_data.drop(columns=['total_fare'])
X_test_scaled = scaler.transform(X_test)

# Predict on the test data
y_test_pred = model.predict(X_test_scaled)

# Compile results for comparison (predicted total fare)
test_results = pd.DataFrame({'Predicted Total Fare': y_test_pred})

# Display the first few rows of the prediction results
print(test_results.head())