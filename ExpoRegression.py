import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Try loading the data from CSV file
try:
    data = pd.read_csv('methane.csv')
except FileNotFoundError:
    print("Error: Could not find 'methane.csv' file. Please check the file path.")
    exit()

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Check if there's any data loaded
if data.empty:
    print("Error: 'methane.csv' file seems empty. Please ensure it contains data.")
    exit()

# Scatter plot of year vs methane emissions
plt.figure(figsize=(10, 6))
plt.scatter(data['year'], data['methane'], color='blue', label='Actual Data')

# Features and target variable
X = data[['year']]
y = data['methane']

# Apply logarithm transformation to the target variable
y_log = np.log(y)

# Create the model
model = LinearRegression()

# Train the model on the transformed data
model.fit(X, y_log)

# Calculate predictions for the training data and transform back to the original scale
y_log_pred = model.predict(X)
y_pred = np.exp(y_log_pred)

# Calculate RMSE and R²
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Predicting future values and transform back to the original scale
future_years = np.array([[2025], [2030], [2035], [2040], [2045], [2050]])
future_log_predictions = model.predict(future_years)
future_predictions = np.exp(future_log_predictions)

# Plot the exponential regression line
X_range = np.linspace(X.min(), future_years.max(), 500).reshape(-1, 1)
y_range_log_pred = model.predict(X_range)
y_range_pred = np.exp(y_range_log_pred)
plt.plot(X_range, y_range_pred, color='red', label='Exponential Regression Line')

# Plot the future predictions (use a different marker style for clarity)
plt.scatter(future_years, future_predictions, color='green', marker='^', label='Future Predictions')

# Display future predictions
for year, emission in zip(future_years, future_predictions):
    print(f"Year: {year[0]}, Predicted Methane Emission: {emission}")

plt.xlabel('Year')
plt.ylabel('Methane Emission')
plt.title('Methane Emissions Over Time with Exponential Regression Predictions')
plt.legend()
plt.grid(True)
plt.show()
