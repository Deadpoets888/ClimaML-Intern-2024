import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Try loading the data from CSV file
try:
    data = pd.read_csv('AnnualFreshWater.csv')
except FileNotFoundError:
    print("Error: Could not find 'AnnualFreshWater.csv' file. Please check the file path.")
    exit()

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Check if there's any data loaded
if data.empty:
    print("Error: 'AnnualFreshWater.csv' file seems empty. Please ensure it contains data.")
    exit()

# Scatter plot of year vs methane emissions
plt.figure(figsize=(10, 6))
plt.scatter(data['year'], data['AnnualFreshWater'], color='blue', label='Actual Data')

# Features and target variable
X = data[['year']]
y = data['AnnualFreshWater']

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predicting future values
future_years = np.array([[2025], [2030], [2035], [2040], [2045], [2050]])
future_predictions = model.predict(future_years)

# Calculate predictions for the training data to evaluate model performance
y_pred = model.predict(X)

# Calculate RMSE and R²
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Plot the regression line
plt.plot(X, y_pred, color='red', label='Regression Line')

# Plot the future predictions (use a different marker style for clarity)
plt.scatter(future_years, future_predictions, color='green', marker='^', label='Future Predictions')

# Display future predictions
for year, emission in zip(future_years, future_predictions):
    print(f"Year: {year[0]}, Predicted Annual freshwater withdrawals, total (billion cubic meters): {emission}")

plt.xlabel('Year')
plt.ylabel('Annual freshwater withdrawals, total (billion cubic meters)')
plt.title('Annual freshwater withdrawals, total (billion cubic meters) Over Time with Linear Predictions')
plt.legend()
plt.grid(True)
plt.show()

