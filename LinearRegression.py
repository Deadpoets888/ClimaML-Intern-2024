import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.DataFrame({
    'year': list(range(1990, 2021)),
    'methane': [
        528077.7, 534386.7, 535736.3, 539941.7, 544615.5, 555229.7, 565094.9, 574291.6, 582210.4,
        589663.7, 594662.8, 597332.6, 591021.9, 599191.2, 606634.5, 620079.6, 629406.7, 642567.2,
        651240.2, 651518.6, 658933.4, 666187.8, 665447.9, 669582.1, 674057.7, 678829.1, 684445.8,
        686785.3, 695493.6, 696431.8, 697654.7
    ]
})

# Create lagged features for the previous 3 years































def create_lagged_features(data, lags=3):
    lagged_data = data.copy()
    for lag in range(1, lags + 1):
        lagged_data[f'lag_{lag}'] = data['methane'].shift(lag)
    return lagged_data.dropna()

# Prepare the data with lagged features
data_with_lags = create_lagged_features(data)

# Features and target variable
X = data_with_lags[['lag_1', 'lag_2', 'lag_3']]
y = data_with_lags['methane']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Function to predict future values based on the last 3 years
def predict_future_values(last_known_values, years, model):
    future_predictions = []
    for year in years:
        pred_input = last_known_values[::-1]
        pred_methane = model.predict([pred_input])[0]
        future_predictions.append(pred_methane)
        last_known_values = np.roll(last_known_values, -1)
        last_known_values[-1] = pred_methane
    return future_predictions

# Predict future values year by year
future_years = [2021, 2022, 2023, 2025, 2030, 2040]
last_known_values = data_with_lags[['methane', 'lag_1', 'lag_2']].iloc[-1].values
future_predictions = predict_future_values(last_known_values, future_years, model)

# Plot the historical data
plt.figure(figsize=(10, 6))
plt.plot(data['year'], data['methane'], label='Historical Data', marker='o')

# Plot the future predictions
plt.plot(future_years, future_predictions, label='Future Predictions', color='green', marker='o')
plt.xlabel('Year')
plt.ylabel('Methane Emission')
plt.title('Methane Emissions Forecast')
plt.legend()
plt.grid(True)
plt.show()
ctual_values_2021_2023 = [697654.7, 696431.8, 695493.6]  # Dummy actual values for 2021, 2022, 2023
predicted_values_2021_2023 = future_predictions[:3]


# Display future predictions
print("Future Predictions:")
for year, emission in zip(future_years, future_predictions):
    print(f"Year: {int(year)}, Predicted Methane Emission: {emission:.2f}")
