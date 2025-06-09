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

# Calculate the 3-year rolling average
data['3_year_avg'] = data['methane'].rolling(window=3).mean()

# Plot the historical data
plt.figure(figsize=(10, 6))
plt.plot(data['year'], data['methane'], label='Historical Data', marker='o')

# Plot the 3-year rolling average
plt.plot(data['year'], data['3_year_avg'], label='3-Year Rolling Average', color='red')

plt.xlabel('Year')
plt.ylabel('Methane Emission')
plt.title('Methane Emissions and 3-Year Rolling Average')
plt.legend()
plt.grid(True)
plt.show()
