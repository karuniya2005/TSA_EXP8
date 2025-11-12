# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 27.10.2025
### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# --- Load dataset ---
data = pd.read_csv("test.csv")

# Try to parse first column as datetime if possible
try:
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
    data.set_index(data.columns[0], inplace=True)
except:
    data.set_index(data.columns[0], inplace=True)

# Select first numeric column for analysis
target_col = data.select_dtypes(include=['number']).columns[0]
ts_data = data[target_col]

print("Selected column for forecasting:", target_col)
print("Data shape:", ts_data.shape)
print(ts_data.head())

# --- Plot original data ---
plt.figure(figsize=(12,6))
plt.plot(ts_data)
plt.title(f'{target_col} Over Time')
plt.xlabel('Time')
plt.ylabel(target_col)
plt.grid()
plt.show()

# --- Rolling Mean ---
rolling_mean_5 = ts_data.rolling(window=5).mean()
rolling_mean_10 = ts_data.rolling(window=10).mean()

plt.figure(figsize=(12,6))
plt.plot(ts_data, label='Original Data')
plt.plot(rolling_mean_5, label='MA window=5')
plt.plot(rolling_mean_10, label='MA window=10')
plt.legend()
plt.title(f'Moving Average - {target_col}')
plt.grid()
plt.show()

# --- Scaling the data ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1))
scaled_data = pd.Series(scaled_data.flatten(), index=ts_data.index)

# --- Train-Test Split ---
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# --- Exponential Smoothing Model (Additive Seasonality) ---
model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=11).fit()
test_predictions = model.forecast(len(test_data))

# --- Plot Predictions ---
plt.figure(figsize=(12,6))
plt.plot(train_data, label="Train Data")
plt.plot(test_data, label="Test Data")
plt.plot(test_predictions, label="Predictions")
plt.legend()
plt.title(f'Forecasting on {target_col} Data')
plt.show()

# --- RMSE ---
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print("RMSE:", rmse)

```

### OUTPUT:


<img width="1001" height="535" alt="image" src="https://github.com/user-attachments/assets/32319bed-8efa-40e3-a9dc-3b7dac23b6e8" />

<img width="980" height="529" alt="image" src="https://github.com/user-attachments/assets/a137d0a1-68e2-40c6-8af7-7fcae9e6d2d7" />

<img width="1014" height="549" alt="image" src="https://github.com/user-attachments/assets/8b951d8d-ac10-4439-bb19-c4f5e7480ba0" />




### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
