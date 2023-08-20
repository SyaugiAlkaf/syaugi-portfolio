import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('sales.csv')

# Convert 'Order Date' to datetime format
data['Order Date'] = pd.to_datetime(data['Order Date'], dayfirst=True)

# Extract year, month, day
data['Year'] = data['Order Date'].dt.year
data['Month'] = data['Order Date'].dt.month
data['Day'] = data['Order Date'].dt.day

# Drop unnecessary columns
data = data[['Order Date', 'Sales']]

# Split the data into training and test sets
X = data[['Order Date']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ARIMA model
arima_model = ARIMA(y_train, order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Predict on test set
arima_predictions = arima_model_fit.forecast(len(y_test))

# Calculate the error for ARIMA model
arima_error = np.sqrt(mean_squared_error(y_test, arima_predictions))

print(f'ARIMA Root Mean Squared Error: {arima_error}')
