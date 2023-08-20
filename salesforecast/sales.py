import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv('sales.csv')

# Convert 'Order Date' to datetime format
data['Order Date'] = pd.to_datetime(data['Order Date'], dayfirst=True)

# Extract year, month, day
data['Year'] = data['Order Date'].dt.year
data['Month'] = data['Order Date'].dt.month
data['Day'] = data['Order Date'].dt.day

# Drop unnecessary columns
data = data[['Year', 'Month', 'Day', 'Sales']]

# Split the data into training and test sets
X = data[['Year', 'Month', 'Day']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate the error
error = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {error}')
