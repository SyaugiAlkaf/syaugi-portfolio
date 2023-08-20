import pandas as pd

# Load the dataset
data_path = "sales.csv"
data = pd.read_csv(data_path)

# 1. Datetime Conversion
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y')
data['Ship Date'] = pd.to_datetime(data['Ship Date'], format='%d/%m/%Y')

# 2. Handling Missing Data
data['Postal Code'].fillna(data['Postal Code'].mode()[0], inplace=True)  # Filling with mode

# 3. Encoding Categorical Variables
categorical_features = ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# 4. Feature Engineering
data['Year'] = data['Order Date'].dt.year
data['Month'] = data['Order Date'].dt.month
data['Day'] = data['Order Date'].dt.day
data['Dayofweek'] = data['Order Date'].dt.dayofweek
data['Shipping Duration'] = (data['Ship Date'] - data['Order Date']).dt.days

# 5. Feature Selection
drop_columns = ['Row ID', 'Order ID', 'Customer ID', 'Product ID', 'Product Name', 'Customer Name', 'Order Date', 'Ship Date']
data.drop(columns=drop_columns, inplace=True)

print(data.head())
