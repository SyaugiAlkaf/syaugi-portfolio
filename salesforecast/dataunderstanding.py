import pandas as pd

# Load the dataset
data_path = "sales.csv"  # Adjust the path accordingly
data = pd.read_csv(data_path)

# Display data types
print(data.dtypes)

# Display first few rows
print(data.head())
