import pandas as pd

# Load the dataset
dataset_path = './yt_toxic.csv'
df = pd.read_csv(dataset_path)

# Print the column names
print("Column Names:")
print(df.columns)
