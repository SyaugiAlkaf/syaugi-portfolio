import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the data from the CSV file
data = pd.read_csv('churn.csv')

# Convert TotalCharges to numeric, setting errors='coerce' to handle non-numeric values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Handle missing values if any
data.dropna(inplace=True)

# Drop customerID as it's likely not useful for prediction
data.drop(['customerID'], axis=1, inplace=True)

# Separate features (X) and target variable (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Convert categorical columns to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Get the column names after one-hot encoding
encoded_columns = X.columns.tolist()

# Convert categorical variables to one-hot encoding
encoder = OneHotEncoder(sparse=False, drop='first')
X_categorical = encoder.fit_transform(data[X])
X_categorical_df = pd.DataFrame(X_categorical, columns=encoder.get_feature_names_out(X))

# Combine one-hot encoded categorical features with numerical features
X_combined = pd.concat([X.drop(X, axis=1), X_categorical_df], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Scale the numerical features for better convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Sequential model
model = tf.keras.Sequential()

# Add input layer with Batch Normalization
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())

# Add additional hidden layers with Dropout and Batch Normalization
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Add the output layer (since it's a binary classification problem)
model.add(Dense(1, activation='sigmoid'))

# Compile the model with Adam optimizer and binary cross-entropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implement Early Stopping to prevent overfitting
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

# Train the model
epochs = 50
batch_size = 32

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
