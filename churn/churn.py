# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('churn.csv')
print("\nData after loading:")
print(data.head())

# Convert TotalCharges to numeric, setting errors='coerce' to handle non-numeric values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Handle missing values if any
data.dropna(inplace=True)

# Drop customerID as it's likely not useful for prediction
data.drop(['customerID'], axis=1, inplace=True)

# Convert categorical columns to dummy variables
data = pd.get_dummies(data, drop_first=True)

print("\nData after preprocessing:")
print(data.head())

# Splitting the dataset into train and test sets
X = data.drop('Churn_Yes', axis=1)  # Use 'Churn_Yes' as the target after dummy encoding
y = data['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predicting on the test set
y_pred = clf.predict(X_test)

# Model Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# ========================================
# Accuracy: 0.7862559241706161
# Classification Report:
#                precision    recall  f1-score   support
#
#        False       0.83      0.90      0.86      1549
#         True       0.63      0.48      0.55       561
#
#     accuracy                           0.79      2110
#    macro avg       0.73      0.69      0.70      2110
# weighted avg       0.77      0.79      0.78      2110
#
# Confusion Matrix:
#  [[1387  162]
#  [ 289  272]]
# ========================================