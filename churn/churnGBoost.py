import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('churn.csv')
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)
data.drop(['customerID'], axis=1, inplace=True)
data = pd.get_dummies(data, drop_first=True)

# Splitting the dataset into train and test sets
X = data.drop('Churn_Yes', axis=1)
y = data['Churn_Yes']

# Addressing Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model - trying GradientBoosting for a change
clf = GradientBoostingClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}
grid_clf = GridSearchCV(clf, param_grid, cv=5)
grid_clf.fit(X_train, y_train)

# Predictions
y_pred = grid_clf.predict(X_test)

# Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# ==========================================
# Accuracy: 0.8431245965138799
# Classification Report:
#                precision    recall  f1-score   support
#
#        False       0.86      0.83      0.84      1563
#         True       0.83      0.86      0.84      1535
#
#     accuracy                           0.84      3098
#    macro avg       0.84      0.84      0.84      3098
# weighted avg       0.84      0.84      0.84      3098
#
# Confusion Matrix:
#  [[1294  269]
#  [ 217 1318]]
# ==========================================
