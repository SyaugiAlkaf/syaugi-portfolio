import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download necessary NLTK datasets (only need to run this once)
# nltk.download('punkt')
# nltk.download('stopwords')

# Load your own dataset files
train_df = pd.read_csv('Train.csv')
valid_df = pd.read_csv('Valid.csv')
test_df = pd.read_csv('Test.csv')

# Preprocess the dataset
stop_words = set(stopwords.words('english'))
def preprocess_review(review):
    return [word.lower() for word in word_tokenize(review) if word.isalpha() and word.lower() not in stop_words]

# Assuming your CSV files have 'text' and 'label' columns
train_data = [(preprocess_review(text), label) for text, label in zip(train_df['text'], train_df['label'])]
valid_data = [(preprocess_review(text), label) for text, label in zip(valid_df['text'], valid_df['label'])]
test_data = [(preprocess_review(text), label) for text, label in zip(test_df['text'], test_df['label'])]

# Convert the reviews to feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform([' '.join(words) for words, _ in train_data])
valid_features = vectorizer.transform([' '.join(words) for words, _ in valid_data])
test_features = vectorizer.transform([' '.join(words) for words, _ in test_data])

# Train the sentiment analysis model using Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(train_features, [label for _, label in train_data])

# Evaluate the model on the validation set
predicted_labels = classifier.predict(valid_features)
accuracy = accuracy_score([label for _, label in valid_data], predicted_labels)
print(f"Validation Accuracy: {accuracy:.2f}")

# Generate a classification report for the validation set
print("Validation Classification Report:")
print(classification_report([label for _, label in valid_data], predicted_labels))

# Evaluate the model on the test set
predicted_labels_test = classifier.predict(test_features)
accuracy_test = accuracy_score([label for _, label in test_data], predicted_labels_test)
print(f"Test Accuracy: {accuracy_test:.2f}")

# Generate a classification report for the test set
print("Test Classification Report:")
print(classification_report([label for _, label in test_data], predicted_labels_test))

# ===============================
# Validation Accuracy: 0.87
# Validation Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.86      0.88      0.87      2486
#            1       0.88      0.86      0.87      2514
#
#     accuracy                           0.87      5000
#    macro avg       0.87      0.87      0.87      5000
# weighted avg       0.87      0.87      0.87      5000
#
# Test Accuracy: 0.87
# Test Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.86      0.88      0.87      2495
#            1       0.88      0.86      0.87      2505
#
#     accuracy                           0.87      5000
#    macro avg       0.87      0.87      0.87      5000
# weighted avg       0.87      0.87      0.87      5000
# ===============================
