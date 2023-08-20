import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the dataset
dataset_path = './yt_toxic.csv'
df = pd.read_csv(dataset_path)

# Select relevant features and labels
comments = df['Text'].tolist()  # Use the correct column name 'Text' for comment text
subclassifications = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 'IsObscene',
                      'IsHatespeech', 'IsRacist', 'IsNationalist', 'IsSexist', 'IsHomophobic',
                      'IsReligiousHate', 'IsRadicalism']
labels = df[subclassifications].values  # Use the subclassifications column names from your dataset

# Tokenize the text
vocab_size = 10000  # Adjust as needed
max_seq_length = 150  # Adjust as needed
embedding_dim = 100  # Set an appropriate value

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(comments)
word_index = tokenizer.word_index

# Convert text to sequences and pad sequences
sequences = tokenizer.texts_to_sequences(comments)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, truncating='post', padding='post')

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=64))
model.add(Dense(units=len(subclassifications), activation='sigmoid'))  # Sigmoid for multi-label classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
num_epochs = 10  # Adjust as needed
batch_size = 32  # Adjust as needed

model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Make predictions
predictions = model.predict(X_test)

# Calculate the average probability of toxicity
toxicity_probabilities = predictions[:, 0]  # 'IsToxic' probability column
average_toxicity_probability = toxicity_probabilities.mean()

print(f"Average Probability of Toxicity: {average_toxicity_probability:.4f}")

# Save the model for future use
model.save('toxicity_classifier.h5')

print("Code execution completed.")
