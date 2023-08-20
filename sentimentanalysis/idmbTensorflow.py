import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the datasets
train_df = pd.read_csv('Train.csv')
valid_df = pd.read_csv('Valid.csv')
test_df = pd.read_csv('Test.csv')

# Combine train and validation datasets for tokenization
texts = pd.concat([train_df['text'], valid_df['text']], axis=0)

# Hyperparameters
max_words = 10000
max_sequence_length = 100
embedding_dim = 100

# Preprocess text data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
train_sequences = tokenizer.texts_to_sequences(train_df['text'])
valid_sequences = tokenizer.texts_to_sequences(valid_df['text'])
test_sequences = tokenizer.texts_to_sequences(test_df['text'])

word_index = tokenizer.word_index
x_train = pad_sequences(train_sequences, maxlen=max_sequence_length)
x_valid = pad_sequences(valid_sequences, maxlen=max_sequence_length)
x_test = pad_sequences(test_sequences, maxlen=max_sequence_length)

y_train = train_df['label'].values
y_valid = valid_df['label'].values
y_test = test_df['label'].values

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (positive/negative sentiment)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_valid, y_valid))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# ==================================
# Epoch 9/10
# 1250/1250 [==============================] - 27s 22ms/step - loss: 7.7376e-06 - accuracy: 1.0000 - val_loss: 0.9021 - val_accuracy: 0.8486
# Epoch 10/10
# 1250/1250 [==============================] - 27s 21ms/step - loss: 4.0237e-06 - accuracy: 1.0000 - val_loss: 0.9369 - val_accuracy: 0.8492
# 157/157 [==============================] - 0s 2ms/step - loss: 0.9381 - accuracy: 0.8482
# Test Accuracy: 0.8482000231742859
# ==================================
