import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hyperparameters
vocab_size = 10000
embedding_dim = 256
hidden_units = 512
max_decoder_seq_length = 20

# Sample dataset
questions = ["hello", "how are you", "what is your name", "exit"]
answers = ["Hi there!", "I'm good, thanks.", "I'm a chatbot.", "Goodbye!"]

# Tokenization and preprocessing
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(questions + answers)
questions_seqs = tokenizer.texts_to_sequences(questions)
answers_seqs = tokenizer.texts_to_sequences(answers)
max_seq_length = max(max(len(seq) for seq in questions_seqs), max(len(seq) for seq in answers_seqs))
questions_padded = pad_sequences(questions_seqs, maxlen=max_seq_length, padding='post')
answers_padded = pad_sequences(answers_seqs, maxlen=max_seq_length, padding='post')

# Build the model using functional API
encoder_inputs = Input(shape=(max_seq_length,))
encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(hidden_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_seq_length,))
decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
attention = Attention()([decoder_outputs, encoder_outputs])
decoder_combined = Concatenate(axis=-1)([decoder_outputs, attention])
decoder_dense = Dense(vocab_size, activation='softmax')
output = decoder_dense(decoder_combined)

model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
model.fit([questions_padded, questions_padded], answers_padded, epochs=10, batch_size=32)


# Inference
def chatbot_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_seq, maxlen=max_seq_length, padding='post')

    initial_states = encoder_model.predict(input_padded)  # Use encoder_model for prediction
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<sos>']  # Token ID for start-of-sequence

    stop_condition = False
    response = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + initial_states)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word[sampled_token_index]
        response += sampled_word + " "

        if sampled_word == "<eos>" or len(response.split()) > max_decoder_seq_length:
            stop_condition = True

        target_seq[0, 0] = sampled_token_index
        initial_states = [h, c]

    return response


# User interaction loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)
