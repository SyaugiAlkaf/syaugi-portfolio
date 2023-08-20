import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = load_model('toxicity_classifier.h5')  # Replace with the actual path to your model file

# Initialize the tokenizer (with the same parameters you used during training)
vocab_size = 10000  # Adjust as needed
max_seq_length = 150  # Adjust as needed
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts([])  # No need to fit on any text here


def get_toxic_prob(comment):
    # Tokenize and pad the comment
    test_sequence = tokenizer.texts_to_sequences([comment])
    padded_test_sequence = pad_sequences(test_sequence, maxlen=max_seq_length, truncating='post', padding='post')

    # Make predictions using the model
    predictions = model.predict(padded_test_sequence)

    # Return probability of toxicity only
    toxicity_probability = predictions[0][0]

    return toxicity_probability


# Example usage
input_comment = "i hate you so much"
toxicity_probability = get_toxic_prob(input_comment)

print(f"Probability of toxicity: {toxicity_probability:.4f}")
