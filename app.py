from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from Levenshtein import distance as levenshtein_distance

app = Flask(__name__)

# Load the model
model_save_path = '/home/omarkhaled/code/omar-khaled-coder/NLP-Assignment-SpellChecker/text_classification_model.h5'
model = load_model(model_save_path)

# Load the tokenizer
tokenizer_save_path = '/home/omarkhaled/code/omar-khaled-coder/NLP-Assignment-SpellChecker/tokenizer.pkl'
with open(tokenizer_save_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define MAX_LENGTH
MAX_LENGTH = 100

# Load vocabulary from the tokenizer
vocab_words = list(tokenizer.word_index.keys())

# Function to calculate Levenshtein distance and correct the word
def correct_word(input_word, vocab_words):
    min_distance = float('inf')
    corrected_word = input_word
    for word in vocab_words:
        distance = levenshtein_distance(input_word, word)
        if distance < min_distance:
            min_distance = distance
            corrected_word = word
    return corrected_word

# Function to test and correct the input sentence
def test_input_sentence(input_sentence):
    input_seq = tokenizer.texts_to_sequences([input_sentence])
    input_padded = pad_sequences(input_seq, maxlen=MAX_LENGTH, padding='post')

    prediction = model.predict(input_padded)

    # Debugging outputs
    print("Model prediction:", prediction)

    predicted_label = (prediction > 0.5).astype("int32")
    print("Predicted label:", predicted_label)

    corrected_words = []
    errors = []
    words = input_sentence.split()

    for idx, word in enumerate(words):
        corrected_word = correct_word(word, vocab_words)
        corrected_words.append(corrected_word)

        if word != corrected_word:  # If there’s a difference, mark it as an error
            errors.append({"index": idx, "original": word, "suggestion": corrected_word})

    return input_sentence, errors

# Define the home route to serve the index.html
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to test input sentences via a POST request
@app.route('/correct_sentence', methods=['POST'])
def correct_sentence():
    data = request.json
    input_sentence = data['sentence']
    original_sentence, errors = test_input_sentence(input_sentence)
    return jsonify({'original_sentence': original_sentence, 'errors': errors})

if __name__ == '__main__':
    app.run(debug=True)
