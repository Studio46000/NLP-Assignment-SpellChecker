from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from Levenshtein import distance as levenshtein_distance
import language_tool_python

app = Flask(__name__)

# Load the model
model_save_path = '/home/omarkhaled/code/omar-khaled-coder/NLP-Assignment-SpellChecker/text_classification_model.h5'
model = load_model(model_save_path)

# Load the tokenizer
tokenizer_save_path = '/home/omarkhaled/code/omar-khaled-coder/NLP-Assignment-SpellChecker/tokenizer.pkl'
with open(tokenizer_save_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize LanguageTool for grammar checking
tool = language_tool_python.LanguageTool('en-US')

# Define MAX_LENGTH
MAX_LENGTH = 100

# Load vocabulary from the tokenizer
vocab_words = list(tokenizer.word_index.keys())

# Function to calculate Levenshtein distance and correct the word
def correct_word(input_word, vocab_words):
    # Prevent correcting "I" or similar key words
    if input_word.lower() == "i":
        return "I"

    min_distance = float('inf')
    corrected_word = input_word

    # Use Levenshtein distance to find the closest word in the vocab
    for word in vocab_words:
        distance = levenshtein_distance(input_word.lower(), word)
        if distance < min_distance and distance <= 2:  # Correct if Levenshtein distance <= 2
            min_distance = distance
            corrected_word = word

    # Check for grammar-based suggestions using LanguageTool
    matches = tool.check(input_word)
    if matches:
        # If the grammar tool suggests a replacement, use it (like "red" to "read")
        for match in matches:
            if match.replacements:  # Use the first replacement suggestion
                corrected_word = match.replacements[0]

    # Maintain the capitalization format of the original word
    return corrected_word.capitalize() if input_word[0].isupper() else corrected_word




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

        if word != corrected_word:  # If there's a difference, mark it as an error
            errors.append({"index": idx, "original": word, "suggestion": corrected_word})

    # Join corrected words into a sentence
    corrected_sentence = " ".join(corrected_words)

    # Apply grammar corrections using LanguageTool
    grammar_matches = tool.check(corrected_sentence)
    for match in grammar_matches:
        if match.replacements:  # Apply the first suggestion
            corrected_sentence = corrected_sentence[:match.offset] + match.replacements[0] + corrected_sentence[match.offset + match.errorLength:]

    return corrected_sentence, errors



# Function to check grammar errors
def check_grammar_errors(text):
    matches = tool.check(text)
    grammar_errors = []
    for match in matches:
        # Extract the specific part of the text where the error occurs
        error_context = text[match.offset:match.offset + match.errorLength]

        # Append the detailed grammar error
        grammar_errors.append({
            "error": match.message,  # The error message from LanguageTool
            "suggestions": match.replacements if match.replacements else ["No suggestions available"],  # Suggestions from LanguageTool
            "context": error_context  # The specific text that caused the error
        })
    return grammar_errors


# Define the home route to serve the index.html
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to test input sentences via a POST request
@app.route('/correct_sentence', methods=['POST'])
def correct_sentence():
    data = request.json
    input_sentence = data['sentence']

    # First, check grammar errors before applying corrections
    grammar_errors = check_grammar_errors(input_sentence)

    # Then apply spelling corrections
    corrected_sentence, spelling_errors = test_input_sentence(input_sentence)

    # Return the result, including the original grammar errors (if any)
    return jsonify({
        'corrected_sentence': corrected_sentence,
        'errors': spelling_errors,
        'grammar_errors': grammar_errors  # Grammar errors found before corrections
    })
