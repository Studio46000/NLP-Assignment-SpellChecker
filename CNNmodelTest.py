import pickle
import numpy as np
import tensorflow as tf
import nltk
import tkinter as tk
from nltk.corpus import brown
from difflib import get_close_matches
from collections import defaultdict
from nltk import bigrams
import string
import tkinter.scrolledtext as st
from keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# Download necessary resources
nltk.download('brown')
nltk.download('punkt')

# Preprocess Brown Corpus
brown_words = brown.words()
vocabulary = set(word.lower() for word in brown_words if word.isalpha())

# Helper function to preprocess word
def preprocess_word(word):
    word = word.lower()
    return word.translate(str.maketrans('', '', string.punctuation))

# CNN MODEL LOADING
cnn_model_path = 'C:/Users/User/Desktop/saved_model/text_classification_model.h5' 
cnn_model = tf.keras.models.load_model(cnn_model_path)
print("CNN Model loaded successfully!")

# Load the tokenizer
tokenizer_path = 'C:/Users/User/Desktop/saved_model/tokenizer.pkl'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set maximum sequence length for padding
MAX_LENGTH = 100

# GloVe EMBEDDING LOADING
def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_file_path = 'C:/Users/User/Desktop/saved_model/glove.6B.100d.txt'
embeddings_index = load_glove_embeddings(glove_file_path)
print("GloVe Embeddings loaded successfully!")

# GPT/BERT for Real-word error detection
real_word_model = pipeline("fill-mask", model="bert-base-uncased")

# FUNCTION: Use GloVe to find semantically similar words
def get_semantically_similar_words(word, embeddings_index, n=3):
    if word not in embeddings_index:
        return []
    
    word_vector = embeddings_index[word]
    similarities = {}

    for other_word, other_vector in embeddings_index.items():
        cosine_similarity = np.dot(word_vector, other_vector) / (np.linalg.norm(word_vector) * np.linalg.norm(other_vector))
        similarities[other_word] = cosine_similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_similarities[:n]]

# FUNCTION: Test the input sentence with CNN model (for non-word errors)
def test_input_sentence(input_sentence):
    input_seq = tokenizer.texts_to_sequences([input_sentence])
    input_padded = pad_sequences(input_seq, maxlen=MAX_LENGTH, padding='post')

    prediction = cnn_model.predict(input_padded)
    predicted_label = (prediction > 0.5).astype("int32")
    return predicted_label[0][0]  # 0: no error, 1: error

# Detect non-words using closest matches from vocabulary (edit distance)
def correct_non_word_error(word, vocabulary, n=5):
    word = preprocess_word(word)
    return get_close_matches(word, vocabulary, n=n)

# Bigram model for real-word errors
def build_bigram_model(words):
    bigram_model = defaultdict(lambda: defaultdict(lambda: 0))
    for w1, w2 in bigrams(words):
        bigram_model[w1][w2] += 1
    return bigram_model

bigram_model = build_bigram_model([preprocess_word(word) for word in brown_words if word.isalpha()])

# REAL-WORD CHECK FUNCTION using GPT/BERT
def check_real_word_errors(text):
    tokens = nltk.word_tokenize(text)
    suggestions = []
    for i, word in enumerate(tokens):
        masked_text = text.replace(word, "[MASK]", 1)
        predictions = real_word_model(masked_text)
        # Extract the top 5 predictions
        top_predictions = [pred['token_str'] for pred in predictions[:5]]  # Top 5 predictions
        suggestions.append((word, top_predictions))
    return suggestions

# SPELL CHECK FUNCTION for non-words (CNN and GloVe)
def spell_check_non_words(text):
    tokens = nltk.word_tokenize(text)
    errors = []
    
    for i in range(len(tokens)):
        word = preprocess_word(tokens[i])

        # Non-word errors detection
        if word not in vocabulary:  # Only process if word is not in the vocabulary
            suggestions = correct_non_word_error(word, vocabulary)
            if suggestions:
                errors.append((word, suggestions))
    
    return errors


# GUI with separate sections for Non-word and Real-word errors
class SpellCheckerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spell Checker")

        # Set a smaller window size
        self.root.geometry("600x700")  # Set the window size to fit better on the desktop

        # Non-word spell checking section
        self.non_word_label = tk.Label(self.root, text="Non-word Spell Checker")
        self.non_word_label.pack()
        self.non_word_text_editor = st.ScrolledText(self.root, height=7, width=70)  # Adjusted size
        self.non_word_text_editor.pack(padx=10, pady=5)

        # Real-word spell checking section
        self.real_word_label = tk.Label(self.root, text="Real-word Spell Checker")
        self.real_word_label.pack()
        self.real_word_text_editor = st.ScrolledText(self.root, height=7, width=70)  # Adjusted size
        self.real_word_text_editor.pack(padx=10, pady=5)

        # Buttons
        self.check_non_word_button = tk.Button(self.root, text="Check Non-word Spelling", command=self.check_non_words)
        self.check_non_word_button.pack(pady=5)

        self.check_real_word_button = tk.Button(self.root, text="Check Real-word Spelling", command=self.check_real_words)
        self.check_real_word_button.pack(pady=5)

        # Suggestions box for non-words
        self.non_word_suggestions_box = tk.Text(self.root, height=9, width=70, state='disabled')  # Adjusted size
        self.non_word_suggestions_box.pack(pady=5)

        # Suggestions box for real-words
        self.real_word_suggestions_box = tk.Text(self.root, height=9, width=70, state='disabled')  # Adjusted size
        self.real_word_suggestions_box.pack(pady=5)

    def check_non_words(self):
        # Clear previous suggestions
        self.non_word_suggestions_box.configure(state='normal')
        self.non_word_suggestions_box.delete(1.0, tk.END)

        # Get input text
        input_text = self.non_word_text_editor.get("1.0", "end-1c")

        # Run spell check
        errors = spell_check_non_words(input_text)

        if errors:
            for word, suggestions in errors:
                self.non_word_suggestions_box.insert(tk.END, f"Word: {word}, Suggestions: {', '.join(suggestions)}\n")
        else:
            self.non_word_suggestions_box.insert(tk.END, "No non-word errors found!")
        
        self.non_word_suggestions_box.configure(state='disabled')

    def check_real_words(self):
        # Clear previous suggestions
        self.real_word_suggestions_box.configure(state='normal')
        self.real_word_suggestions_box.delete(1.0, tk.END)

        # Get input text
        input_text = self.real_word_text_editor.get("1.0", "end-1c")

        # Run real-word check
        suggestions = check_real_word_errors(input_text)

        if suggestions:
            for word, top_predictions in suggestions:
                self.real_word_suggestions_box.insert(tk.END, f"Word: {word}, Suggestions: {', '.join(top_predictions)}\n")
        else:
            self.real_word_suggestions_box.insert(tk.END, "No real-word errors found!")
        
        self.real_word_suggestions_box.configure(state='disabled')

# Initialize GUI
root = tk.Tk()
gui = SpellCheckerGUI(root)
root.mainloop()
