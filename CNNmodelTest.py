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

# Preprocess words (lowercase, remove punctuation)
def preprocess_word(word):
    return word.lower().translate(str.maketrans('', '', string.punctuation))

# Load CNN Model
cnn_model_path = 'C:/Users/User/Desktop/saved_model/text_classification_model.h5'
cnn_model = tf.keras.models.load_model(cnn_model_path)
print("CNN Model loaded successfully!")

# Load the tokenizer
tokenizer_path = 'C:/Users/User/Desktop/saved_model/tokenizer.pkl'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set max sequence length for padding
MAX_LENGTH = 100

# Load GloVe Embeddings
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

# Get semantically similar words using GloVe
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

# Test the input sentence with CNN model for non-word errors
def test_input_sentence(input_sentence):
    input_seq = tokenizer.texts_to_sequences([input_sentence])
    input_padded = pad_sequences(input_seq, maxlen=MAX_LENGTH, padding='post')
    prediction = cnn_model.predict(input_padded)
    return (prediction > 0.5).astype("int32")[0][0]

# Detect non-words using closest matches from vocabulary (edit distance)
def correct_non_word_error(word, vocabulary, n=5):
    word = preprocess_word(word)
    return get_close_matches(word, vocabulary, n=n)

# Build a bigram model for real-word errors
def build_bigram_model(words):
    bigram_model = defaultdict(lambda: defaultdict(lambda: 0))
    for w1, w2 in bigrams(words):
        bigram_model[w1][w2] += 1
    return bigram_model

bigram_model = build_bigram_model([preprocess_word(word) for word in brown_words if word.isalpha()])

# Check real-word errors using GPT/BERT
def check_real_word_errors(text):
    tokens = nltk.word_tokenize(text)
    suggestions = []
    for i, word in enumerate(tokens):
        masked_text = text.replace(word, "[MASK]", 1)
        predictions = real_word_model(masked_text)
        top_predictions = [pred['token_str'] for pred in predictions[:5]]
        suggestions.append((word, top_predictions))
    return suggestions

# Spell check for non-words using CNN and GloVe
def spell_check_non_words(text):
    tokens = nltk.word_tokenize(text)
    errors = []
    for word in tokens:
        word = preprocess_word(word)
        if word not in vocabulary:
            suggestions = correct_non_word_error(word, vocabulary)
            if suggestions:
                errors.append((word, suggestions))
    return errors

# GUI Setup for Non-word and Real-word error checking
class SpellCheckerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spell Checker")
        self.root.geometry("600x700")

        # Non-word spell checking section
        tk.Label(root, text="Non-word Spell Checker").pack()
        self.non_word_text = st.ScrolledText(root, height=7, width=70)
        self.non_word_text.pack(padx=10, pady=5)

        # Real-word spell checking section
        tk.Label(root, text="Real-word Spell Checker").pack()
        self.real_word_text = st.ScrolledText(root, height=7, width=70)
        self.real_word_text.pack(padx=10, pady=5)

        # Buttons
        tk.Button(root, text="Check Non-word", command=self.check_non_words).pack(pady=5)
        tk.Button(root, text="Check Real-word", command=self.check_real_words).pack(pady=5)

        # Suggestions
        self.non_word_suggestions = tk.Text(root, height=9, width=70, state='disabled')
        self.non_word_suggestions.pack(pady=5)
        self.real_word_suggestions = tk.Text(root, height=9, width=70, state='disabled')
        self.real_word_suggestions.pack(pady=5)

    def check_non_words(self):
        self.non_word_suggestions.configure(state='normal')
        self.non_word_suggestions.delete(1.0, tk.END)
        input_text = self.non_word_text.get("1.0", "end-1c")
        errors = spell_check_non_words(input_text)
        if errors:
            for word, suggestions in errors:
                self.non_word_suggestions.insert(tk.END, f"Word: {word}, Suggestions: {', '.join(suggestions)}\n")
        else:
            self.non_word_suggestions.insert(tk.END, "No non-word errors found!")
        self.non_word_suggestions.configure(state='disabled')

    def check_real_words(self):
        self.real_word_suggestions.configure(state='normal')
        self.real_word_suggestions.delete(1.0, tk.END)
        input_text = self.real_word_text.get("1.0", "end-1c")
        suggestions = check_real_word_errors(input_text)
        if suggestions:
            for word, top_predictions in suggestions:
                self.real_word_suggestions.insert(tk.END, f"Word: {word}, Suggestions: {', '.join(top_predictions)}\n")
        else:
            self.real_word_suggestions.insert(tk.END, "No real-word errors found!")
        self.real_word_suggestions.configure(state='disabled')

# Initialize GUI
root = tk.Tk()
gui = SpellCheckerGUI(root)
root.mainloop()
