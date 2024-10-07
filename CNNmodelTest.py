import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Levenshtein import distance as levenshtein_distance
import numpy as np
<<<<<<< Updated upstream

# Load the saved model
model_save_path = 'C:/Users/User/Desktop/saved_model/text_classification_model.h5' # Change the path where the model is located
model = load_model(model_save_path)
print("Model loaded successfully!")
=======
import tensorflow as tf
import nltk
import tkinter as tk
import keras
from nltk.corpus import brown
from difflib import get_close_matches
from collections import defaultdict
from nltk import bigrams
import string
import tkinter.scrolledtext as st
from keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

print(keras.__version__)
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
>>>>>>> Stashed changes

# Load the tokenizer
tokenizer_save_path = 'C:/Users/User/Desktop/saved_model/tokenizer.pkl' # Change the path where the tokenizer is located
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

# Function to test the model on user input
def test_input_sentence(input_sentence):
    # Tokenize and pad the input sentence
    input_seq = tokenizer.texts_to_sequences([input_sentence])
    input_padded = pad_sequences(input_seq, maxlen=MAX_LENGTH, padding='post')
    
    # Predict with the model
    prediction = model.predict(input_padded)
    predicted_label = (prediction > 0.5).astype("int32")
    
    if predicted_label[0] == 1:  # If there's an error
        words = input_sentence.split()
        corrected_sentence = []
        for word in words:
            corrected_word = correct_word(word, vocab_words)
            corrected_sentence.append(corrected_word)
        corrected_sentence = " ".join(corrected_sentence)
        print(f"Did you mean: \"{corrected_sentence}\"?")
    else:
        print("No errors found.")

# Test the system with a user input
user_input = "detp leawning can indeed be complix, especialpy when working with large datasets and intrikate model arhitectures"
test_input_sentence(user_input)
