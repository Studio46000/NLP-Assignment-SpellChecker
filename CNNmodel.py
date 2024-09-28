import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Embedding, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'/Users/amirrulrasyid/Amirrul (Asia Pacific University)/Semester 2 - Jul 2024/Natural Language Processing/Assignment/Program/NLPmodel/NLP-Assignment-SpellChecker/wiked_error_correction.csv'
df = pd.read_csv(file_path)

erroneous_sentences = df['Error'].tolist()
corrected_sentences = df['Correct'].tolist()

# Combine the sentences for tokenizer fitting
all_sentences = erroneous_sentences + corrected_sentences

# Initialize the tokenizer and fit on the combined raw sentences
VOCAB_SIZE = 100000
MAX_LENGTH = 100
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(all_sentences)

# Save the tokenizer
import pickle
tokenizer_save_path = '/Users/amirrulrasyid/Amirrul (Asia Pacific University)/Semester 2 - Jul 2024/Natural Language Processing/Assignment/Program/NLPmodel/NLP-Assignment-SpellChecker/tokenizer.pkl'
with open(tokenizer_save_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert the erroneous and corrected sentences to sequences
error_sequences = tokenizer.texts_to_sequences(erroneous_sentences)
correct_sequences = tokenizer.texts_to_sequences(corrected_sentences)

# Pad the sequences
X_error = pad_sequences(error_sequences, maxlen=MAX_LENGTH, padding='post')
X_correct = pad_sequences(correct_sequences, maxlen=MAX_LENGTH, padding='post')

# Labels: 1 for error, 0 for correct
y_error = [1] * len(X_error)
y_correct = [0] * len(X_correct)

# Combine data and labels
X = np.vstack((X_error, X_correct))
y = np.hstack((y_error, y_correct))

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
EMBEDDING_DIM = 100
GLOVE_FILE = r'/Users/amirrulrasyid/Amirrul (Asia Pacific University)/Semester 2 - Jul 2024/Natural Language Processing/Assignment/Program/NLPmodel/NLP-Assignment-SpellChecker/glove.6B.100d.txt'

# Load GloVe embeddings
embedding_index = {}
with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embedding_vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = embedding_vector

# Prepare embedding matrix
word_index = tokenizer.word_index
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < VOCAB_SIZE:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Create the CNN-LSTM hybrid model
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LENGTH, 
                    weights=[embedding_matrix], trainable=False))
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
model.fit(X_train, y_train, epochs=50, batch_size=256, validation_split=0.4, callbacks=[early_stopping, lr_scheduler])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Predict on the test data
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")  # Convert probabilities to binary 0 or 1

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Save the model
model_save_path = '/Users/amirrulrasyid/Amirrul (Asia Pacific University)/Semester 2 - Jul 2024/Natural Language Processing/Assignment/Program/NLPmodel/NLP-Assignment-SpellChecker/text_classification_model.h5'
model.save(model_save_path)
