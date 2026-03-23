import pandas as pd
import numpy as np
import re
import emoji

from sklearn.model_selection import train_test_split
from joblib import dump

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# -------------------------------
# Emoji Processing
# -------------------------------
def emoji_to_text(text):
    text = emoji.demojize(text)
    text = re.sub(r':([a-z_]+):', r'\1', text)
    return text


def clean_text(text):
    text = str(text).lower()

    # Convert emojis
    emoji_text = emoji_to_text(text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Combine + boost emoji importance
    text = text + " " + emoji_text + " " + emoji_text

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv(
    r"C:\Users\hp\Downloads\BIA\Twitter_Sentiment_Analysis\Data\cleaned_data.csv"
)

# Preprocess text
data['clean_text'] = data['text'].apply(clean_text)

# -------------------------------
# Label mapping
# -------------------------------
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

data = data.dropna(subset=['sentiment'])
y = data['sentiment'].map(label_map)
X = data['clean_text']

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Tokenization
# -------------------------------
tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 80

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Save tokenizer
dump(tokenizer, 'tokenizer.joblib')

# -------------------------------
# One-hot encoding
# -------------------------------
num_classes = 3

y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

# -------------------------------
# Load GloVe embeddings
# -------------------------------
print("Loading GloVe embeddings...")

embeddings_index = {}
glove_path = r"C:\Users\hp\Downloads\BIA\Twitter_Sentiment_Analysis\glove.6B.50d.txt"

with open(glove_path, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_dim = 50
word_index = tokenizer.word_index
num_words = min(15000, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i < num_words:
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[i] = vector

print("GloVe loaded successfully!")

# -------------------------------
# Model (Best Balanced Version)
# -------------------------------
model = tf.keras.Sequential([
    Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=True
    ),

    Bidirectional(LSTM(
        96,
        dropout=0.3,
        recurrent_dropout=0.3
    )),

    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu'),

    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# Class weights
# -------------------------------
class_counts = y.value_counts().to_dict()
total = sum(class_counts.values())

class_weight = {
    cls: total / (num_classes * count)
    for cls, count in class_counts.items()
}

print("Class weights:", class_weight)

# -------------------------------
# Callbacks (IMPORTANT)
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,   # 🔥 stops at best epoch (~3)
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# -------------------------------
# Training
# -------------------------------
history = model.fit(
    X_train_pad, y_train_onehot,
    epochs=30,
    batch_size=32,
    validation_data=(X_test_pad, y_test_onehot),
    class_weight=class_weight,
    callbacks=[early_stop, checkpoint]
)

# -------------------------------
# Save final model
# -------------------------------
model.save("sentiment_model.h5")

print("✅ Model & tokenizer saved successfully")