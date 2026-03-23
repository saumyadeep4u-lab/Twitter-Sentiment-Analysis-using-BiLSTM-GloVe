# Twitter-Sentiment-Analysis-using-BiLSTM-GloVe
This project focuses on building a deep learning-based sentiment analysis model to classify tweets into Positive, Neutral, and Negative sentiments.

The model leverages Natural Language Processing (NLP) techniques combined with a Bidirectional LSTM (BiLSTM) architecture and pre-trained GloVe embeddings to understand contextual meaning in text data.

🎯 Objectives
Perform sentiment classification on Twitter data
Handle noisy and unstructured text
Improve semantic understanding using word embeddings
Address class imbalance in real-world datasets

🧠 Model Architecture

The model is built using TensorFlow/Keras and includes:

Embedding Layer (GloVe 50d)
Bidirectional LSTM (128 units)
Dense Layer (ReLU activation)
Dropout Layer (0.5)
Output Layer (Softmax for 3 classes)

⚙️ Tech Stack
Python 🐍
TensorFlow / Keras
Scikit-learn
Pandas, NumPy
Joblib

📊 Data Preprocessing
Converted text to lowercase
Tokenization using Keras Tokenizer
Vocabulary size limited to 20,000 words
Sequence padding to max length = 100
Label encoding:
Negative → 0
Neutral → 1
Positive → 2

🔍 Key Features
✅ Pre-trained GloVe embeddings for semantic understanding
✅ Bidirectional LSTM for contextual learning
✅ Class weight balancing to handle imbalanced data
✅ EarlyStopping to prevent overfitting
✅ Tokenizer saved for deployment

📈 Key Findings
The dataset contains three sentiment classes: Positive, Neutral, and Negative.
Class imbalance was present and handled using class weights.
Tweets are highly unstructured and require preprocessing for meaningful analysis.
GloVe embeddings significantly improved model performance by capturing semantic relationships.
Bidirectional LSTM effectively captured context from both directions.
Positive and Negative sentiments were easier to classify compared to Neutral.
Neutral sentiment showed overlap with other classes, making it harder to predict.
EarlyStopping improved generalization and reduced overfitting.
Handling sarcasm and informal language remains a challenge.

📉 Model Training
Train-Test Split: 80% / 20%
Batch Size: 32
Epochs: 30 (with EarlyStopping)
Loss Function: Categorical Crossentropy
Optimizer: Adam

🌍 Real-World Applications
Brand sentiment monitoring
Customer feedback analysis
Social media trend tracking
Product review classification


