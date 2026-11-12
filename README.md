# 🧠 End-to-End Sentiment Analysis System (BiLSTM + GloVe)

An end-to-end Sentiment Analysis system that classifies text into **Positive**, **Neutral**, or **Negative** using a Deep Learning model (BiLSTM with pre-trained GloVe embeddings), hybrid rule-based NLP enhancements, and an interactive Streamlit web app.

🌐 **Live Demo:** [Try the Streamlit Web App Here](https://YOUR-STREAMLIT-LINK.streamlit.app)

---

## 🚀 Features
- 🔥 **Deep Learning Model:** Bidirectional LSTM (BiLSTM) for context-aware sequence modeling
- 📊 **Pre-trained Embeddings:** 50-dimensional GloVe word vectors
- 😊 **Emoji-Aware Detection:** Converts and weights emojis for social media accuracy
- 🤖 **Hybrid Rule-Based + ML System:** Custom logic for sarcasm and edge cases
- ⚖️ **Imbalance Handling:** Uses class weights to prevent majority-class bias
- 💾 **Training Callbacks:** Model checkpointing and early stopping to prevent overfitting
- 🌐 **Interactive Web App:** Deployed on Streamlit for real-time predictions

---

## 🧠 Model Architecture
- **Embedding Layer:** Pre-trained GloVe (50D)
- **Recurrent Layer:** Bidirectional LSTM (96 units)
- **Regularization:** Dense layers with L2 Regularization + Dropout (0.5)
- **Output Layer:** Softmax activation (3 classes: Positive, Neutral, Negative)

---

## 🔄 Workflow

### 1. Data Preprocessing
- Lowercasing and noise removal (URLs, special characters, punctuation)
- Emoji-to-text conversion
- Emoji sentiment boosting

### 2. Text Representation & Embeddings
- Tokenization and sequence padding
- Mapping vocabulary to pre-trained 50D GloVe embeddings

### 3. Model Training
- BiLSTM training with class weights
- Early stopping and best-model checkpointing

### 4. Prediction Enhancement (Post-Processing)
- Emoji sentiment overrides
- Keyword-based correction
- Sarcasm detection logic

### 5. Deployment
- Interactive Streamlit UI for real-time user input and confidence visualization

---

## 📈 Key Highlights
- Improved generalization using pre-trained **GloVe embeddings**
- Stronger real-world social media performance with **emoji handling**
- Smart handling of **sarcasm and implicit sentiment**
- Production-ready deployment with **Streamlit Community Cloud**

---

## 💼 Business Implications
This project demonstrates how Sentiment Analysis drives measurable business value across industries:

- **📊 Customer Insights & Decision Making:** Automatically analyzes feedback from social media (e.g., X/Twitter) to track satisfaction trends in real time and guide product improvements.
- **🛎️ Customer Support Optimization:** Detects negative sentiment instantly to prioritize urgent complaints, bugs, or service failures and improve customer experience (CX).
- **📈 Brand Monitoring & Reputation Management:** Tracks brand perception across high-volume user content and flags PR risks or viral negative trends early.
- **🎯 Marketing & Campaign Analysis:** Evaluates audience reactions to product launches and promotions to refine targeting and messaging strategies.
- **🤖 Scalable Automation:** Replaces manual review of thousands of comments with a cost-effective pipeline that integrates into dashboards or APIs.

### 🚀 Potential Industry Use Cases
- **E-commerce:** Product review analysis
- **Telecom:** Automated service complaint triage
- **Banking & Finance:** App and customer feedback tracking
- **SaaS Products:** User experience monitoring
- **Media & Entertainment:** Audience reaction tracking

---

## 💻 How to Run Locally

1. **Clone the repository:** `git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git`
2. **Open the project folder:** `cd YOUR-REPO-NAME`
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Launch the Streamlit app:** `streamlit run app.py`
