# 🎬 IMDb Movie Review Sentiment Analysis

This project demonstrates how to build a binary text classification model using TensorFlow and the IMDb Reviews dataset. The model predicts whether a given movie review is **positive** or **negative**.

---

## 📌 Objective

To use Natural Language Processing (NLP) and Deep Learning to classify movie reviews into positive or negative sentiments using the **IMDb Reviews dataset**.

---

## 🧰 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **TensorFlow Datasets (TFDS)**
- **Jupyter Notebook**

---

## 📁 Dataset

- **IMDb Movie Reviews** from [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
- 50,000 movie reviews split into:
  - **25,000** training
  - **25,000** testing

Each review is labeled as either:
- **1** (Positive)
- **0** (Negative)

---

## 🚀 Workflow

1. **Load the IMDb dataset** using `tfds.load()`.
2. **Preprocess the data**:
   - Tokenization
   - Padding sequences
   - Word encoding
3. **Build a neural network** with:
   - `Embedding` layer
   - `Flatten` layer
   - Dense hidden layers
   - Final `sigmoid` output layer
4. **Compile** the model using:
   - `binary_crossentropy` loss
   - `adam` optimizer
   - `accuracy` as a metric
5. **Train** the model for multiple epochs and evaluate on test data.
6. **Visualize results** and decode reviews to interpret predictions.

---

## 🧠 Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

📚 Learnings
Handling large text datasets using TensorFlow
Understanding tokenization, word embedding, and padding
Creating and training a simple text classification model

Sample Output
Review: This was an absolutely terrible movie...
Predicted Sentiment: Negative

📬 Contact
For questions, reach out to sharma.deepak011199@gmail.com or connect on LinkedIn - https://www.linkedin.com/in/deepak-sharma-451918171/

