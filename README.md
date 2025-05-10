# Sentiment Analysis Using Machine Learning

This repository contains a sentiment analysis project using machine learning models, including **Word2Vec + Random Forest**, **TF-IDF + Logistic Regression**, and other models. The goal is to classify app reviews into three categories: **negative**, **neutral**, and **positive**.

## Table of Contents

1. [Installation](#installation)
2. [Dependencies](#dependencies)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Inference](#inference)
7. [Evaluation](#evaluation)
8. [License](#license)

## Installation

To use this project, follow the steps below:

### 1. Clone the repository

Clone the repository to your local machine.

```bash
git clone https://github.com/username/sentiment-analysis.git
cd sentiment-analysis

2. Create a virtual environment (optional but recommended)
It is recommended to use a virtual environment to avoid conflicts with other projects' dependencies.

bash
Copy
Edit
python -m venv venv
Activate the virtual environment:

On Windows:

bash
Copy
Edit
venv\Scripts\activate
On macOS/Linux:

bash
Copy
Edit
source venv/bin/activate
3. Install dependencies
Once the virtual environment is activated, install the required dependencies by running:

bash
Copy
Edit
pip install -r requirements.txt
4. Download Word2Vec pre-trained model
You can download the pre-trained Google News Word2Vec model from this link.

Once downloaded, extract the .bin.gz file to the data/ directory.

bash
Copy
Edit
# Assuming you download and extract it to:
# C:/Users/YourUsername/Downloads/GoogleNews-vectors-negative300.bin.gz
Dependencies
This project uses the following libraries:

pandas

numpy

nltk

scikit-learn

gensim (optional for Word2Vec)

matplotlib

tqdm

Setup
Download the dataset
The dataset is uber_reviews_labeled.csv, which contains labeled app reviews with columns like reviewId, content, score, and sentiment.

The dataset should be placed in the data/ directory for easy access.

Load and Preprocess Data
Data preprocessing includes:

Lowercasing text

Removing unnecessary columns

Tokenization (optional)

Vectorization using TF-IDF or Word2Vec

Usage
1. Training the Model
You can run the following Python script to train the model using different algorithms.

Train using TF-IDF + Logistic Regression:
bash
Copy
Edit
python train_tf_idf_logreg.py
Train using Word2Vec + Random Forest:
bash
Copy
Edit
python train_word2vec_rf.py
2. Inference or Testing the Model
To perform inference (testing) on new data, use the following script. This will classify new reviews as negative, neutral, or positive.

bash
Copy
Edit
python inference.py
This will output the sentiment class of the input review.

Example of Inference Script
python
Copy
Edit
# Example of running inference on a single review
from sentiment_model import load_model, preprocess

model = load_model('logreg')  # load the trained logistic regression model

review = "I love this app, it's great!"
preprocessed_review = preprocess(review)
prediction = model.predict([preprocessed_review])

print(f"Sentiment: {prediction[0]}")
3. Evaluate the Model
After training, the model will output various performance metrics such as accuracy, precision, recall, and F1-score.

bash
Copy
Edit
python evaluate_model.py
Model Training
Training models are done in separate scripts:

TF-IDF + Logistic Regression:

The model vectorizes text using TF-IDF and trains a Logistic Regression classifier.

Results will show the accuracy and classification report.

Word2Vec + Random Forest:

Word2Vec model is used for text vectorization, and a Random Forest classifier is trained.

Results will show the accuracy and classification report.

Inference
After training, you can perform inference using the trained models. For inference, provide a text review, and the model will predict if the sentiment is positive, neutral, or negative.

Example:

python
Copy
Edit
from sentiment_model import load_model, preprocess

# Load model
model = load_model('logreg')  # or 'rf' for Random Forest model

# Review to be predicted
review = "This app is fantastic!"

# Preprocess the text (lowercase, tokenize, etc.)
preprocessed_review = preprocess(review)

# Predict sentiment
prediction = model.predict([preprocessed_review])

print(f"Predicted Sentiment: {prediction[0]}")
Evaluation
After performing inference, the model's performance can be evaluated using:

Accuracy: The proportion of correctly predicted labels.

Precision, Recall, and F1-Score: Useful for evaluating the performance of classification models, especially when dealing with imbalanced datasets.

Example output after running the evaluation script:

bash
Copy
Edit
Accuracy: 0.909
Classification Report:
               precision    recall  f1-score   support
    negative       0.74      0.81      0.77       313
     neutral       0.82      0.85      0.84       274
    positive       0.97      0.94      0.95      1413
License
This project is licensed under the MIT License - see the LICENSE file for details.

markdown
Copy
Edit

### Penjelasan
- **Instalasi**: Panduan lengkap untuk menginstal dan mengonfigurasi lingkungan virtual, serta mengunduh dataset dan model pre-trained.
- **Dependencies**: Daftar semua library yang dibutuhkan.
- **Setup**: Instruksi tentang bagaimana menyiapkan file dan direktori yang diperlukan.
- **Usage**: Cara menjalankan skrip untuk melatih dan menguji model.
- **Evaluation**: Penjelasan tentang cara mengevaluasi hasil model.

Semoga file `README.md` ini membantu dalam menjalankan dan memelihara proyekmu!
