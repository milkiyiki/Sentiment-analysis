import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# Unduh resource VADER (hanya sekali)
nltk.download('vader_lexicon')

# Load dataset hasil scraping
df = pd.read_csv("uber_reviews_en.csv")

# Buat instance VADER
sid = SentimentIntensityAnalyzer()

def clean_text(text):
    # Hapus URL, karakter non-alfabet, dsb.
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower().strip()
    return text

def get_sentiment_label(text):
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Terapkan pembersihan teks
df['clean_text'] = df['content'].astype(str).apply(clean_text)

# Terapkan pelabelan VADER
df['sentiment'] = df['clean_text'].apply(get_sentiment_label)

# Tampilkan distribusi label
print(df['sentiment'].value_counts())

# Simpan hasil
df.to_csv("uber_reviews_labeled.csv", index=False)
print("Preprocessing dan labeling selesai. Data disimpan ke 'uber_reviews_labeled.csv'")
