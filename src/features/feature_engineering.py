import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import logging
from src.data.data_ingestion import load_config, load_data
import string
import nltk
import os


nltk.download('stopwords', quiet= True)


stem = PorterStemmer()
stop_words = set(stopwords.words('english'))



def preprocess_text(text: str) -> str:
    # 1. Lowercase
    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', 'URL_TOKEN', text) # Mark URLs instead of deleting
    text = re.sub(r'[^\w\s]', 'SYM_TOKEN', text) # Mark punctuation/symbols instead of deleting
    text = re.sub(r'\d+', 'NUM_TOKEN', text)
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 3. Remove special quotes and HTML tags
    text = re.sub(r"[‘’“”«»]", "", text)
    text = re.sub(r'<.*?>', '', text)
    # 4. Remove digits
    text = re.sub(r'\d+', '', text)
    # 5. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 6. Tokenize
    words = word_tokenize(text)
    # 7. Remove stopwords & Stemming
    words = [stem.stem(word) for word in words if word not in stop_words]
    
    return " ".join(words)


def build_features(X_train, X_test, y_train, y_test):
    config = load_config()
    max_features = config['features']['max_features']

    logging.info('Preprocessing text...')

    X_train_clean = X_train.apply(preprocess_text)
    X_test_clean = X_test.apply(preprocess_text)

    logging.info('Fitting the tfidf..')

    tfidf = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = tfidf.fit_transform(X_train_clean)
    X_test_tfidf = tfidf.transform(X_test_clean)

    logging.info('Encoding the labels.....')

    le = LabelEncoder()
    y_train_ecc = le.fit_transform(y_train)
    y_test_ecc = le.transform(y_test)

    os.makedirs('model', exist_ok= True)
    with open('model/tfidf.pkl', 'wb') as file:
        pickle.dump(tfidf, file)

    with open('model/label.pkl', 'wb') as file:
        pickle.dump(le, file)
    
    logging.info('All the tfid and Label encoding is saved')

    return X_train_tfidf, X_test_tfidf, y_train_ecc, y_test_ecc


