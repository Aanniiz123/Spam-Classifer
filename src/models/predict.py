import pickle
from src.data.data_ingestion import load_config
from src.features.feature_engineering import preprocess_text
from src.utils.logger import logging


def load_artificts():
    
    with open('model/tfidf.pkl', 'rb') as file:
        tfidf = pickle.load(file)

    with open('model/label.pkl', 'rb') as file:
        le = pickle.load(file)

    with open('model/bernoulli.pkl', 'rb') as file:
        model = pickle.load(file)

    return tfidf, le, model


def predict(text: str) -> str:
    tfidf, le, model = load_artificts()

    clean = preprocess_text(text)
    vector = tfidf.transform([clean])
    result = model.predict(vector)
    label = le.inverse_transform(result)[0]
    logging.info(f"Input: {text[:50]}... | Prediction: {label}")

    return label

