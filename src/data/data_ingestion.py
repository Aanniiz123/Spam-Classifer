import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from src.utils.logger import logging

def load_config(path = 'configs/config.yaml'):
    with open(path , 'r') as f:
        return yaml.safe_load(f)


def load_data():
    config = load_config()
    path = config['data']['raw_path']
    text_col = config['data']['text_column']
    label_col = config['data']['label_column']

    logging.info('All the data is loaded sucussfully')

    df = pd.read_csv(path, encoding='latin1')
    
    
    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info('data is loaded sucuessfully')
    return X_train, X_test, y_train, y_test

