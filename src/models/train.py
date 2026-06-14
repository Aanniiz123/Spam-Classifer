import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

from src.utils.logger import logging
from src.data.data_ingestion import load_config




def train_model(X_train, X_test, y_train, y_test):
    config = load_config()
    alpha = config['model']['alpha']

    logging.info('Training the bernoulli navi bayes')

    model = BernoulliNB(alpha= alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    logging.info(f'classifaciton : {classification_report(y_test, y_pred)}')

    with open('model/bernoulli.pkl','wb') as file:
        pickle.dump(model, file)

    logging.info('model saved')

    return model

