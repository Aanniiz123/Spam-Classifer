import sys
import os
import nltk
from app.app import app

nltk.download('stopwords')
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == '__main__':
    app.run()
