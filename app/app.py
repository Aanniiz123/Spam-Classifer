from flask import Flask, render_template, request
from src.models.predict import predict as spam_predict
from src.utils.logger import logging

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text', '')

        if not text.strip():
            return render_template(
                'index.html',
                error='Please enter some text'
            )

        result = spam_predict(text)

        logging.info(f"Prediction made: {result}")

        return render_template(
            'index.html',
            prediction=result,
            text=text
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")

        return render_template(
            'index.html',
            error=str(e)
        )


if __name__ == '__main__':
    app.run(debug=True)