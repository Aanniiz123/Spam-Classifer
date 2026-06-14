from src.data.data_ingestion import load_data
from src.features.feature_engineering import build_features
from src.models.train import train_model
from src.models.predict import predict

if __name__ == '__main__':
    # Step 1 - Load data
    X_train, X_test, y_train, y_test = load_data()

    # Step 2 - Feature engineering
    X_train_t, X_test_t, y_train_e, y_test_e = build_features(
        X_train, X_test, y_train, y_test
    )

    # Step 3 - Train model
    train_model(X_train_t, X_test_t, y_train_e, y_test_e)

    # Step 4 - Test prediction
    test_input = "I just reached home. Let me know when you're free." 
    result = predict(test_input)
    print(f"\n🔍 Prediction: {result}")