import sys
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from src.data_processing import load_and_preprocess_data
from src.evaluation import evaluate_model
from src.model import train_model

sys.path.append('./src')


def rank_customers(model, X_test, test_ids):
    predictions = model.predict_proba(X_test)[:, 1]
    output_df = pd.DataFrame({
    'CUSTOMER_ID': test_ids,
    'CHURN_PROBABILITY': predictions,
    'CHURN_LABEL': ['LEAVE' if prob > 0.5 else 'STAY' for prob in predictions],
    'CLIENT_TO_CONTACT': ['YES' if prob > 0.5 else 'NO' for prob in predictions],
    'DISCOUNT': [min(10 / prob, 50) if prob > 0.5 else 0 for prob in predictions]
})
    results = output_df.sort_values(by='CHURN_PROBABILITY', ascending=False)
    return results

def main():
    # Load and preprocess the data
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, test_ids = load_and_preprocess_data('./data/training.csv', './data/validation.csv')
    print("Data loaded and preprocessed successfully!")

    # Train the model
    print("Training the model...")
    model = train_model(X_train, y_train)
    print("Model trained successfully!")

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_train, y_train)
    print("Model evaluation completed!")

    # Rank customers by churn probability
    print("Ranking customers by churn probability...")
    ranked_customers = rank_customers(model, X_test, test_ids)
    print("Customer ranking completed!")

    # Save the ranked customers to a CSV file
    ranked_customers.to_csv('./data/ranked_customers.csv', index=False)
    print("Ranked customers saved to './data/ranked_customers.csv'")

if __name__ == "__main__":
    main()
