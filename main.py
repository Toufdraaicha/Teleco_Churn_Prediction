import sys
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from src.data_processing import load_and_preprocess_data
from src.evaluation import evaluate_model
from src.model import train_model

sys.path.append('./src')


def should_contact_client(churn_probability, revenue, cost_to_contact=10):
    churn_label = 'LEAVE' if churn_probability > 0.5 else 'STAY'

    if churn_label == 'LEAVE':
        expected_loss = revenue * churn_probability

        if expected_loss > 10:
            # Calcul de la réduction
            max_discount = min(expected_loss - cost_to_contact, revenue * 0.5)
            return 'YES', max_discount

    return 'NO', 0


# Supposons que `test_ids` et `predictions` soient des listes, et que `test_data` contienne les revenus.
discounts = []
contact_decisions = []


def rank_customers(predictions, X_test, test_ids):
    discounts = []
    contact_decisions = []

    for i in range(len(test_ids)):
        churn_probability = predictions[i]
        revenue = X_test.at[i, 'REVENUE']

        contact_decision, discount = should_contact_client(churn_probability, revenue)

        contact_decisions.append(contact_decision)
        discounts.append(discount)
    output_df = pd.DataFrame({
        'CUSTOMER_ID': test_ids,
        'CHURN_PROBABILITY': predictions,
        'CHURN_LABEL': ['LEAVE' if prob > 0.5 else 'STAY' for prob in predictions],
        'CLIENT_TO_CONTACT': contact_decisions,
        'DISCOUNT': discounts
    })

    results = output_df.sort_values(by='CHURN_PROBABILITY', ascending=False)
    return results


def main():
    # Load and preprocess the data
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, test_ids, test_df = load_and_preprocess_data('./data/training.csv',
                                                                           './data/validation.csv')
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
    predictions = model.predict_proba(X_test)[:, 1]
    ranked_customers = rank_customers(predictions, test_df, test_ids)
    print("Customer ranking completed!")

    # Save the ranked customers to a CSV file
    ranked_customers.to_csv('./data/ranked_customers.csv', index=False)
    print("Ranked customers saved to './data/ranked_customers.csv'")


if __name__ == "__main__":
    main()
