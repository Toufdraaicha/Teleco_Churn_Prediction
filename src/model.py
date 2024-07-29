import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import os
from src.data_processing import load_and_preprocess_data


def train_model(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, '../models/gradient_boosting_model.pkl')
    print("Model trained and saved successfully!")

    return model

