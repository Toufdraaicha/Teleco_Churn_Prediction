import nbformat as nbf

# Crée un nouveau notebook
nb = nbf.v4.new_notebook()

# Ajoute des cellules avec du code
cells = []

# Cellule 1: Importer les bibliothèques
cells.append(nbf.v4.new_code_cell("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
"""))

# Cellule 2: Charger et Prétraiter les Données
cells.append(nbf.v4.new_code_cell("""
def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    categorical_cols = ['COLLEGE', 'JOB_CLASS', 'REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL', 'CONSIDERING_CHANGE_OF_PLAN']
    numerical_cols = ['DATA', 'INCOME', 'OVERCHARGE', 'LEFTOVER', 'HOUSE', 'LESSTHAN600k', 'CHILD', 'REVENUE', 'HANDSET_PRICE', 'OVER_15MINS_CALLS_PER_MONTH', 'TIME_CLIENT', 'AVERAGE_CALL_DURATION']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    X = train_df.drop(columns=['CUSTOMER_ID', 'CHURNED'])
    y = train_df['CHURNED']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(test_df.drop(columns=['CUSTOMER_ID']))

    test_ids = test_df['CUSTOMER_ID']
    return X_train, X_test, y_train, test_ids

train_path = 'data/train.csv'
test_path = 'data/test.csv'
X_train, X_test, y_train, test_ids = load_and_preprocess_data(train_path, test_path)
"""))

# Cellule 3: Entraîner le Modèle
cells.append(nbf.v4.new_code_cell("""
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)
"""))

# Cellule 4: Faire des Prédictions
cells.append(nbf.v4.new_code_cell("""
def predict(model, X_test):
    return model.predict_proba(X_test)[:, 1]

predictions = predict(model, X_test)
"""))

# Cellule 5: Générer les Résultats
cells.append(nbf.v4.new_code_cell("""
output_df = pd.DataFrame({
    'CUSTOMER_ID': test_ids,
    'CHURN_PROBABILITY': predictions,
    'CHURN_LABEL': ['LEAVE' if prob > 0.5 else 'STAY' for prob in predictions],
    'CLIENT_TO_CONTACT': ['YES' if prob > 0.5 else 'NO' for prob in predictions],
    'DISCOUNT': [min(10 / prob, 50) if prob > 0.5 else 0 for prob in predictions]
})

output_path = 'data/output.csv'
output_df.to_csv(output_path, index=False)
print(f'Résultats sauvegardés dans {output_path}')
"""))

# Ajouter les cellules au notebook
nb['cells'] = cells

# Sauvegarder le notebook
with open('TELCO_Churn_Prediction.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook généré: TELCO_Churn_Prediction.ipynb")
