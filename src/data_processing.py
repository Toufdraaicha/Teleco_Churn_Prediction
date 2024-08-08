import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
categorical_cols = ['COLLEGE', 'JOB_CLASS', 'REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL', 'CONSIDERING_CHANGE_OF_PLAN']
numerical_cols = ['DATA', 'INCOME', 'OVERCHARGE', 'LEFTOVER', 'HOUSE', 'CHILD', 'REVENUE', 'HANDSET_PRICE', 'OVER_15MINS_CALLS_PER_MONTH', 'TIME_CLIENT', 'AVERAGE_CALL_DURATION']

def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Fill missing values
    train_df['HOUSE'].fillna(train_df['HOUSE'].median(), inplace=True)
    train_df['LESSTHAN600k'].fillna(train_df['LESSTHAN600k'].mode()[0], inplace=True)

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

    # Split features and target
    X = train_df.drop(columns=['CUSTOMER_ID', 'CHURNED'])
    y = train_df['CHURNED']

    # Apply preprocessing
    X_train = preprocessor.fit_transform(X)
    X_test = preprocessor.transform(test_df.drop(columns=['CUSTOMER_ID']))

    return X_train, y, X_test, test_df['CUSTOMER_ID'],test_df

if __name__ == "__main__":
    X_train, y_train, X_test, test_ids ,test_df= load_and_preprocess_data('../data/training.csv', '../data/validation.csv')
    print("Data loaded and preprocessed successfully!")
