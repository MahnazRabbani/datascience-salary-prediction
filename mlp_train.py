import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def load_data(path):
    return pd.read_csv(path)

def encode_experience(df):
    ordinal_mapping = {'MI': 1, 'SE': 2}
    df['experience_level'] = df['experience_level'].map(ordinal_mapping)
    return df

def preprocess_data(df, numerical_features, categorical_features):
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor.fit_transform(df)

def get_preprocessed_data():
    df = load_data('ds_salaries.csv')
    df = encode_experience(df)
    numerical_features = ['work_year', 'remote_ratio', 'experience_level']
    categorical_features = ['employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']

    X = df[numerical_features + categorical_features]
    y = df['salary_in_usd']

    X_preprocessed = preprocess_data(X, numerical_features, categorical_features)
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def get_mlp_model():
    return MLPRegressor(random_state=42)

def main():
    X_train, X_test, y_train, y_test = get_preprocessed_data()

    # Initialize MLPRegressor
    mlp = get_mlp_model()

    # Train the model
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for MLP: {mse}')

if __name__ == "__main__":
    main()
