import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def load_data(path):
    """
    Load the dataset from a specified path.
    
    Parameters:
    path (str): The path to the dataset.

    Returns:
    DataFrame: The loaded dataset.
    """
    return pd.read_csv(path)

def encode_experience(df):
    """
    Encodes experience level with ordinal values.
    
    Parameters:
    df (DataFrame): The dataset to encode.

    Returns:
    DataFrame: The encoded dataset.
    """
    ordinal_mapping = {'MI': 1, 'SE': 2}  # modify this based on your knowledge of the levels
    df['experience_level'] = df['experience_level'].map(ordinal_mapping)
    return df

def preprocess_data(df, numerical_features, categorical_features):
    """
    Preprocesses the dataset by filling missing values, standardizing numerical features,
    and one-hot encoding categorical features.
    
    Parameters:
    df (DataFrame): The dataset to preprocess.
    numerical_features (list): The list of numerical features.
    categorical_features (list): The list of categorical features.

    Returns:
    ndarray: The preprocessed dataset.
    """
    # Define the transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine transformers in the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Apply transformations
    return preprocessor.fit_transform(df)


def main():
    # Load the dataset
    df = load_data('ds_salaries.csv')

    # Perform feature engineering
    df = encode_experience(df)

    # Define the features
    numerical_features = ['work_year', 'remote_ratio', 'experience_level']
    categorical_features = ['employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']

    # Split the target variable from features
    X = df[numerical_features + categorical_features]
    y = df['salary_in_usd']

    # Preprocess the dataset
    X_preprocessed = preprocess_data(X, numerical_features, categorical_features)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    # Define the models
    models = [
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor()),
        ('SVM', SVR()),
        ('XGBoost', XGBRegressor()),
        ('LightGBM', LGBMRegressor())
    ]

    # Train and evaluate each model
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error for {name}: {mse}')

if __name__ == "__main__":
    main()
