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
    order = ['EN', 'MI', 'SE', 'EX']
    mapping = {category: i+1 for i, category in enumerate(order)} #{'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}

    df['experience_level'] = df['experience_level'].map(mapping)
    return df

def drop_columns(df):
    """
    Drop specified columns from a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.
        columns (list): List of column names to drop.

    Returns:
        pandas.DataFrame: The modified DataFrame with specified columns dropped.
    """
    columns = ['salary', 'salary_currency']
    df_copy = df.drop(columns=columns)  # Drop specified columns from the DataFrame
    return df_copy

#work_year             object
#experience_level       int64
#employment_type       object
#job_title             object
#salary                 int64
#salary_currency       object
#salary_in_usd          int64
#employee_residence    object
#remote_ratio          object
#company_location      object
#company_size          object

def convert_columns_to_str(df):
    """
    Convert 'remote_ratio' and 'work_year' columns in a DataFrame to string data type.

    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.

    Returns:
        None. The DataFrame is modified in place.
    """
    df['remote_ratio'] = df['remote_ratio'].astype(str)  # Convert 'remote_ratio' to string data type
    df['work_year'] = df['work_year'].astype(str)  # Convert 'work_year' to string data type
    return df

def preprocess_data(df, categorical_features):
    """
    Preprocesses the dataset by filling missing values and one-hot encoding categorical features.

    Parameters:
        df (DataFrame): The dataset to preprocess.
        categorical_features (list): The list of categorical features.

    Returns:
        ndarray: The preprocessed dataset.
    """

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine transformers in the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)])

    # Apply transformations
    return preprocessor.fit_transform(df)


def train():
    # Load the dataset
    df = load_data('ds_salaries.csv')

    # Perform feature engineering
    df = encode_experience(df)
    df = convert_columns_to_str(df)
    df = drop_columns(df)

    # Define the features
    numerical_features = ['experience_level']
    categorical_features = ['work_year', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio','company_location', 'company_size']
    # Split the target variable from features
    X = df[numerical_features + categorical_features]
    y = df['salary_in_usd']

    # Preprocess the dataset
    X_preprocessed = preprocess_data(X, categorical_features) #shape:(3755, 257)

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
    trained_models = []
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error for {name}: {mse}')
        trained_models.append((name, mse))

    return trained_models

def main():
    # Call the train function
    models = train()

    # Sort models from best to worst based on MSE
    sorted_models = sorted(models, key=lambda x: x[1])

    # Output the trained models from best to worst
    print("Trained models from best to worst:")
    for model in sorted_models:
        print(f"{model[0]} - MSE: {model[1]}")

if __name__ == "__main__":
    main()