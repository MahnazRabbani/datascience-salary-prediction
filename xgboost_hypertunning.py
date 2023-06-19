import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump, load

from model_selection import load_data, encode_experience, drop_columns, convert_columns_to_str, preprocess_data

def preprocess_and_split_data(df):
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
    X_preprocessed = preprocess_data(X, categorical_features)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def hyperparameter_tuning(X_train, y_train):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'learning_rate': [0.08, 0.1, 0.12],
    'max_depth': [4, 5, 6],
    'n_estimators': [90, 100, 110],
    'reg_alpha': [0.03, 0.05, 0.07, 0],
    'reg_lambda': [0.03, 0.05, 0.07]
    }

    # Perform hyperparameter tuning using GridSearchCV
    xgb = XGBRegressor()
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Save the best parameters to a file
    params_filename = 'best_params.joblib'
    dump(best_params, params_filename)

    return best_params

def train_with_best_params(X_train, X_test, y_train, y_test, best_params):
    # Train the model with the best hyperparameters
    xgb = XGBRegressor(**best_params)
    xgb.fit(X_train, y_train)

    # Save the trained model
    model_filename = 'xgboost_model.joblib'
    dump(xgb, model_filename)

    # Make predictions on the test set
    y_pred = xgb.predict(X_test)

    # Calculate accuracy
    mse = mean_squared_error(y_test, y_pred)

    return mse

def main():
    # Load the dataset
    df = load_data('ds_salaries.csv')

    # Preprocess and split the data
    X_train, X_test, y_train, y_test = preprocess_and_split_data(df)

    # Hyperparameter tuning
    best_params = hyperparameter_tuning(X_train, y_train)

    # Print the best hyperparameters
    print("Best Hyperparameters:")
    print(best_params)

    # Train the model with the best parameters
    mse_best_params = train_with_best_params(X_train, X_test, y_train, y_test, best_params)

    # Output the accuracy of the model
    print(f"\nMean Squared Error with Best Parameters: {mse_best_params}")

if __name__ == "__main__":
    main()
