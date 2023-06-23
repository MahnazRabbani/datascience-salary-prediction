# Salary Prediction Based on Job Characteristics and Employee Information

## Objective

The primary goal of this project is to predict an individual's salary in USD using a range of influencing factors. These factors include:

- `work_year`: This represents the years the individual's salary has been recorded.
- `experience_level`: This indicates the individual's level of professional experiences, such as entry, mid, or senior level.
- `employment_type`: This refers to the form of employment the individual is engaged in, for instance, full-time, part-time, or contract.
- `job_title`: This is the designation or title of the job the individual is currently performing.
- `employee_residence`: This refers to the geographical location where the employee is residing.
- `remote_ratio`: This indicates the proportion of work that the individual performs remotely.
- `company_location`: This is the geographical location where the company that employs the individual is based.
- `company_size`: This signifies the size of the employing company in terms of the number of employees or other such measures.

We are going to compare different classic ML algorithms and their performance on this dataset. 

## Project Structure

The project structure is organized as follows:   

ds_salaries.csv: Dataset containing salary information.    
initial_data_analysis.ipynb: Jupyter Notebook for the initial data analysis.      
models_comparision.py: Python script for comparing different models and selectin the best one for hyperparameter tuning.
xgboost_hypertunning.py: Python script for hyperparameter tuning of the XGBoost model.
best_params.joblib: Joblib file storing the best parameters for model training.
xgboost_model.joblib: Joblib file containing the trained XGBoost model.
AnalysisResults.ipynb: Jupyter Notebook containing the analysis results. 


## Data Source

The data used for this project was sourced from the [HuggingFace website](https://huggingface.co/datasets/Einstellung/demo-salaries). 
Please note that the actual salary figures are anonymized and do not correspond to specific individuals.
