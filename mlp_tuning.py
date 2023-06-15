import numpy as np
from sklearn.model_selection import GridSearchCV
from mlp_train import get_preprocessed_data, get_mlp_model

# Get the preprocessed data
X_train, X_test, y_train, y_test = get_preprocessed_data()

# Get the MLP model
mlp = get_mlp_model()

# Define the parameter grid
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# Set up GridSearchCV
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)

# Fit the model and find optimal hyperparameters
clf.fit(X_train, y_train)

# Best parameter set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# Predict on test set with optimized parameters
y_pred = clf.predict(X_test)
