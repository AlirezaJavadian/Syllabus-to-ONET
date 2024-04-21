""" Script for training predictive models for ONET abilities based on their DWA scores """

import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib



inputs_path = './DWA2Ability_inputs'
ability_models_path = './ability_models'
abilities_list_file = os.path.join(inputs_path, 'abilities_name.txt')
performance_plots_path = os.path.join(ability_models_path, 'performance_plots')
SOC_abilitiesScores_and_relevantDWAs_file = os.path.join(inputs_path, 'SOC_abilitiesScores_and_relevantDWAs.csv')



inputs_path = './DWA2Ability_inputs'
ability_models_path = './ability_models'
abilities_list_file = os.path.join(inputs_path, 'abilities_name.txt')
performance_plots_path = os.path.join(ability_models_path, 'performance_plots')
SOC_abilitiesScores_and_relevantDWAs_file = os.path.join(inputs_path, 'SOC_abilitiesScores_and_relevantDWAs.csv')




with open(abilities_list_file, 'r') as file:
    abilities_list = file.readlines()
    abilities_list = [line.rstrip() for line in abilities_list]

SOC_abilitiesScores_and_relevantDWAs_df = pd.read_csv(SOC_abilitiesScores_and_relevantDWAs_file, index_col=0)


scale = True

dwas_col_names = [i for i in list(SOC_abilitiesScores_and_relevantDWAs_df.columns) if i not in abilities_list]
X = SOC_abilitiesScores_and_relevantDWAs_df[dwas_col_names].values

for target_ability in abilities_list:

    target_ability = abilities_list[0]

    if scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        y = min_max_scaler.fit_transform(SOC_abilitiesScores_and_relevantDWAs_df[[target_ability]]).ravel()
    else:
        y = SOC_abilitiesScores_and_relevantDWAs_df[[target_ability]].values.ravel()

    filename = '{}.joblib'.format(target_ability)
    model_save_path = os.path.join(ability_models_path, 'models', filename)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating a Random Forest Regressor model
    model = RandomForestRegressor(random_state=42)

    # Creating a pipeline
    pipeline = Pipeline([
        ('regressor', model)
    ])


    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 4, 6],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['sqrt', 'log2', None]
    }


    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

    # Fitting the model
    grid_search.fit(X_train, y_train)

    # Best parameters
    print("Best parameters found: ", grid_search.best_params_)

    # Predicting with the best estimator
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    best_pipeline = grid_search.best_estimator_

    # Perform cross-validation
    cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    # The mean of the cross-validated scores
    mean_cv_score = np.mean(cv_scores)

    # best_params = grid_search.best_params_
    best_params = {k.replace('regressor__', ''): v for k, v in grid_search.best_params_.items()}

    model = RandomForestRegressor(**best_params, random_state=42)

    # Train the model
    model.fit(X, y)

    # Save the model
    joblib.dump(model, model_save_path)


    model = joblib.load(model_save_path)

    predictions = model.predict(X)


    performance_str = "MSE (Test): {}, MSE (CV): {}".format(round(mse, 3), round(mean_cv_score, 3))

    temp_dict = grid_search.best_params_
    params_string = ''
    counter = 1
    for key in temp_dict.keys():
        this_str = key.split('regressor__')[1] + ': '
        this_value = temp_dict[key]
        if this_value == None:
            this_value = 'None'
        this_str = this_str + str(this_value) + ', '
        params_string = params_string + this_str
        if counter%2 == 0:
            params_string = params_string + '\n'
        counter +=1 
    params_string = params_string[0:-2]


    plt.figure(figsize=(4,4.5))
    plt.scatter(y, predictions, alpha=0.4)
    plt.xlabel('Real')
    plt.ylabel('Predicted')
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)


    plt.title('{}\n{}\n{}'.format(' '.join(target_ability.split('_')), performance_str, params_string))
    plt.tight_layout()
    plt.savefig(os.path.join(performance_plots_path, target_ability + '.png'))
    print("Finished: ", target_ability)
    