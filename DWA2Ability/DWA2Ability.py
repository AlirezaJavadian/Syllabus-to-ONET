""" Script for estimating tha abilities scores from the syllabi DWA scores using the trained predictive models for ONET abilities based on their DWA scores (see DWA2Ability_model_training.py) """

import os
import pandas as pd
import joblib

inputs_path = './DWA2Ability_inputs'
ability_models_path = './ability_models'
abilities_list_file = os.path.join(inputs_path, 'abilities_name.txt')

# Please specify the desired path to save the abilities scores
results_f_path = None


datasets_path = '../datasets'

dwa_scores_f_path = os.path.join(datasets_path, 'detailed_work_activities_scores.gzip')

# In addition to the abilities columns, the columns that you want to keep in your results dataframe.
cols_to_keep = ['id']


def dwa_to_abilities_mapper(abilities_list, dwa_df, cols_to_keep, ability_models_path, results_f_path):


    dwas_col_names = list(dwa_df.columns)
    dwas_col_names.remove('id')

    dwa_scores_df = dwa_df[dwas_col_names].values
    abilities_df = dwa_df[cols_to_keep]
    
    models_path = os.path.join(ability_models_path, 'models')

    for ability_name in abilities_list:
        model = joblib.load(os.path.join(models_path, '{}.joblib'.format(ability_name)))
        predictions = model.predict(dwa_scores_df)
        abilities_df[ability_name] = predictions
    if results_f_path != None:
        abilities_df.to_csv(results_f_path, compression='gzip', index=False)
    
    return abilities_df


with open(abilities_list_file, 'r') as file:
    abilities_list = file.readlines()
    abilities_list = [line.rstrip() for line in abilities_list]
    

dwa_df = pd.read_csv(dwa_scores_f_path, compression='gzip')



abilities_df = dwa_to_abilities_mapper(abilities_list, dwa_df, cols_to_keep, ability_models_path, results_f_path)
