"""
This project uses user access to the application as an example, and can be modified to apply to
web-side data analysis. When users access an application, they do so through an API. Specific API
sequences result in specific business logic being available to users. Depending on the business
logic being accessed, there may be multiple sequences of API calls that, when aggregated, become
the API call graph for that user. When there are hundreds of users, many users will generate
identical or similar API call graphs. In this case, users are grouped into a single cluster and all
users have the same graph according to the graph clustering algorithm. The clusters were analyzed
manually and each plot was manually classified as normal or outlier, which can be found in the
classification column. Each row in supervised_dataset.csv is a plot like this. In addition to class
labels, some metrics of the graph are also available, which can be used as features.
"""

import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, MetricVisualizer, Pool
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20, 10)


# Load the training and evaluation datasets
training_df = pd.read_csv('/kaggle/input/api-access-behaviour-anomaly-dataset/supervised_dataset.csv', index_col=False).drop('Unnamed: 0', axis=1)
evaluation_df = pd.read_csv('/kaggle/input/api-access-behaviour-anomaly-dataset/remaining_behavior_ext.csv', index_col=False).drop('Unnamed: 0', axis=1)

training_df.columns, evaluation_df.columns, len(training_df)

# Using catboost to model the 2 class classifier
train, test = train_test_split(training_df, test_size=0.4, random_state=1073)

param_grid = {
    "iterations": [10, 100, 1000], # number of trees
    "learning_rate": [0.1, 0.01],
    "custom_loss": ['CrossEntropy', 'AUC', 'Logloss', ]
}
gclf = GridSearchCV(estimator=CatBoostClassifier(), param_grid=param_grid, cv=4)

gclf.fit(train.drop(['classification', 'source', '_id'], axis=1), train['classification'], cat_features=['ip_type'], verbose=1000)

gclf.best_params_

# Evaluation
evaluation_cols_removed_df= evaluation_df.drop(['behavior_type', 'behavior', 'source', '_id'], axis=1)
test_cols_removed_df= test.drop(['source', '_id', 'classification'], axis=1)

prediction=gclf.predict(evaluation_cols_removed_df)
after_prediction_pd = evaluation_df[['behavior_type', 'behavior', 'source']].copy()
after_prediction_pd['prediction']=prediction
eval_cols_evaluation_set = after_prediction_pd[['behavior_type', 'prediction']]

prediction=gclf.predict(test_cols_removed_df)
test_after_prediction_pd = test[['classification']].copy()
test_after_prediction_pd['prediction']=prediction
test_cols_evaluation_set = test_after_prediction_pd[['classification', 'prediction']]


# Weighted misclassification error
def calculate_misclassification(eval_cols, predicted_col, orig_col):
    classes = dict(eval_cols[orig_col].value_counts())

    misclassification_error = 0
    for class_name in classes:
        selection = eval_cols[ (eval_cols[orig_col] == class_name)]
        weight = 1.0/classes[class_name]
        misclassifiction_cases = selection[(selection[predicted_col] != class_name)]
        misclassification_count = len(misclassifiction_cases)
        print({'class': class_name, 'weight': weight, 'count': len(selection), 'misclassification':misclassification_count})

        misclassification_error += weight*misclassification_count
    misclassification_error = misclassification_error / len(classes)
    return misclassification_error

#
# print('Classification of test data')
# print("Misclassification error: ",calculate_misclassification(test_cols_evaluation_set, 'prediction', 'classification'))
#
# print("Classsification of evaluation_df")
# print("Misclassification error: ",calculate_misclassification(eval_cols_evaluation_set, 'prediction', 'behavior_type'))




































