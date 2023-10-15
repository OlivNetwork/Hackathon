"""
This project uses new shopee users as a case for experimental analysis.

This basic data set is collected from new users in Shopee, whose behavior we want to see. The time
window here is unclear, so I'm assuming we want to predict whether a user will churn or retain on
day 15 (one day after the snapshot date, which is two weeks after first signing up).

Business Background: We want to find the best user segment with the highest probability of retention
and we can expect that segment to get the same retention rate. By doing this, we can target our
campaigns only to this segment of users who are more likely to remain in Shopee.

For example, we were able to find segmented users with an average model probability or score above
70% (>70% chance of retention). We will then focus our activities on this user group with the
expectation that from this user group we may get over 70% retention.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import statistics as st
import warnings
import shap

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,train_test_split
from sklearn.model_selection import ShuffleSplit,KFold,cross_val_score
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

df = pd.read_csv("/kaggle/input/shopee-new-user-behavior/sample_data_DStest.csv")
df.head()
df.info()



"""
Checking The Cleanliness of The Data
"""
# Check Missing Values
df = df.rename(columns={'label':'retained_label'})

cat_features = ['gender','age_group','region','is_rural_or_urban','new_buyer_initiative','is_dp_buyer_14d',
                'is_buyer_14d','activate_shopeepay']

all_features = df.columns.values.tolist()

num_features = [f for f in all_features
                if f not in cat_features and f not in ['user_id','regist_date','retained_label']]

(df[cat_features].isnull().sum()/df.shape[0]).sort_values()

(df[num_features].isnull().sum()/df.shape[0]).sort_values()

# Check Duplicates
print(df.shape)
print(df['user_id'].nunique())

# Check Imbalanced Class
df['retained_label'].value_counts(normalize=True, dropna=False)

sns.set_theme(style='whitegrid')
sns.set(rc = {'figure.figsize':(13,5)})
df['retained_label'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()

# Check Outliers
df[num_features].describe()

def detecting_outliers(num_feat):
    df_tmp = df.sort_values(num_feat)

    Q1 = df_tmp[num_feat].quantile(0.25)
    Q3 = df_tmp[num_feat].quantile(0.75)
    IQR = Q3 - Q1

    upper_bound = df[df[num_feat] > (Q3 + (1.5 * IQR))]
    lower_bound = df[df[num_feat] < (Q1 - (1.5 * IQR))]

    print(num_feat)
    print('Lower Outlier: ', lower_bound.shape[0])
    print('Upper Outlier: ', upper_bound.shape[0])
    print('-' * 40, '\n')


for num_feat in num_features:
    detecting_outliers(num_feat)




"""
Exploratory Analysis
"""
# Univariate Analysis
df['regist_date'] = pd.to_datetime(df['regist_date'])

print(df['regist_date'].min())
print(df['regist_date'].max())

for cat in cat_features:
    print(cat)
    display(df[cat].value_counts(1).sort_values())
    df[cat].value_counts(dropna=False).plot(kind="pie", autopct="%.2f")
    plt.show()
    print('-'*60,'\n')

# Bivariate Analysis
sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='age_group',hue='retained_label',data=df,kind="count",height=8, aspect=1.5)
plt.title('Distribution of Age Group Related To Number of Retention Users', fontsize=20)

sns.catplot(x='is_rural_or_urban',hue='retained_label',data=df,kind="count",height=5, aspect=1)
plt.title('Distribution of Rural or Urban Users Related To Number of Retention Users', fontsize=15)

sns.catplot(x='region',hue='retained_label',data=df,kind="count",height=5, aspect=1.5)
plt.title('Distribution of User Region Related To Number of Retention Users', fontsize=15)

sns.catplot(x='is_buyer_14d',hue='retained_label',data=df,kind="count",height=5, aspect=1)
plt.title('Distribution of Buyer Users In the Last 14 Days Related To Number of Retention Users', fontsize=15)

sns.catplot(x='activate_shopeepay',hue='retained_label',data=df,kind="count",height=5, aspect=1)
plt.title('Distribution of Users Who Activate Shopeepay Related To Number of Retention Users', fontsize=15)

# Numerical Features and User Retention
df_retained = df[df['retained_label']==1]
df_not_retained = df[df['retained_label']==0]
feats = ['time_spent_platform_14d','avg_time_per_session_14d','top_up_14d','total_order_14d']

for feat in feats:
    fig, ax = plt.subplots(1, 1,figsize=(10, 5))
    print(feat)
    df_retained[feat].plot.hist(density=True, label="Retained Users", ax=ax, bins=30)
    df_not_retained[feat].plot.hist(density=True, alpha=0.5, label="Non Retained Users", ax=ax,bins=30)
    plt.legend()
    plt.show()
    print('-'*100)

# Multivariate Analysis
df.groupby(['activate_shopeepay','is_buyer_14d']).agg({'retained_label':['mean',len]})

tmp = df.copy()


def total_order_bin(row):
    x = row['total_order_14d']
    if 0 <= x < 3:
        return 1
    elif 3 <= x < 5:
        return 2
    elif 5 <= x < 7:
        return 3
    elif 7 <= x < 10:
        return 4
    elif x >= 10:
        return 5
    else:
        return np.nan


tmp['total_order_14d_bin'] = tmp.apply(total_order_bin, axis=1)

tmp.groupby(['total_order_14d_bin']).agg({'retained_label': ['mean', len]})


def total_login_days_bin(row):
    x = row['total_login_days_l14d']
    if 0 <= x < 3:
        return 1
    elif 3 <= x < 6:
        return 2
    elif 6 <= x < 9:
        return 3
    elif 9 <= x < 12:
        return 4
    elif x >= 12:
        return 5
    else:
        return np.nan


tmp['total_login_days_l14d_bin'] = tmp.apply(total_login_days_bin, axis=1)

tmp.groupby(['total_login_days_l14d_bin']).agg({'retained_label': ['mean', len]})

# combining total order and total login days in the last 14 days
tmp.groupby(['total_login_days_l14d_bin','total_order_14d_bin']).agg({'retained_label':['mean',len]})

tmp.groupby(['total_login_days_l14d_bin','total_order_14d_bin']).agg({'retained_label':['mean']}).unstack().plot(
    kind='bar').legend(bbox_to_anchor=(1.0, 1.0),fontsize='small',)


def shop_views_bin(row):
    x = row['shop_views_14d']
    if 0 <= x < 5:
        return 1
    elif 5 <= x < 10:
        return 2
    elif 10 <= x < 15:
        return 3
    elif 15 <= x < 20:
        return 4
    elif x >= 20:
        return 5
    else:
        return np.nan


tmp['shop_views_14d_bin'] = tmp.apply(shop_views_bin, axis=1)

tmp.groupby(['shop_views_14d_bin']).agg({'retained_label': ['mean', len]})

# combining total order and shop views in the last 14 days
tmp.groupby(['shop_views_14d_bin','total_order_14d_bin']).agg({'retained_label':['mean',len]})

tmp.groupby(['shop_views_14d_bin','total_order_14d_bin']).agg({'retained_label':['mean']}).unstack().plot(
    kind='bar').legend(bbox_to_anchor=(1.0, 1.0),fontsize='small',)


def flash_sale_bin(row):
    x = row['shop_flash_sale']
    if x == 0:
        return 1
    elif x == 1:
        return 2
    elif x >= 2:
        return 3
    else:
        return np.nan


tmp['shop_flash_sale_bin'] = tmp.apply(flash_sale_bin, axis=1)

tmp.groupby(['shop_flash_sale_bin']).agg({'retained_label': ['mean', len]})

# combining total order and shop flash sale in the last 14 days
tmp.groupby(['shop_flash_sale_bin','total_order_14d_bin']).agg({'retained_label':['mean',len]})

# Features Correlation
num_features1 = num_features[0:12]
num_features2 = num_features[12:len(num_features)]

# plt.figure(figsize=(16,9))
fig, axes  = plt.subplots(2,1,figsize=(15, 15))
fig.subplots_adjust(hspace=0.7, wspace=0.3)
sns.heatmap(df[num_features1].corr(),annot = True,cmap = 'viridis',ax=axes[0])
sns.heatmap(df[num_features2].corr(),annot = True,cmap = 'viridis', ax=axes[1])
plt.title('Features Correlation Heatmap', fontsize=20)
plt.show()



"""
Data Preprocessing
"""
df_clean = df.copy()
# Handling Outliers
for num_feat in num_features:
    percentile_90 = df_clean[num_feat].quantile(0.9)
    percentile_10 = df_clean[num_feat].quantile(0.1)
    df_clean[num_feat] = np.where(df_clean[num_feat] > percentile_90 , percentile_90, df_clean[num_feat])
    df_clean[num_feat] = np.where(df_clean[num_feat] < percentile_10 , percentile_10, df_clean[num_feat])

df[num_features].describe()

df_clean[num_features].describe()

# Missing Values Handling
df_clean = df_clean.drop('top_up_14d',axis=1)

df_clean['new_buyer_initiative'].fillna('FS0',inplace=True)
df_clean['new_buyer_initiative'].isnull().sum()

num_features2 = [f for f in num_features if f!= 'top_up_14d']
for num_feat in num_features2:
    df_clean[num_feat] = df_clean[num_feat].fillna(df_clean[num_feat].mean())

df_clean.isnull().sum().sum()

# Categorical Feature Label Encoding
for cat_feature in cat_features:
    print(cat_feature)
    print(df_clean[cat_feature].nunique())
    print('-'*10)


def one_hot_encoder(data, feature, keep_first=True):
    one_hot_cols = pd.get_dummies(data[feature], dtype=bool)

    for col in one_hot_cols.columns:
        one_hot_cols.rename({col: f'{feature}_' + col}, axis=1, inplace=True)

    new_data = pd.concat([data, one_hot_cols], axis=1)
    new_data.drop(feature, axis=1, inplace=True)

    if keep_first == False:
        new_data = new_data.iloc[:, 1:]

    return new_data


for col in ['gender', 'age_group', 'region', 'is_rural_or_urban', 'new_buyer_initiative']:
    df_clean = one_hot_encoder(df_clean, col)

df_clean.head()

for col in df_clean.select_dtypes(include='bool').columns:
    df_clean[col] = df_clean[col].astype(int)

df_clean.head()

df_clean = df_clean.rename(columns={'age_group_<19':'age_group_under_19',
                                   'age_group_>35':'age_group_above_35'})

# Removing Unnecessary Feature
df_clean = df_clean.drop(['use_shopeepaylater','user_id','regist_date'],axis=1)
df_clean.head()

# Data Train and Test Split
features_df = df_clean.drop('retained_label',axis = 1)
target_df = df_clean['retained_label']


# X_train, X_test, y_train, y_test = train_test_split(features_df,target_df,test_size = 0.2,stratify=target_df)
feature_list = features_df.columns.values.tolist()
target = 'retained_label'

# Split
df_clean_shuffle = df_clean.reset_index(drop=True).sample(frac=1, random_state=512).reset_index(drop=True)

df_train = df_clean_shuffle.iloc[: int(0.8 * df_clean_shuffle.shape[0])]
df_val = df_clean_shuffle.iloc[int(0.8 * df_clean_shuffle.shape[0]) : int(0.9 * df_clean_shuffle.shape[0])]
df_test = df_clean_shuffle.iloc[int(0.9 * df_clean_shuffle.shape[0]) :]

print(df_train.shape , df_val.shape, df_test.shape )

# Imbalance Data
df_clean['retained_label'].value_counts(1)

Counter(df_train[target])

over = SMOTE(sampling_strategy=0.95)
# under = RandomUnderSampler(sampling_strategy=0.9)

steps = [('over', over)]
pipeline = Pipeline(steps=steps)

# transform the dataset
df_train2, df_train2['retained_label'] = pipeline.fit_resample(df_train[feature_list], df_train[target])
# summarize the new class distribution
counter_new = Counter(df_train2['retained_label'])
print(counter_new)

# Feature Scaling
scaler = StandardScaler()
scaler.fit(df_train[feature_list],df_train[target])

df_train[feature_list] = scaler.transform(df_train[feature_list])
df_train2[feature_list] = scaler.transform(df_train2[feature_list])
df_val[feature_list] = scaler.transform(df_val[feature_list])
df_test[feature_list] = scaler.transform(df_test[feature_list])

print(df_train['retained_label'].mean())
print(df_train2['retained_label'].mean())
print(df_val['retained_label'].mean())
print(df_test['retained_label'].mean())




"""
Modeling Process:
There will be some performance metrics that i will be using:

Accuracy : Ratio of count of true/correct predictions to all predictions
Precision : Percentage of our correct positive predictions. It measures the extent of error caused by False Positives (FPs)
F1-Score : Harmonic mean of the precision and recall (percentage of actual positive samples was correctly classified)
ROC-AUC : Degree of separability/distinction or intermingling/crossover between the predictions of the positive and negative classes
Gini : Measure the separability and ordering of the predictions and actual values
"""
# Basic Models

# A. Logistic Regression
model_logreg = LogisticRegression()
model_logreg.fit(df_train[feature_list],df_train[target])


def Confusion_Matrix(y_test, ypred):
    cfmat = confusion_matrix(y_test, ypred, labels=[0, 1])
    print('Confusion Matrix: \n', classification_report(y_test, ypred, labels=[0, 1]))
    print("\n")
    print('TN - True Negative: {}'.format(cfmat[1, 1]))
    print('FP - False Positive: {}'.format(cfmat[1, 0]))
    print('FN - False Negative: {}'.format(cfmat[0, 1]))
    print('TP - True Positive: {}'.format(cfmat[0, 0]))
    print('Accuracy Rate: {}'.format(np.divide(np.sum([cfmat[0, 0], cfmat[1, 1]]), np.sum(cfmat))))
    print('Misclassification Rate: {}'.format(np.divide(np.sum([cfmat[0, 1], cfmat[1, 0]]), np.sum(cfmat))))
    print('Precision: {}'.format(np.divide(cfmat[0, 0], np.sum([cfmat[0, 0], cfmat[1, 0]]))))
    print('F1-Score: {}'.format(f1_score(y_test, ypred, average='macro')))
    print('ROC-AUC: {}'.format(roc_auc_score(y_test, ypred)))


def get_gini(a, b):
    try:
        return round(2 * roc_auc_score(a, b) - 1, 5)
    except:
        return None


ypred_logreg = model_logreg.predict(df_test[feature_list])
df_train['proba'] = model_logreg.predict_proba(df_train[feature_list])[:, 1]
df_val['proba'] = model_logreg.predict_proba(df_val[feature_list])[:, 1]
df_test['proba'] = model_logreg.predict_proba(df_test[feature_list])[:, 1]

Confusion_Matrix(df_test[target],ypred_logreg)

print('Gini on Train: ', get_gini(df_train['retained_label'], df_train['proba']))
print('Gini on Val: ', get_gini(df_val['retained_label'], df_val['proba']))
print('Gini on Test: ', get_gini(df_test['retained_label'], df_test['proba']))

df_test['proba_bin'] = pd.qcut(df_test['proba'],q=10)
df_test.groupby(['proba_bin']).agg({'proba':'mean','retained_label':'mean'})

# B. Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(df_train[feature_list],df_train[target])

ypred_rf = model_rf.predict(df_test[feature_list])
df_train['proba'] = model_rf.predict_proba(df_train[feature_list])[:, 1]
df_val['proba'] = model_rf.predict_proba(df_val[feature_list])[:, 1]
df_test['proba'] = model_rf.predict_proba(df_test[feature_list])[:, 1]

Confusion_Matrix(df_test[target],ypred_rf)

print('Gini on Train: ', get_gini(df_train['retained_label'], df_train['proba']))
print('Gini on Val: ', get_gini(df_val['retained_label'], df_val['proba']))
print('Gini on Test: ', get_gini(df_test['retained_label'], df_test['proba']))

df_test['proba_bin'] = pd.qcut(df_test['proba'],q=10)
df_test.groupby(['proba_bin']).agg({'proba':'mean','retained_label':'mean'})

# C. Extreme Gradient Boosting
model_xgb = XGBClassifier()
model_xgb.fit(df_train[feature_list],df_train[target])

ypred_xgb = model_xgb.predict(df_test[feature_list])
df_train['proba'] = model_xgb.predict_proba(df_train[feature_list])[:, 1]
df_val['proba'] = model_xgb.predict_proba(df_val[feature_list])[:, 1]
df_test['proba'] = model_xgb.predict_proba(df_test[feature_list])[:, 1]

Confusion_Matrix(df_test[target],ypred_xgb)

print('Gini on Train: ', get_gini(df_train['retained_label'], df_train['proba']))
print('Gini on Val: ', get_gini(df_val['retained_label'], df_val['proba']))
print('Gini on Test: ', get_gini(df_test['retained_label'], df_test['proba']))

df_test['proba_bin'] = pd.qcut(df_test['proba'],q=10)
df_test.groupby(['proba_bin']).agg({'proba':'mean','retained_label':'mean'})

# Basic Models With Oversampling Data

# A. Logistic Regression
model_logreg = LogisticRegression()
model_logreg.fit(df_train2[feature_list],df_train2[target])

ypred_logreg = model_logreg.predict(df_test[feature_list])
df_train2['proba'] = model_logreg.predict_proba(df_train2[feature_list])[:, 1]
df_val['proba'] = model_logreg.predict_proba(df_val[feature_list])[:, 1]
df_test['proba'] = model_logreg.predict_proba(df_test[feature_list])[:, 1]

Confusion_Matrix(df_test[target],ypred_logreg)

print('Gini on Train: ', get_gini(df_train2['retained_label'], df_train2['proba']))
print('Gini on Val: ', get_gini(df_val['retained_label'], df_val['proba']))
print('Gini on Test: ', get_gini(df_test['retained_label'], df_test['proba']))

df_test['proba_bin'] = pd.qcut(df_test['proba'],q=10)
df_test.groupby(['proba_bin']).agg({'proba':'mean','retained_label':'mean'})

# B. Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(df_train2[feature_list],df_train2[target])

ypred_rf = model_rf.predict(df_test[feature_list])
df_train2['proba'] = model_rf.predict_proba(df_train2[feature_list])[:, 1]
df_val['proba'] = model_rf.predict_proba(df_val[feature_list])[:, 1]
df_test['proba'] = model_rf.predict_proba(df_test[feature_list])[:, 1]

Confusion_Matrix(df_test[target],ypred_rf)

print('Gini on Train: ', get_gini(df_train2['retained_label'], df_train2['proba']))
print('Gini on Val: ', get_gini(df_val['retained_label'], df_val['proba']))
print('Gini on Test: ', get_gini(df_test['retained_label'], df_test['proba']))

df_test['proba_bin'] = pd.qcut(df_test['proba'],q=10)
df_test.groupby(['proba_bin']).agg({'proba':'mean','retained_label':'mean'})

# C. Extreme Gradient Boosting
model_xgb = XGBClassifier()
model_xgb.fit(df_train2[feature_list],df_train2[target])

ypred_xgb = model_xgb.predict(df_test[feature_list])
df_train2['proba'] = model_xgb.predict_proba(df_train2[feature_list])[:, 1]
df_val['proba'] = model_xgb.predict_proba(df_val[feature_list])[:, 1]
df_test['proba'] = model_xgb.predict_proba(df_test[feature_list])[:, 1]

Confusion_Matrix(df_test[target],ypred_xgb)

print('Gini on Train: ', get_gini(df_train2['retained_label'], df_train2['proba']))
print('Gini on Val: ', get_gini(df_val['retained_label'], df_val['proba']))
print('Gini on Test: ', get_gini(df_test['retained_label'], df_test['proba']))

df_test['proba_bin'] = pd.qcut(df_test['proba'],q=10)
df_test.groupby(['proba_bin']).agg({'proba':'mean','retained_label':'mean'})

# Model Hyperparameter Tuning

# A. Grid Search Hyperparameter Tuning
def best_model(X, y):
    models = {
        'Logistic_Regression': {
            'model': LogisticRegression(),
            'params': {'penalty': ['l1', 'l2'],
                       'solver': ['newton-cg', 'lbfgs'],
                       'max_iter': [20, 30]}
        },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'params': {'n_estimators': [100, 200],
                       'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2'],
                       'max_depth': [5, 10], 'max_leaf_nodes': [5, 10]}
        }}

    scores = []
    cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
    for model_name, config in models.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'Model': model_name,
            'Best_Score': gs.best_score_,
            'Best_Params': gs.best_params_
        })

    return scores, pd.DataFrame(scores, columns=['Model', 'Best_Score', 'Best_Params'])

p,q = best_model(df_train[feature_list],df_train[target])

# B. Bayesian Optimization
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }


def objective(space):
    clf = XGBClassifier(
        n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
        colsample_bytree=int(space['colsample_bytree']))

    evaluation = [(df_train[feature_list], df_train[target]), (df_val[feature_list], df_val[target])]

    clf.fit(df_train[feature_list], df_train[target],
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10, verbose=False)

    pred = clf.predict(df_val[feature_list])
    accuracy = accuracy_score(df_val[target], pred > 0.5)
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)

# Model With Tuned Hyperparameter

# A. Logistic Regression
model_logreg = LogisticRegression(max_iter= 30, penalty= 'l2', solver= 'lbfgs')
model_logreg.fit(df_train[feature_list],df_train[target])

ypred_logreg = model_logreg.predict(df_test[feature_list])
df_train['proba'] = model_logreg.predict_proba(df_train[feature_list])[:, 1]
df_val['proba'] = model_logreg.predict_proba(df_val[feature_list])[:, 1]
df_test['proba'] = model_logreg.predict_proba(df_test[feature_list])[:, 1]

Confusion_Matrix(df_test[target],ypred_logreg)

print('Gini on Train: ', get_gini(df_train['retained_label'], df_train['proba']))
print('Gini on Val: ', get_gini(df_val['retained_label'], df_val['proba']))
print('Gini on Test: ', get_gini(df_test['retained_label'], df_test['proba']))

df_test['proba_bin'] = pd.qcut(df_test['proba'],q=10)
df_test.groupby(['proba_bin']).agg({'proba':'mean','retained_label':'mean'})

# B. Random Forest
model_rf = RandomForestClassifier(criterion= 'gini',
   max_depth= 10,
   max_features= 'sqrt',
   max_leaf_nodes= 10,
   n_estimators= 200)

model_rf.fit(df_train[feature_list],df_train[target])

ypred_rf = model_rf.predict(df_test[feature_list])
df_train['proba'] = model_rf.predict_proba(df_train[feature_list])[:, 1]
df_val['proba'] = model_rf.predict_proba(df_val[feature_list])[:, 1]
df_test['proba'] = model_rf.predict_proba(df_test[feature_list])[:, 1]

Confusion_Matrix(df_test[target],ypred_rf)

print('Gini on Train: ', get_gini(df_train['retained_label'], df_train['proba']))
print('Gini on Val: ', get_gini(df_val['retained_label'], df_val['proba']))
print('Gini on Test: ', get_gini(df_test['retained_label'], df_test['proba']))

df_test['proba_bin'] = pd.qcut(df_test['proba'],q=10)
df_test.groupby(['proba_bin']).agg({'proba':'mean','retained_label':'mean'})

# C. Extreme Gradient Boosting
model_xgb = XGBClassifier(colsample_bytree= 0.546746256699493,
                          gamma= 2.562308691353075,
                          max_depth= 15,
                          min_child_weight= 8.0,
                          reg_alpha= 40,
                          reg_lambda= 0.4570338892)

model_xgb.fit(df_train[feature_list],df_train[target])

ypred_xgb = model_xgb.predict(df_test[feature_list])
df_train['proba'] = model_xgb.predict_proba(df_train[feature_list])[:, 1]
df_val['proba'] = model_xgb.predict_proba(df_val[feature_list])[:, 1]
df_test['proba'] = model_xgb.predict_proba(df_test[feature_list])[:, 1]

Confusion_Matrix(df_test[target],ypred_xgb)

print('Gini on Train: ', get_gini(df_train['retained_label'], df_train['proba']))
print('Gini on Val: ', get_gini(df_val['retained_label'], df_val['proba']))
print('Gini on Test: ', get_gini(df_test['retained_label'], df_test['proba']))

df_test['proba_bin'] = pd.qcut(df_test['proba'],q=10)
df_test.groupby(['proba_bin']).agg({'proba':'mean','retained_label':'mean'})




"""
Summary
"""
# Model Feature Shap Analysis
explainer = shap.Explainer(model_xgb, df_train[feature_list])
shap_values = explainer(df_train[feature_list])
shap.plots.bar(shap_values,max_display=12)

# Model Score/Proba and Retention Rate Separation
df_test['proba_bin'] = pd.qcut(df_test['proba'],q=10)
df_test.groupby(['proba_bin']).agg({'proba':'mean','retained_label':'mean'})







































