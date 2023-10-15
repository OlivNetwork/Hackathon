"""
This is a recommender system project could be used on chain for Ecommerce Recommendation.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import calendar
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objs as go
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from lightfm import LightFM
from lightfm.evaluation import auc_score
import cufflinks as cf
import gc
import plotly.graph_objs as go

pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Reading Data
df_events = pd.read_csv('/kaggle/input/ecommerce-dataset/events.csv')

df_events['event_datetime'] = pd.to_datetime(df_events['timestamp'], unit = 'ms')

df_items1 = pd.read_csv('/kaggle/input/ecommerce-dataset/item_properties_part1.csv')
df_items2 = pd.read_csv('/kaggle/input/ecommerce-dataset/item_properties_part2.csv')
df_items = pd.concat([df_items1, df_items2])

df_items['event_datetime'] = pd.to_datetime(df_items['timestamp'], unit = 'ms')

df_category = pd.read_csv('/kaggle/input/ecommerce-dataset/category_tree.csv')

# Data Exploration
df_events['event_datetime'].describe()
df_events.head(10)

df_events['event'].value_counts()

sns.countplot(x = 'event', data = df_events, palette="icefire")

# Check Duplication
df_duplicate = df_events[df_events.duplicated(keep ='first')]

# df_duplicate
# df_events

df_events['visitorid'].value_counts()

df_events.loc[df_events.visitorid == 1150086]

most_viewed_items = df_events[df_events['event'] == 'view'].itemid.value_counts()

# most_viewed_items[:10].plot.bar(x = 'item id', y = 'views count',figsize = (10, 6))
x = most_viewed_items[:5].index
y = most_viewed_items[:5].values
sns.barplot(x = x,
            y = y,
            order = x,
            palette="icefire")

# Check the most added to cart item
most_added_to_cart_items = df_events[df_events['event'] == 'addtocart'].itemid.value_counts()

x = most_added_to_cart_items[:5].index
y = most_added_to_cart_items[:5].values
sns.barplot(x = x,
            y = y,
            order = x,
            palette="icefire")

# do the same for the most purchased item
# check the number of transactions occured
df_events[df_events['event'] == 'transaction'].shape[0]

most_purchased_items = df_events[df_events['event'] == 'transaction'].itemid.value_counts()

x = most_purchased_items[:5].index
y = most_purchased_items[:5].values
sns.barplot(x = x,
            y = y,
            order = x,
            palette="icefire")


"""
Feature Engineering
"""
#extract date from timestamp
df_events = df_events.assign(date=pd.Series(datetime.datetime.fromtimestamp(i/1000).date() for i in df_events.timestamp))
df_events.head()


def get_day(x):
    day = calendar.day_name[x.weekday()]
    return day


df_events['day_of_week'] = df_events['event_datetime'].map(get_day)
df_events['Year'] = df_events['event_datetime'].map(lambda x: x.year)
df_events['Month'] = df_events['event_datetime'].map(lambda x: x.month)
df_events['Day'] = df_events['event_datetime'].map(lambda x: x.day)
df_events['Hour'] = df_events['event_datetime'].map(lambda x: x.hour)
df_events['minute'] = df_events['event_datetime'].map(lambda x: x.minute)

df_events.describe()


def get_time_periods(hour):
    if hour >= 3 and hour < 7:
        return 'Dawn'
    elif hour >= 7 and hour < 12:
        return 'Morning'
    elif hour >= 12 and hour < 16:
        return 'Afternoon'
    elif hour >= 16 and hour < 22:
        return 'Evening'
    else:
        return 'Night'



df_events['Day Period'] = df_events['Hour'].map(get_time_periods)
df_events['Day Period'].value_counts()

start_date = '2015-5-3'
end_date = '2015-8-31'
fd = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
df_events_viz = df_events[(df_events.date >= fd(start_date))
                   & (df_events.date <= fd(end_date))]


# insights about purchasing behaviour during day
data = pd.DataFrame(df_events.groupby(by=['Day Period','event'])['event_datetime'].count()).reset_index()
fig, count = plt.subplots(figsize = (15,10))

sorted_periods = {'Dawn': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4, 'Late Night': 5}
data['Day Period Value'] = data['Day Period'].map(sorted_periods)
data = data.sort_values('Day Period Value').drop('Day Period Value',axis=1)
data = data.reset_index(drop =True)
line = sns.lineplot(x=data['Day Period'], y=data['event_datetime'],sort = False,hue = data['event'])
line.set_title('Total activity of the day')
line.set_ylabel('Count')
line.set_xlabel('Day Period')

data = pd.DataFrame(df_events[df_events['event'] == "transaction"]["Day Period"].value_counts()).reset_index()
fig, count = plt.subplots(figsize = (15,10))

sorted_periods = {'Dawn': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4, 'Late Night': 5}
data['Day Period Value'] = data['index'].map(sorted_periods)
data = data.sort_values('Day Period Value').drop('Day Period Value',axis=1)
data = data.reset_index(drop =True)
line = sns.lineplot(x=data['index'], y=data['Day Period'],sort = False)
line.set_title('Total purchase activity of the day')
line.set_ylabel('Count')
line.set_xlabel('Day Period')



"""
Funnal Analysis
"""
init_notebook_mode(connected=True)
cf.go_offline(connected=True)

brand = df_events['itemid'].value_counts()
# create a list of brands (different from Not available) mentioned more than 10000 times.
brand_list = [461686]
# filter out a list of rows with brands in top 6.1%
best_brands = df_events[df_events['itemid'].isin(brand_list)]
fig = go.Figure()
for i in range(len(brand_list)):
    name = str(brand_list[i])
    j = best_brands[best_brands['itemid']==int(name)]['event'].value_counts()

    fig.add_trace(go.Funnel(
        name = name,
        y = j.index,
        x = j,
        orientation = "h",
        textposition = "inside",
        textinfo = "value+percent initial"))

fig.update_layout(
    title_text='Customer behavior for the top brand', # title of plot
    yaxis_title_text='Customer behavior', # xaxis label
    xaxis_title_text='Brand performance', # yaxis label
    )

# fig.show(renderer="colab")
fig.show()



"""
Hourly Traffic on website
"""
grp_by_hr_event_type = df_events_viz.groupby(['Hour','event']).count()
layout= dict(title="Hourly Store Traffic", xaxis_title="Time of Day", yaxis_title="Number of  Users")
grp_by_hr_event_type['visitorid'].unstack(1).iplot(kind="bar", layout=layout) #Check the peak hrs





"""
Conversion Rate
"""
custmoer_behavior_share =df_events.event.value_counts()/len(df_events)*100
custmoer_behavior_share

labels = custmoer_behavior_share.index.tolist()
values = custmoer_behavior_share.values.tolist()

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)), )
fig.update_layout(title="Customer Beahviour",

                  font=dict(
                      family="Courier New, monospace",
                      size=18,
                      color="#7f7f7f"))
# fig.show(renderer="colab")
fig.show()




"""
Recommender:

We need to make some kind of recommendations to the user, like People also bought as an example. 
This requires us to have two latent vectors from purchases data (items, users) and apply 
Facorization Machine algorithm on them to get the recommendations.

One kind of recommendations that we can make is to show the users some items that people also 
bought with the items that they are viewing now. This can guide our customers to get better items 
and make the shopping process easier.
"""
# create a list of users who made a single purchase
customers_purchases = df_events[df_events['transactionid'].notnull()].visitorid.unique()
# create a list of items that people purchased
purchased_items = []

for customer in customers_purchases:
    purchased_items.append(list(df_events.loc[(df_events.visitorid == customer)
                                               & (df_events.transactionid.notnull())].itemid.values))


def recommend_items(item_id, purchased_items):
    recommended_items = []

    for items in purchased_items:

        if item_id in items:
            recommended_items += items

    # reomve duplicated items and merge the lists in one lsit
    recommended_items = list(set(recommended_items) - set([item_id]))
    return recommended_items



"""
Events Prediction
"""
df_events_lr = df_events.copy()

df_events_lr['purchased'] = df_events_lr.transactionid.notnull().astype(int)

le = LabelEncoder()
categorical_cols = list(df_events_lr.select_dtypes(include=object))
df_events_lr[categorical_cols] = df_events_lr[categorical_cols].apply(le.fit_transform)

X = df_events_lr.drop(['visitorid', 'event_datetime', 'transactionid', 'date', 'timestamp', 'event'], axis = 'columns')
y = df_events_lr.purchased

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = 0.7)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# use the model to predict the test features
y_pred_class = logreg.predict(X_test)

# Generate the prediction values for each of the test observations using predict_proba() function rather than just predict
preds = logreg.predict_proba(X_test)[:,1]

# Store the false positive rate(fpr), true positive rate (tpr) in vectors for use in the graph
fpr, tpr, _ = metrics.roc_curve(y_test, preds)

# Store the Area Under the Curve (AUC) so we can annotate our graph with theis metric
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc = "lower right")
plt.show()



"""
Data Preparation
"""
# create the final matrix
df_events = df_events.sort_values('date').reset_index(drop=True)
df_events = df_events[['visitorid','itemid','event', 'date']]
df_events.head(5)

start_date = '2015-5-3'
end_date = '2015-5-18'
fd = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
events = df_events[(df_events.date >= fd(start_date))
                   & (df_events.date <= fd(end_date))]

# split the data into train and test data
split_point = np.int(np.round(df_events.shape[0]*0.8))
events_train = df_events.iloc[0:split_point]
events_test = df_events.iloc[split_point::]
# check that visitorid and itemid already exist on the train data
events_test = events_test[(events_test['visitorid'].isin(events_train['visitorid']))
                          & (events_test['itemid'].isin(events_train['itemid']))]


"""
Label Encoding
"""
id_cols=['visitorid','itemid']
trans_cat_train = dict()
trans_cat_test = dict()

for k in id_cols:
    cat_enc = preprocessing.LabelEncoder()
    trans_cat_train[k] = cat_enc.fit_transform(events_train[k].values)
    trans_cat_test[k] = cat_enc.transform(events_test[k].values)

ratings = dict()

cat_enc = preprocessing.LabelEncoder()
ratings['train'] = cat_enc.fit_transform(events_train.event)
ratings['test'] = cat_enc.transform(events_test.event)

n_users=len(np.unique(trans_cat_train['visitorid']))
n_items=len(np.unique(trans_cat_train['itemid']))

rate_matrix = dict()

rate_matrix['train'] = coo_matrix((ratings['train'],
                                   (trans_cat_train['visitorid'],
                                    trans_cat_train['itemid'])),
                                    shape=(n_users,n_items))

rate_matrix['test'] = coo_matrix((ratings['test'],
                                   (trans_cat_test['visitorid'],
                                    trans_cat_test['itemid'])),
                                    shape=(n_users,n_items))




"""
LightFM Model
"""
gc.collect()

del df_events_viz, df_events_lr

gc.collect()

# model creation and training
model = LightFM(no_components=10, loss='warp')
model.fit(rate_matrix['train'], epochs=100, num_threads=8)

auc_score(model, rate_matrix['train'], num_threads=8).mean()

auc_score(model, rate_matrix['test'], num_threads=10).mean()
















