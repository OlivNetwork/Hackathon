"""
Customer Behavior Analytics:

In this project, we will analyze customer behavior on our platform,then we will recommend which
banners are associated with conversion to be displayed based on our findings.
"""

# Importing essential libraries

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from pandas.plotting import parallel_coordinates
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


# %matplotlib inline

#color palletes
color_blue = '#697AF5'
color_purp = '#A37DF5'
color_seblue = '#58A1E8'
light = '#EBBDFF'

colors = [color_blue, color_purp, light, color_seblue]


# Data Preparing and Processing
df = pd.read_csv('../input/how-to-do-product-analytics/product.csv', parse_dates = True, infer_datetime_format = True)
df.head()

# Dropping unuseful columns to free up memory
df.drop('page_id', axis=1, inplace=True)
df.info()

# Checking for nulls and duplicates
print(f"There is {df.isna().sum()} null values")
print(f"There is {df.duplicated().sum()} duplicated rows")

# drop duplicated rows
df.drop_duplicates()

# Convert time column to datetime
df['time'] = pd.to_datetime(df['time'])
df.head()

df.user_id.nunique(),df.user_id.count()

"""
Probability:

Here we are going to explore different probabilities for events such as banner show, banner click 
and order for the two website versions.
"""
df.target.value_counts('target')*100
df.groupby(df['site_version']).target.value_counts('target')[1]
df.groupby(df['site_version']).target.value_counts('target')[3]

# mobile version is represented more than desktop version

p_version = df.site_version.value_counts(normalize=True)

p_version.plot(kind='barh', figsize=(16, 6), color=color_blue)

plt.title('User proportion Vs. Site version', fontweight="bold", fontsize=14)
plt.axvline(p_version[1])
plt.axvline(p_version[0])

# Given that mobile version is way more represented than desktop version,  we will use proportions instead.
rates = (df.groupby(['site_version']).title.value_counts('site_version')*100).unstack()

rates.plot(kind='bar', figsize=(16, 6), color=colors)

plt.title('Rates By Website Versions', fontweight="bold", size=14)
plt.ylabel('Percentage User')

# Splitting the data based on site version
desk = df[df['site_version'] == 'desktop']
mob = df[df['site_version'] == 'mobile']

# desktop conversion rate
desk[desk['title'] == 'order'].shape[0]/desk[desk['title'] == 'banner_click'].shape[0]

# mobile conversion rate
mob[mob['title'] == 'order'].shape[0]/mob[mob['title'] == 'banner_click'].shape[0]
df.head()

df['product'].value_counts(normalize=True)

# Split the data into converted and non converted
converted = df.query('target == 1')
converted.head()

converted.groupby('site_version').product.value_counts(normalize=True).unstack()

converted.groupby('site_version').product.value_counts(normalize=True).unstack().plot(kind='bar', figsize=(16, 6), color=colors)
plt.title('Conversion Rates By Products', fontweight="bold", size=14)
plt.ylabel('Percentage User')


"""
Product Performance:
"""

prod_trends = df.copy()
prod_trends['date'] = prod_trends['time'].dt.date
prod_trends['weekday'] = prod_trends['time'].dt.day_name()
prod_trends['hour'] = prod_trends['time'].dt.hour

prod_trends.head()

mobile = prod_trends[prod_trends['site_version'] == 'mobile']
desk = prod_trends[prod_trends['site_version'] == 'desktop']
desk.head()

mob_clicks = mobile[mobile['title'] == 'banner_click']
mobile_prod = mob_clicks.groupby(['date', 'product']).user_id.agg('nunique')
mobile_prod_df = pd.DataFrame(mobile_prod.unstack(level=1))

desk_clicks = desk[desk['title'] == 'banner_click']
desk_prod = desk_clicks.groupby(['date', 'product']).user_id.agg('nunique')
desk_prod_df = pd.DataFrame(desk_prod.unstack(level=1))

desk_prod_df.head()


# This function makes 2 trendline plots at once
def trend_time(data_1, data_2):
    plt.figure(1)

    data_1.plot(figsize=(12, 6), color=colors)
    title_1 = str(input("title")).title()
    xlabels = str(input("xlable")).title()
    ylabels = str(input("ylabel")).title()
    plt.title(title_1, fontsize=20, fontweight='bold')
    plt.xlabel(xlabels, fontweight='bold')
    plt.ylabel(ylabels, fontweight='bold')

    plt.figure(2)
    data_2.plot(figsize=(12, 6), color=colors)
    title_2 = str(input("title")).title()
    plt.title(title_2, fontsize=20, fontweight='bold')
    plt.xlabel(xlabels, fontweight='bold')
    plt.ylabel(ylabels, fontweight='bold')

    plt.show()


#plot_function

#trend_time(mobile_prod_df,desk_prod_df)

# Purchases
mobile_orders = mobile[mobile['target'] == 1]
mobile_orders_prod =mobile.groupby(['date', 'product']).order_id.agg('nunique')
mobile_orders_prod_df = pd.DataFrame(mobile_orders_prod.unstack(level=1))

desk_orders = desk[desk['target'] == 1]
desk_orders_prod =desk.groupby(['date', 'product']).order_id.agg('nunique')
desk_orders_prod_df = pd.DataFrame(desk_orders_prod.unstack(level=1))

#trend_time(mobile_orders_prod_df,desk_orders_prod_df)

# Seasonality analysis
# Sort the data by Id and Event Date
df_sorted = df.sort_values(['user_id', 'time'], ascending=True)


def get_mon(x):
    return dt.datetime(x.year, x.month, 1)


s_df = df_sorted.copy()

s_df['weekday'] = s_df['time'].dt.weekday

s_df_desktop = s_df[s_df['site_version'] == 'desktop']
s_df_mob = s_df[s_df['site_version'] == 'mobile']

s_df_desktop['event_month'] = s_df_desktop['time'].apply(get_mon)
s_df_mob['event_month'] = s_df_mob['time'].apply(get_mon)

s_df_mob['acquired_date'] = s_df_mob.groupby('user_id')['event_month'].transform('min')
s_df_mob.head()

day_mob = s_df_mob.groupby('weekday').order_id.nunique()
day_mob.plot(kind='barh', figsize=(16, 6), color=color_blue)
plt.title('Sales By WeekDays - Mobile Users', fontweight="bold", size=14)
plt.ylabel('Number Of Unique Orders', fontweight='bold')

day_desk = s_df_desktop.groupby('weekday').order_id.nunique()
day_desk.plot(kind='barh', figsize=(16, 6), color=color_purp)
plt.title('Sales By WeekDays - Desktop Users', fontweight="bold", size=14)
plt.ylabel('Number Of Unique Orders', fontweight='bold')

# Because users made multiple events, we will aggregate by user id and date, to get if the user converted in a certain month or not
purchases_desk = s_df_desktop.groupby(['user_id', 'event_month']).agg({'target': ['sum']})
purchases_desk.columns = purchases_desk.columns.droplevel(level=1)
purchases_desk.reset_index(inplace=True)
purchases_desk.head()

purchases_mob = s_df_mob.groupby(['user_id', 'event_month']).agg({'target': ['sum']})
purchases_mob.columns = purchases_mob.columns.droplevel(level=1)
purchases_mob.reset_index(inplace=True)
purchases_mob.head()

purchases_desk = purchases_desk.groupby('event_month').agg({'target': ['mean']})
purchases_desk.columns = purchases_desk.columns.droplevel(level=1)
purchases_desk.reset_index(inplace=True)


purchases_mob = purchases_mob.groupby('event_month').agg({'target': ['mean']})
purchases_mob.columns = purchases_mob.columns.droplevel(level=1)
purchases_mob.reset_index(inplace=True)

conv_mob = s_df_mob.set_index('time').groupby(
    pd.Grouper(freq='M')
).agg({'target': ['mean']}).reset_index()

conv_desk = s_df_desktop.set_index('time').groupby(
    pd.Grouper(freq='M')
).agg({'target': ['mean']}).reset_index()

conv_desk.plot(x='time', y='target', figsize=(16, 6), color=colors)
plt.title("Conversion Rates By Month Using Desktop Devices", fontsize=20, fontweight='bold')
plt.xlabel('Month', fontweight='bold')
plt.ylabel('Conversion Rates', fontweight='bold')

conv_mob.plot(x='time', y='target', figsize=(16, 6), color=colors)
plt.title("Conversion Rates By Month Using Mobile Devices", fontsize=20, fontweight='bold')
plt.xlabel('Month', fontweight='bold')
plt.ylabel('Conversion Rates', fontweight='bold')


"""
The Mean Duration Customer Takes To Convert:
"""

# New column 'events' to calculate the count of events a user made
df_sorted['events'] = df_sorted.groupby('user_id')['user_id'].transform('count')
df_sorted.head()

# Selecting the subset of users with multiple events
multiple_events = df_sorted[df_sorted['events'] > 1]
multiple_events.head()

# New column for calculating the time between user events
multiple_events['duration'] = multiple_events.groupby('user_id')['time']\
                                .transform('diff').dt.days\
                                .fillna(0)

# Selecting users whom converted
multiple_events_converted = multiple_events[multiple_events['target'] == 1]
multiple_events_converted.head()


# This function removes outliers
def truncated_mean(data):
    top = data.quantile(.9)
    low = data.quantile(.1)

    trunc_data = data[(data <= top) & (data >= low)]
    mean = trunc_data.mean()
    return(mean)

# Calculating conversion duration mean and median for each site version
multiple_events_converted.groupby('site_version').duration.agg(['mean','median',truncated_mean])

valse = pd.DataFrame(multiple_events_converted.duration.value_counts().reset_index())
valse.sort_values('duration', ascending=False)
valse.columns = ['days_to_convert', 'user_count']
valse.head()


def plot_graph(data, col1, col2):
    data.plot(x=col1, y=col2, figsize=(16, 8), color=colors)
    plt.title(col1.replace('_', ' ') + ' Vs ' + col2.replace('_', ' '), fontsize=20, fontweight='bold')
    plt.xlabel(col1.replace('_', ' '), fontweight='bold')
    plt.ylabel(col2.replace('_', ' '), fontweight='bold')
    plt.show()


plot_graph(valse, 'days_to_convert', 'user_count')


"""
Which Banner Show / Click Is Driving More Conversions:
"""

# This gets the action row before conversion
converted = multiple_events.loc[multiple_events['target'] == 1]
prior_conv = multiple_events.loc[multiple_events['target'].shift(-1) == 1]
merged = converted.merge(prior_conv, how='outer', sort=True)
merged['event_product'] = merged['title'] + '_on_' + merged['product']
merged.head()

actions = pd.DataFrame(merged.groupby('user_id').event_product.unique().reset_index())
journey = actions['event_product']

journey = list(journey)

# Instantiate transaction encoder and identify unique items in transactions
encoder = TransactionEncoder().fit(journey)

# One-hot encode transactions
onehot = encoder.transform(journey)

# Convert one-hot encoded data to DataFrame
onehot = pd.DataFrame(onehot, columns=encoder.columns_)

# Print the one-hot encoded transaction dataset
onehot.head()

onehot.mean()

# Compute frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(onehot,
                            min_support=0.006,
                            use_colnames=True)

frequent_itemsets.head(10)

print(len(frequent_itemsets)),print(len(onehot.columns))

rules_1 = association_rules(frequent_itemsets,
                            metric="support",
                             min_threshold=0.0015)

rules_1.sort_values(['support', 'confidence'], ascending=False).head()

#Getting only the antecedents that predicts a conversion on clothes products
targeted_rules = rules_1[(rules_1['consequents'] == {'order_on_clothes'})].copy()
targeted_rules_acc = rules_1[(rules_1['consequents'] == {'order_on_accessories'})].copy()
targeted_rules_sn = rules_1[(rules_1['consequents'] == {'order_on_sneakers'})].copy()
targeted_rules_co = rules_1[(rules_1['consequents'] == {'order_on_company'})].copy()
ttargeted_rules_sn = rules_1[(rules_1['consequents'] == {'order_on_sports_nutrition'})].copy()
ttargeted_rules_sn.sort_values(['support', 'confidence'], ascending=False)

#Further filtration to see more accurate results with high support and confidence and lower randomness
filtered_rules = targeted_rules_sn[(targeted_rules_sn['antecedent support'] > 0.01) &
                               (targeted_rules_sn['support'] > 0.009) &
                               (targeted_rules_sn['confidence'] > 0.10)]

filtered_rules.sort_values(['support', 'confidence'], ascending=False)

data_list = [targeted_rules, targeted_rules_acc, targeted_rules_sn, targeted_rules_co, ttargeted_rules_sn]


def filter_rules(data):
    for x in data:
        filtered_rules = x[(x['antecedent support'] > 0.01) &
                           (x['support'] > 0.009) &
                           (x['confidence'] > 0.10)]

        return (display(data[0].sort_values(['support', 'confidence'], ascending=False).head()),
                display(data[1].sort_values(['support', 'confidence'], ascending=False).head()),
                display(data[2].sort_values(['support', 'confidence'], ascending=False).head()),
                display(data[3].sort_values(['support', 'confidence'], ascending=False).head()),
                display(data[4].sort_values(['support', 'confidence'], ascending=False).head()))


filter_rules(data_list)

rules_1['antecedents'] = rules_1['antecedents'].apply(lambda a: ','.join(list(a)))
rules_1['consequents'] = rules_1['consequents'].apply(lambda a: ','.join(list(a)))
rules_1[['antecedents', 'consequents']]

support_table = rules_1.pivot(index='consequents', columns='antecedents', values='support')

plt.figure(figsize=(16, 10))

sns.heatmap(support_table, annot=True, cmap='Blues')
plt.title("Banners Associations Heatmap", fontsize=20, fontweight='bold')
plt.xlabel('Antecedents', fontweight='bold')
plt.ylabel('Consequents', fontweight='bold')

# Compute frequent itemsets using the Apriori algorithm
frequent_itemsets_2 = apriori(onehot,
                            min_support=0.020,
                            use_colnames=True,
                            max_len=2)

rules_2 = association_rules(frequent_itemsets_2,
                            metric="support",
                            min_threshold=0.00)

rules_2['antecedents'] = rules_2['antecedents'].apply(lambda a: list(a)[0])
rules_2['consequents'] = rules_2['consequents'].apply(lambda a: list(a)[0])
rules_2['rule'] = rules_2.index

coords = rules_2[['antecedents', 'consequents', 'rule']]
coords.head(2)

plt.figure(figsize = (16,10))
parallel_coordinates(coords, 'rule',colormap= 'ocean')
plt.title("Parallel Coordinates - Support is 0.020", fontsize = 20, fontweight = 'bold')


"""
Cohort Analysis & Retention:
"""


# A function that gets the month and year for any order the customer made
def get_month(x):
    return dt.datetime(x.year, x.month, 1)


retention = df.copy()
retention['date'] = retention['time'].dt.date
retention['month'] = retention['time'].apply(get_month)

#mobile
retention_purchased = retention[retention['target'] == 1]
retention_purchased_mob = retention_purchased[retention_purchased['site_version'] == 'mobile']

#desktop
retention_purchased_des = retention_purchased[retention_purchased['site_version'] == 'desktop']

retention_purchased_mob.head()

monthly_repeat_customers_df_mob = retention_purchased_mob.set_index('time').groupby([
    pd.Grouper(freq='M'), 'user_id'# group the index InvoiceDate by each month and by CustomerID
]).filter(lambda x: len(x) > 1). resample('M').nunique()['user_id']


monthly_repeat_customers_df_des = retention_purchased_des.set_index('time').groupby([
    pd.Grouper(freq='M'), 'user_id'# group the index InvoiceDate by each month and by CustomerID
]).filter(lambda x: len(x) > 1). resample('M').nunique()['user_id']

monthly_unique_customers_df_mob = retention_purchased_mob.set_index('time')['user_id'].resample('M').nunique()

monthly_unique_customers_df_des = retention_purchased_des.set_index('time')['user_id'].resample('M').nunique()

monthly_repeat_percentage_mob = monthly_repeat_customers_df_mob / monthly_unique_customers_df_mob * 100.0

monthly_repeat_percentage_des = monthly_repeat_customers_df_des / monthly_unique_customers_df_des * 100.0

# Visualize all this data in a chart
ax = pd.DataFrame(monthly_repeat_customers_df_mob.values).plot(
    figsize=(10,7)
)

pd.DataFrame(monthly_unique_customers_df_mob.values).plot(
    ax=ax,
    grid=True
)


ax2 = pd.DataFrame(monthly_repeat_percentage_mob.values).plot.bar(
    ax=ax,
    grid=True,
    secondary_y=True, # add another y-axis on the rightside of the chart
    color=colors,
    alpha=0.2
)

ax.set_xlabel('date')
ax.set_ylabel('number of customers')
ax.set_title('Number of All vs. Repeat Customers Over Time - Mobile')

ax2.set_ylabel('percentage (%)')

ax.legend(['Repeat Customers', 'All Customers'])
ax2.legend(['Percentage of Repeat'], loc='upper right')


ax2.set_ylim([0, 20])

plt.xticks(
    range(len(monthly_repeat_customers_df_mob.index)),
    [x.strftime('%m.%Y') for x in monthly_repeat_customers_df_mob.index],
    rotation=45
)

plt.show()

# Visualize all this data in a chart
ax = pd.DataFrame(monthly_repeat_customers_df_des.values).plot(
    figsize=(10,7)
)

pd.DataFrame(monthly_unique_customers_df_des.values).plot(
    ax=ax,
    grid=True
)


ax2 = pd.DataFrame(monthly_repeat_percentage_des.values).plot.bar(
    ax=ax,
    grid=True,
    secondary_y=True, # add another y-axis on the rightside of the chart
    color='blue',
    alpha=0.2
)

ax.set_xlabel('date')
ax.set_ylabel('number of customers')
ax.set_title('Number of All vs. Repeat Customers Over Time - Desktop')

ax2.set_ylabel('percentage (%)')

ax.legend(['Repeat Customers', 'All Customers'])
ax2.legend(['Percentage of Repeat'], loc='upper right')


ax2.set_ylim([0, 20])

plt.xticks(
    range(len(monthly_unique_customers_df_des.index)),
    [x.strftime('%m.%Y') for x in monthly_unique_customers_df_des.index],
    rotation=45
)

plt.show()


def get_month (x):
    return dt.datetime(x.year, x.month,1)

# Extracting orders data
orders = df[df['title'] == 'order']

# Convert time column to a timeseries data type
orders['time'] = pd.to_datetime(orders['time'])

# In this cell we get the first month the user made an order
orders['acquired_month'] = retention['time'].apply(get_month)
orders['cohort_month'] = orders.groupby('user_id').acquired_month.transform('min')

# Extracting the month from both acquired and cohort dates
acq_month = orders['acquired_month'].dt.month
coh_month = orders['cohort_month'].dt.month

#This formula calculates the number of months the user have been retained
month_diff = (acq_month - coh_month)+1

# Cohort index column refers to how many months have the customer been retained
orders['cohort_index'] = month_diff
orders.head()

# Splitting the data by Site version.
Desktop_retention = orders[orders['site_version'] == 'desktop']
mobile_retention = orders[orders['site_version'] == 'mobile']

# Group unique costumer counts by month and index
cohort_data_desktop = Desktop_retention.groupby(['cohort_month','cohort_index']).user_id.nunique().reset_index()
cohort_data_mobile = mobile_retention.groupby(['cohort_month','cohort_index']).user_id.nunique().reset_index()

cohort_data_desktop.head()

# Creating cohort tables for both versions
cohort_table_desktop = cohort_data_desktop.pivot(index='cohort_month', columns='cohort_index', values='user_id')
cohort_table_mob = cohort_data_mobile.pivot(index='cohort_month', columns='cohort_index', values='user_id')

# Converting values to percentages
cohort_desk_per = cohort_table_desktop.divide(cohort_table_desktop.iloc[:, 0], axis=0)
cohort_mob_perc = cohort_table_mob.divide(cohort_table_mob.iloc[:, 0], axis=0)

cohort_desk_per.index = cohort_table_desktop.index.strftime('%B %Y')
cohort_mob_perc.index = cohort_table_mob.index.strftime('%B %Y')

# Plotting cohort tables for both versions
plt.figure(figsize=(10, 16))

plt.subplot(2, 1, 1)
sns.heatmap(cohort_desk_per, annot=True, cmap='Blues')
plt.title("Cohort Analysis For Desktop Users", fontsize=20, fontweight='bold')
plt.xlabel('Cohort Index', fontweight='bold')
plt.ylabel('Cohort Month', fontweight='bold')

plt.subplot(2, 1, 2)
sns.heatmap(cohort_mob_perc, annot=True, cmap='Blues')
plt.title("Cohort Analysis For Mobile Users", fontsize=20, fontweight='bold')
plt.xlabel('Cohort Index', fontweight='bold')
plt.ylabel('Cohort Month', fontweight='bold')
plt.show()














