"""
market segmentation:

1. Demographic Segmentation
Often the most common and easiest to create, demographic segmentation covers the standard, factual or statistical type information about people. Think, all the different “demographics” you might find in your Google Analytics review.

Some examples of demographic segmentation include:

Age
Gender
Income
Occupation
Family size
Race
Religion
Marital Status
Education
Ethnicity

2. Psychographic Segmentation
Psychographic segmentation focuses on grouping your target audience by their personalities or inner
traits. These aren’t quite as obvious as demographics because they don’t show up on the surface.

These are some examples of psychographic segmentation:

Values
Goals
Needs
Pain points
Hobbies
Personality traits
Interests
Political party affiliation
Sexual orientation

3. Geographic Segmentation
Geographic segmentation is, you guessed it, market segmentation based on the customer’s geographic
location. This type of segmentation is especially useful if you have multiple brick-and-mortar locations
or offices.

These are a few ways you might think about creating a geographic segment:

Zip code/post code
City
Country
Population density
Distance from a certain location (like your office or store)
Climate
Time zone
Dominate language

4. Behavioral Segmentation
Behavioral segmentation focuses on the actions your target audience takes. Similar to psychographic
segmentation, behavioral segmentation can often take a bit more research and data than geographic or
demographic because it goes deeper than surface-level info.

However, unlike psychographic, behavioral segmentation tends to focus more on purchases and interactions
rather than opinions or thoughts.

Here are some examples of behavioral market segmentation you may want to consider:

Purchasing habits
Brand interactions (for example, following or interacting on social media vs calling customer service)
Spending habits
Customer loyalty
Actions taken on a website (such as reading a blog or signing up for your newsletter)
"""

"""
customer segmentation:

1. Recommender systems & Collaboration Filtering & Content-Based Filtering.
Recommender systems are techniques that allow companies to develop sales and marketing and as a result, 
attract more customers. A recommendation system tries to predict the evaluation of a product made by the
user or which product the user will prefer. Collaborative filtering is based on the assumption that 
people who agreed in the past will agree in the future, and that they will like similar kinds of items 
as they liked in the past. Recommender systems work very simply like this: For example, if there are two
people named Robert and William, let's assume that after Robert likes a product, William likes that 
product too. Later, when Robert likes another product, this time the recommender system presents the 
second product Robert liked in front of William.

Clustering is used to build recommender systems, this is a detailed topic, but in short: 
using customers' weighted RFM(Recency - Frequency - Monetary) and expectation maximization clustering 
algorithms and their combination with the closest K-neighbors, recommendations for each cluster is 
independently extracted.

In Content-Based Filtering, on the other hand, two things are handled, the first is the user's profile,
that is, what he likes or dislikes, and the second is the product profile, that is, the product's 
features. If the user profile and the product profile match, the product is recommended to the user.

2. Special Offers
As a result of the evaluations made through customer segmentation, the profit volume of the companies 
can be increased by making special campaigns for the target audience.

3. Threat and Fraud Detection
Outlier data, that is, outlier for clustering, does not belong to any group, since this outlier data is
a threat and fraud, we teach them to the machine like this.

4. Filling Missing Data
For successful customer segmentation, there must be no missing data. Clustering provides an important 
benefit in filling the missing data, for example, if the customer's salary information is missing, 
instead of filling it with the average of all customers' salaries, it is better to fill it with the 
average salary of the cluster that the customer is in. Thus, the empty parts in the data set are filled
with more accurate data, which positively affects the performance of the models. 
"""

# Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import sys
import warnings
from sklearn.preprocessing import LabelEncoder

if not sys.warnoptions:
    warnings.simplefilter("ignore")

data_path = '../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv'
df = pd.read_csv(data_path)

df.head()

df.drop('CustomerID', axis=1, inplace = True)
df.head()

df.shape()
df.info()

df.isnull().sum()

# visualize the correlations.
cor = df.corr()
sns.set(font_scale=1.4)
plt.figure(figsize=(9,8))
sns.heatmap(cor, annot=True, cmap='plasma')
plt.tight_layout()
plt.show()

# -Distribution Plots-
plt.figure(figsize=(16,12),facecolor='#9DF08E')

# Spending Score
plt.subplot(3,3,1)
plt.title('Spending Score\n', color='#FF000B')
sns.distplot(df['Spending Score (1-100)'], color='orange')

# Age
plt.subplot(3,3,2)
plt.title('Age\n', color='#FF000B')
sns.distplot(df['Age'], color='#577AFF')

# Annual Income
plt.subplot(3,3,3)
plt.title('Annual Income\n', color='#FF000B')
sns.distplot(df['Annual Income (k$)'], color='black')

plt.suptitle(' Distribution Plots\n', color='#0000C1', size = 30)
plt.tight_layout()

# Before-After Label Encoder
print('\033[0;32m' + 'Before Label Encoder\n' + '\033[0m' + '\033[0;32m', df['Gender'])

le = LabelEncoder()
df['Gender'] = le.fit_transform(df.iloc[:,0])

print('\033[0;31m' + '\n\nAfter Label Encoder\n' + '\033[0m' + '\033[0;31m', df['Gender'])

# calculate how much to shop for which gender
spending_score_male = 0
spending_score_female = 0

for i in range(len(df)):
    if df['Gender'][i] == 1:
        spending_score_male = spending_score_male + df['Spending Score (1-100)'][i]
    if df['Gender'][i] == 0:
        spending_score_female = spending_score_female + df['Spending Score (1-100)'][i]


print('\033[1m' + '\033[93m' + f'Males Spending Score  : {spending_score_male}')
print('\033[1m' + '\033[93m' + f'Females Spending Score: {spending_score_female}')

# try to understand the relationship between gender and spending score.
# Number of genders

plt.figure(figsize=(16, 16), facecolor='#54C6C0')
plt.subplot(3, 3, 1)
plots = sns.barplot(x=['Female', 'Male'], y=df['Gender'].value_counts(), data=df)

for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=13, xytext=(0, 8),
                   textcoords='offset points', color='red')

plt.xlabel("Gender", size=14)
plt.ylabel("Number", size=14)
plt.yticks(np.arange(0, 116, 10), size='14')
plt.grid(False)
plt.title("Number of Genders\n", color="red", size='22')

# Gender & Total Spending Score
list_genders_spending_score = [int(spending_score_female), int(spending_score_male)]
series_genders_spending_score = pd.Series(data=list_genders_spending_score)

plt.subplot(3, 3, 2)
plots = sns.barplot(x=['Female', 'Male'], y=series_genders_spending_score, palette=['yellow', 'purple'])

for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=13, xytext=(0, 8),
                   textcoords='offset points', color='red')
plt.xlabel("Gender", size=14)
plt.ylabel("Total Spending Score", size=14)
plt.yticks(np.arange(0, 6001, 1000), size='14')
plt.grid(False)
plt.title("Gender & Total Spending Score\n", color="red", size='22')

# Gender & Mean Spending Score

list_genders_spending_score_mean = [int(spending_score_female / df['Gender'].value_counts()[0]),
                                    int(spending_score_male / df['Gender'].value_counts()[1])]
series_genders_spending_score_mean = pd.Series(data=list_genders_spending_score_mean)

plt.subplot(3, 3, 3)
plots = sns.barplot(x=['Female', 'Male'], y=series_genders_spending_score_mean, palette='hsv')

for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=13, xytext=(0, 8),
                   textcoords='offset points', color='red')

plt.xlabel("Gender", size=14)
plt.ylabel("Mean Spending Score", size=14)
plt.yticks(np.arange(0, 71, 10), size='14')
plt.grid(False)
plt.title("Gender & Mean Spending Score\n", color="red", size='22')
plt.tight_layout()
plt.show()

# visualize the relationship between Age and Spending score
plt.figure(figsize=(12,8))
sns.scatterplot(x = df['Age'], y = df['Spending Score (1-100)'])
plt.title('Age - Spending Score', size = 23, color='red')

# visualize the relationship between Annual Income and Spending Score
plt.figure(figsize=(12,8))
sns.scatterplot(x = df['Annual Income (k$)'], y = df['Spending Score (1-100)'], palette = "red")
plt.title('Annual Income - Spending Score', size = 23, color='red')




"""
CLUSTERINGS 
(Check out the documentation for some other methods of dimensionality reduction and 
encoding)
"""
# PCA
# x assignment
x = df.iloc[:,0:].values
print("\033[1;31m"  + f'X data before PCA:\n {x[0:5]}')


# standardization before PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)


# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_2D = pca.fit_transform(X)
print("\033[0;32m" + f'\nX data after PCA:\n {X_2D[0:5,:]}')

# finding optimum number of clusters
from sklearn.cluster import KMeans
wcss_list = []

for i in range(1,11):
    kmeans_test = KMeans(n_clusters = i, init ='k-means++', random_state=88)
    kmeans_test.fit(X_2D)
    wcss_list.append(kmeans_test.inertia_)

plt.figure(figsize=(9,6))
plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method', color='red',fontsize='23')
plt.xlabel('Number of clusters')
plt.xticks(np.arange(1,11))
plt.ylabel('WCSS')
plt.show()

# KMeans
kmeans = KMeans(n_clusters = 4, init ='k-means++', random_state=88)
y_kmeans = kmeans.fit_predict(X_2D)

# clusters visualization
plt.figure(1 , figsize = (16 ,9))
plt.scatter(X_2D[y_kmeans == 0, 0], X_2D[y_kmeans == 0, 1], s = 80, c = 'orange', label = 'Cluster-1')
plt.scatter(X_2D[y_kmeans == 1, 0], X_2D[y_kmeans == 1, 1], s = 80, c = 'red', label = 'Cluster-2')
plt.scatter(X_2D[y_kmeans == 2, 0], X_2D[y_kmeans == 2, 1], s = 80, c = 'green', label = 'Cluster-3')
plt.scatter(X_2D[y_kmeans == 3, 0], X_2D[y_kmeans == 3, 1], s = 80, c = 'purple', label = 'Cluster-4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 375, c = 'brown', label = 'Centroids')
plt.title("Customers' Clusters")
plt.xlabel('PCA Variable-1', color='red')
plt.ylabel('PCA Variable-2', color='red')
plt.legend()
plt.show()




"""
Clustering (Age & Annual Income & Spending Score)
"""
# x assignment
x = df[['Age','Annual Income (k$)','Spending Score (1-100)']].values
x_df = df[['Age','Annual Income (k$)','Spending Score (1-100)']] # this line for 3d scatter plot

# finding optimum number of clusters
wcss_list = []

for i in range(1,11):
    kmeans_test = KMeans(n_clusters = i, init ='k-means++', random_state=88)
    kmeans_test.fit(x)
    wcss_list.append(kmeans_test.inertia_)

plt.figure(figsize=(9,6))
plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method', color='red',fontsize='23')
plt.xlabel('Number of clusters')
plt.xticks(np.arange(1,11))
plt.ylabel('WCSS')
plt.show()

# KMeans
kmeans = KMeans(n_clusters = 6, init ='k-means++', random_state=88)
clusters = kmeans.fit_predict(x_df)
x_df['label'] = clusters

# # clusters visualization
fig = px.scatter_3d(data_frame=x_df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',color = 'label', size = 'label')
fig.show()



"""
Clustering (Age & Annual Income)
"""
# x assignment
x = df[['Age','Annual Income (k$)']].values

# finding optimum number of clusters
wcss_list = []

for i in range(1,11):
    kmeans_test = KMeans(n_clusters = i, init ='k-means++', random_state=88)
    kmeans_test.fit(x)
    wcss_list.append(kmeans_test.inertia_)

plt.figure(figsize=(9,6))
plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method', color='red',fontsize='23')
plt.xlabel('Number of clusters')
plt.xticks(np.arange(1,11))
plt.ylabel('WCSS')
plt.show()

# KMeans
kmeans = KMeans(n_clusters = 2, init ='k-means++', random_state=88)
y_kmeans = kmeans.fit_predict(x)

# clusters visualization
plt.figure(1 , figsize = (16 ,9))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 80, c = '#13DB8C', label = 'Cluster-1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 80, c = '#72BAFF', label = 'Cluster-2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 350, c = 'brown', label = 'Centroids')
plt.title("Customers' Clusters")
plt.xlabel('Age', color='red')
plt.ylabel('Annual Income (k$)', color='red')
plt.legend()
plt.show()



"""
Clustering (Annual Income & Spending Score)
"""
# x assignment
x = df[['Annual Income (k$)','Spending Score (1-100)']].values

# finding optimum number of clusters
wcss_list = []

for i in range(1,11):
    kmeans_test = KMeans(n_clusters = i, init ='k-means++', random_state=88)
    kmeans_test.fit(x)
    wcss_list.append(kmeans_test.inertia_)

plt.figure(figsize=(9,6))
plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method', color='red',fontsize='23')
plt.xlabel('Number of clusters')
plt.xticks(np.arange(1,11))
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 5, init ='k-means++', random_state=88)
y_kmeans = kmeans.fit_predict(x)

# clusters visualization
plt.figure(1 , figsize = (16 ,9))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 80, c = 'orange', label = 'Cluster-1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 80, c = 'red', label = 'Cluster-2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 80, c = 'purple', label = 'Cluster-3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 80, c = 'lime', label = 'Cluster-4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 80, c = 'blue', label = 'Cluster-5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 375, c = 'brown', label = 'Centroids')
plt.title("Customers' Clusters")
plt.xlabel('Annual Income (k$)', color='red')
plt.ylabel('Spending Score', color='red')
plt.legend()
plt.show()




"""
Clustering (Age & Spending Score)
"""
# x assignment
x = df[['Age','Spending Score (1-100)']].values

# finding optimum number of clusters
wcss_list = []

for i in range(1,11):
    kmeans_test = KMeans(n_clusters = i, init ='k-means++', random_state=88)
    kmeans_test.fit(x)
    wcss_list.append(kmeans_test.inertia_)

plt.figure(figsize=(9,6))
plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method', color='red',fontsize='23')
plt.xlabel('Number of clusters')
plt.xticks(np.arange(1,11))
plt.ylabel('WCSS')
plt.show()

# KMeans
kmeans = KMeans(n_clusters = 4, init ='k-means++', random_state=88)
y_kmeans = kmeans.fit_predict(x)

# clusters visualization
plt.figure(1 , figsize = (16 ,9))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 80, c = '#00FF00', label = 'Cluster-1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 80, c = '#00FFFF', label = 'Cluster-2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 80, c = '#FF00FF', label = 'Cluster-3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 80, c = '#FF4500', label = 'Cluster-4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 375, c = 'brown', label = 'Centroids')
plt.title("Customers' Clusters")
plt.xlabel('Age', color='red')
plt.ylabel('Spending Score', color='red')
plt.legend()
plt.show()


































