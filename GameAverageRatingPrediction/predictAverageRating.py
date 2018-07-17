# -*- coding: utf-8 -*-
"""
@author: F34R
"""

import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#Read the data
games = pandas.read_csv("data/games.csv")
print(games.columns)
print(games.shape)

plt.hist(games["average_rating"])
plt.show()

# Removed any rows without user reviews.
games = games[games["users_rated"] > 0]
# Removed any rows with missing values.
games = games.dropna(axis=0)

# Get only the numeric columns from games.
good_columns = games._get_numeric_data()
//Using elbow method to find the optimum number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(good_columns)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans_model = KMeans(n_clusters=5, inti = 'k-means++', random_state=42)
kmeans_model.fit(good_columns)

# Get the cluster assignments.
labels = kmeans_model.labels_

# Created a PCA model.
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()

print(games.corr()["average_rating"])

# Get all the columns from the dataframe.
columns = games.columns.tolist()

# Filtered the columns to remove ones I don't want.
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name"]]

# Store the variable we'll be predicting on.
target = "average_rating"

# Generate the training set.  Set random_state to be able to replicate results.
train = games.sample(frac=0.7, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = games.loc[~games.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)

lr = LinearRegression()
# Fit the model to the training data.
model = lr.fit(train[columns], train[target])

predictions = model.predict(test[columns])
print("Mean Squared Error: " + str(mean_squared_error(predictions, test[target])))
