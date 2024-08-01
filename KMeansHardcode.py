# K-Means Clustering Algorithm without Scikit Learn
# Abhyuday Singh
# Online references used for visualization. Works on the Mall_Customers dataset from Assignment 1

import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
dataset.describe()

# Features used are Annual Income and Spending Score
X = dataset.iloc[:, [3, 4]].values

m = X.shape[0]  # Number of training examples
n = X.shape[1]  # Number of features = 2
n_iter = 100

# Number of clusters
K = 5

# Make random centroids
centroids = np.array([]).reshape(n, 0)
for i in range(K):
    rand = rd.randint(0, m-1)
    centroids = np.c_[centroids, X[rand]]

# Clusters as keys with the values in them being the examples in the cluster
Output = {}

# Euclidean Distance
ED = np.array([]).reshape(m, 0)
for k in range(K):
    tempDist = np.sum((X-centroids[:, k]) ** 2, axis=1)
    ED = np.c_[ED, tempDist]
C = np.argmin(ED, axis=1) + 1

Y = {}
for k in range(K):
    Y[k + 1] = np.array([]).reshape(2, 0)

for i in range(m):
    Y[C[i]] = np.c_[Y[C[i]], X[i]]

for k in range(K):
    Y[k + 1] = Y[k + 1].T

for k in range(K):
    centroids[:, k] = np.mean(Y[k + 1], axis=0)


for i in range(n_iter):
    ED = np.array([]).reshape(m, 0)
    for k in range(K):
        tempDist = np.sum((X - centroids[:, k]) ** 2, axis=1)
        ED = np.c_[ED, tempDist]
    C = np.argmin(ED, axis=1) + 1
    # step 2.b
    Y = {}
    for k in range(K):
        Y[k + 1] = np.array([]).reshape(2, 0)
    for i in range(m):
        Y[C[i]] = np.c_[Y[C[i]], X[i]]

    for k in range(K):
        Y[k + 1] = Y[k + 1].T

    for k in range(K):
        centroids[:, k] = np.mean(Y[k + 1], axis=0)
    Output = Y

# Plot unclustered data
plt.scatter(X[:, 0], X[:, 1], c='black', label='Unclustered data')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.title('Plot of data points')
plt.show()

# Plot clustered data
color = ['red', 'blue', 'green', 'yellow', 'magenta']
labels = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5']
for k in range(K):
    plt.scatter(Output[k+1][:, 0], Output[k+1][:, 1], c=color[k], label=labels[k])
plt.scatter(centroids[0, :], centroids[1, :], s=150, c='black', label='Centroids')
plt.xlabel('Income')
plt.ylabel('Number of transactions')
plt.legend()
plt.show()
