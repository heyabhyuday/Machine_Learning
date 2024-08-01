import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

df = pd.read_csv("Mall_Customers.csv")
encoder = LabelEncoder()

df['Gender'] = encoder.fit_transform(df['Gender'])

methods = ['Euclidean']
for method in methods:
    dist = pdist(df, metric=method)
    linkagematrix = linkage(dist, "ward")
    dendrogram(linkagematrix, color_threshold=1, labels=df.index)
    plt.title(method)
    plt.show()

distances = {}
for method in methods:
    distances[method] = pdist(df, metric=method)
finalmethod = 'Euclidean'
linkagematrix = linkage(distances[finalmethod], "ward")

plt.figure(figsize=(10, 7))
clusters = fcluster(linkagematrix, t=5, criterion='distance')
df['cluster'] = clusters

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['cluster'], cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Hierarchical Clustering')

plt.show()
