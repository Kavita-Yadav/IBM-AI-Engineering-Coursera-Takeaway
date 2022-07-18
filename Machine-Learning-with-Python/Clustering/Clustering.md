
# Introduction to Clustering
A group of objects that are similar to other objects in the cluster, and dissimilar to data points in other clusters.

### Clustering Applications:
- Retail/Marketing:
    - Identifying buying patterns of customers.
    - Recommending new books or movies to new customers.
- Banking:
    - Fraud detection in credit card use
    - Identifying clusters of customers(e.g., loyal).
- Insurance:
    - Fraud detection in claims analysis.
    - Insurance risk of customers.
- Publication:
    - Auto-categorizing news based on thier content.
    - Recommending similar news articles
- Medicine:
    - Characterizing patient behaviour.
- Biology:
    - Clustering genetic makers to identify family ties.

### Why Clustering?
- Exploratory data analysis.
- Summary generation or reducing the scale.
- Outlier detection(especially to be used for fraud detection or noise removal.)
- Finding duplicates and datasets.
- Pre-processing step(for prediction, other data mining tasks)

### Clustering Algorithm:
- Partitioned based Clustering:
    - realtively efficient.
    - Eg: k-Means, k-Median, Fuzzy c-Means.
- Hierarchical Clustering:
    - Produces trees of clusters
    - Eg: Agglomerative, Divisive
- Density-based Clustering:
    - Produced arbitary shaped clusters
    - Eg: DBSCAN


### K-Means Clustering:
Determine the similarity and dissimilarity.

#### K-Means Algorithm:
- Partitioning Clustering
- K-means divides the data into non-overlapping subsets(clusters) without any cluster-internal structure.
- Examples within a cluster are very similar.
- Examples across different clusters are very different.
- Maximize inter-cluster distance.
- Minimize intra-cluster distance.

1. Intialize k:
- Determine number of cluster means Initialize k(centroids) value.
- Two approaches to choose centroids, we can randomly choose three observation point out of the dataset and use them as intial means OR we can create three random points as centroids of the clusters which is our choice.
2. Calculate the distance:
- Assign each customer to the closest center. For that, we have to calculate the distance of each data point.
3. Assign to centroid:
- Assign each point to the closest centroid.
- Main objective of K-Means clustering is to minimize the distance of data points from the centroid of this cluster and maximize the distance from other cluster centroid.
- Error is the total distance of each point from its centroid. It can be calculated as SSE(Sum of Square Error).
- we should shape clusters in such a way that the total distance of all members of a cluster from its centroid be minimized.
- For better cluster, we need to move centroid and accordingly data points will recluster. We need to continue this process until the centroid no longer moves.
4. Compute new centroids:
- Compute the new centroids for each cluster.
5. Repeat:
- repeat until there are no more changes.
6. For best result, We need to run the whole process multiple times with different starting condition of random centroid selection.


#### To Summarize:
1. Randomly placing k centroids, one for each cluster.
2. calculate the distance of each point from each centroid(Eculidien distance is widely used to measure distance).
3. Assign each data point (object) to its closest centroid, creating a cluster.
4. Recalculate the position of the k centroids.


#### How to calculate the accuracy of k-means ?
- Extrnal approach:
    - Compare the clusters with the ground truth, if it is available.
- Internal approach:
    - Average the distance between data points within a cluster.
- Because k-Means is an unsupervised algorithm we usually don't have ground truth in real world problems to be used. But there is still a way to say how bad each cluster is, based on the objective of the k-Means. This value is the average distance between data points within a cluster. Also, average of the distances of data points from their cluster centroids can be used as a metric of error for the clustering algorithm.
- The correct choice of K is often ambiguous because it's very dependent on the shape and scale of the distribution of points in a dataset. This means increasing K will always decrease the error.
- The elbow point is determined where the rate of decrease sharply shifts. It is the right K for clustering. This method is called the elbow method.

### Hierarchical Clustering:
Hierarchical clustering algorithms build a hierarchy of clusters where each node is a clutser consists of the clusters of its daughter nodes. Two approach: Agglomerative and divisive.

- Agglomerative clustering:
    - It proceed by merging the clusters. It is bottoms -up approach.
    1. Create n clusters, one for each data point.
    2. Compute the Proximity Matrix.
    3. Repeat:
        - Merge the two closest clusters.
        - Update the proximity matrix.
    4. Until only a single cluster remains.
    - Distance between clusters.
    1. Single-Linkage Clustering: Minimum distance between clusters.
    2. Complete-Linkage Clustering: Maximum distance between clusters.
    3. Average Linkage Clustering: Average distance between clusters.
    4. Centroid Linkage Clustering: Distance between cluster centroids.

Advantages:
- Doesn't required number of clusters to be specified.
- Easy to implement.
- Produce a dendrogram, which helps with understanding the data.
Disadvantages:
- Can never undo any previous steps throughout the algorithm.
- Generally has long runtimes.
- Sometimes difficult to Identify the number of clusters by the dendrogram.

Hierarchical clustering Vs. K-means:
| K-means | Hierarchical Clustering |
| --| --|
| 1. Much more efficient | 1. Can be slow for large datasets |
| 2. Requires the number of clusters to be specified | 2. Doesn't require the number of clusters to run |
| 3. Gives only one partitioning of the data based on the predefined number of clusters | 3. Gives more than one partitioning depending on the resolution |
| 4. Potentially returns different clusters each time it is run due to random intialization of centroids | 4. Always generates the same clusters. |

### Density-based clustering:
Two type of clusters:
- Spherical-shape clusters.
- Arbitrary-shape clusters.

K-Means vs density-based clustering:
- K-Means assigns all points to a cluster even if they do not belong in any. Whereas Density-based Clustering locates regions of high density, and separates outliers.

What is DBSCAN?
- DBSCAN(Density-Based Spatial Clustering of Applications with Noise):
    - Is one of the most common clustering algorithms.
    - Works based on density of objects.
- R(Radius of neighborhood):
    - Radius(R) that if includes enough number of points within, we call it a dense area.
- M(Min number of neighbors):
    - The minimum number of data points we want in a neighborhood to define a cluster.

`Core point:`  A data point is a core point if within our neighborhood of the point there are at least M points. For example, as there are six points in the two centimeter neighbor of the red point, we mark this point as a core point. 
`Border point:` A data point is a border point if A; its neighbourhood contains less than M data points or B; it is reachable from some core point. Here, reachability means it is within our distance from a core point. 
`Outlier:` If it is not a core point nor a border point then it will be labeled as outliers.

- Identify all above three points.
- Connect core points that are neighbors and put them in the same cluster. So, a cluster is formed as at least one core point plus all reachable core points plus all their borders. It's simply shapes all the clusters and finds outliers as well.

Advantages of DBSCAN:
- Arbitrarily shaped clusters.
- Robust to outliers.
- Does not require specification of the number of clusters.
