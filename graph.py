import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import trim_mean

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes):
        centers = KMeans(n_clusters=n_nodes).fit(X).cluster_centers_

        dist = squareform(pdist(centers))
        np.fill_diagonal(dist, np.inf)
        sigma = trim_mean(np.min(dist, axis=1), .25)
        gamma = 2 / sigma**2

        labels = SpectralClustering(n_clusters=self.n_classes, gamma=gamma).fit(centers).labels_

        self.clusters = []

        for i in range(self.n_classes):
            nodes = centers[labels == i]
            mst = minimum_spanning_tree(squareform(pdist(nodes))).toarray()
            self.clusters.append((mst, nodes))
