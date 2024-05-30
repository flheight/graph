import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

class Graph:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, n_nodes):
        centers = KMeans(n_clusters=n_nodes).fit(X).cluster_centers_

        dist = squareform(pdist(centers))
        np.fill_diagonal(dist, np.inf)
        sigma = np.mean(np.min(dist, axis=1))
        gamma = 2 / sigma**2

        spectral_clustering = SpectralClustering(n_clusters=self.n_classes, gamma=gamma).fit(centers)
        spectral_labels = spectral_clustering.labels_

        self.clusters = []

        for i in range(self.n_classes):
            nodes = centers[spectral_labels == i]
            mst = minimum_spanning_tree(squareform(pdist(nodes))).toarray()
            self.clusters.append((mst, nodes))
