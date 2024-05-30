import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import make_moons
from graph import Graph

np.random.seed(0)

X, y = make_moons(n_samples=1024, noise=0.1, random_state=42)

plt.scatter(X[:, 0], X[:, 1], color='black', s=.5)

net = Graph(n_classes=2)

net.fit(X, 15)

colors = plt.cm.tab10(np.arange(2))

for i, (mst, nodes) in enumerate(net.clusters):
    plt.scatter(nodes[:, 0], nodes[:, 1], color=colors[i], label=f'Cluster {i}')
    for r in range(mst.shape[0]):
        for c in range(mst.shape[1]):
            if mst[r, c]:
                plt.plot([nodes[r, 0], nodes[c, 0]],
                         [nodes[r, 1], nodes[c, 1]],
                         color=colors[i], linewidth=3)

plt.show()
