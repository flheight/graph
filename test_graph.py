import numpy as np
import matplotlib.pyplot as plt
import umap
from datasets import load_dataset
from graph import Graph

image_dataset = load_dataset('mnist', split='train', streaming=False, trust_remote_code=True)

np.random.seed(0)

X = np.stack([np.array(image).reshape(-1) for image in image_dataset['image']])
y = np.array(image_dataset['label'])

X_reduced = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], color='black', s=.5)

net = Graph(n_classes=10)

print("here")
net.fit(X_reduced, 50)
print("done")

colors = plt.cm.tab10(np.arange(10))

for i, (mst, nodes) in enumerate(net.clusters):
    plt.scatter(nodes[:, 0], nodes[:, 1], color=colors[i], label=f'Cluster {i}')
    for r in range(mst.shape[0]):
        for c in range(mst.shape[1]):
            if mst[r, c]:
                plt.plot([nodes[r, 0], nodes[c, 0]],
                         [nodes[r, 1], nodes[c, 1]],
                         color=colors[i], linewidth=3)

plt.show()
