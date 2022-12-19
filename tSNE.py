# Import the necessary libraries
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate three random datasets
data1, labels1 = make_blobs(n_samples=1000, n_features=10, random_state=0, centers=6)
data2, labels2 = make_blobs(n_samples=1000, n_features=10, random_state=1, centers=6)
data3, labels3 = make_blobs(n_samples=1000, n_features=10, random_state=2, centers=6)

# Create a PCA instance and fit the data
pca = PCA(n_components=2)
pca_results1 = pca.fit_transform(data1)
pca_results2 = pca.fit_transform(data2)
pca_results3 = pca.fit_transform(data3)

# Visualize the results
fig, axs = plt.subplots(3, 4)

# Visualize the results of PCA on the three datasets
axs[0, 0].scatter(pca_results1[:, 0], pca_results1[:, 1], c=labels1)
axs[0, 0].set_title("PCA 1")

axs[1, 0].scatter(pca_results2[:, 0], pca_results2[:, 1], c=labels2)
axs[1, 0].set_title("PCA 2")

axs[2, 0].scatter(pca_results3[:, 0], pca_results3[:, 1], c=labels3)
axs[2, 0].set_title("PCA 3")

# Create a TSNE instance with perplexity 10 and fit the data
tsne1 = TSNE(n_components=2, perplexity=10)
tsne_results1 = tsne1.fit_transform(data1)
tsne_results2 = tsne1.fit_transform(data2)
tsne_results3 = tsne1.fit_transform(data3)

# Visualize the results of tSNE with perplexity 10 on the three datasets
axs[0, 1].scatter(tsne_results1[:, 0], tsne_results1[:, 1], c=labels1)
axs[0, 1].set_title("TSNE 1 (perplexity=10)")

axs[0, 2].scatter(tsne_results2[:, 0], tsne_results2[:, 1], c=labels2)
axs[0, 2].set_title("TSNE 1 (perplexity=30)")

axs[0, 3].scatter(tsne_results3[:, 0], tsne_results3[:, 1], c=labels3)
axs[0, 3].set_title("TSNE 1 (perplexity=50)")

# Create a TSNE instance with perplexity 30 and fit the data
tsne2 = TSNE(n_components=2, perplexity=30)
tsne_results1 = tsne2.fit_transform(data1)
tsne_results2 = tsne2.fit_transform(data2)
tsne_results3 = tsne2.fit_transform(data3)

# Visualize the results of tSNE with perplexity 30 on the three datasets
axs[1, 1].scatter(tsne_results1[:, 0], tsne_results1[:, 1], c=labels1)
axs[1, 1].set_title("TSNE 2 (perplexity=10)")

axs[1, 2].scatter(tsne_results2[:, 0], tsne_results2[:, 1], c=labels2)
axs[1, 2].set_title("TSNE 2 (perplexity=30)")

axs[1, 3].scatter(tsne_results3[:, 0], tsne_results3[:, 1], c=labels3)
axs[1, 3].set_title("TSNE 2 (perplexity=50)")

# Create a TSNE instance with perplexity 50 and fit the data
tsne3 = TSNE(n_components=2, perplexity=50)
tsne_results1 = tsne3.fit_transform(data1)
tsne_results2 = tsne3.fit_transform(data2)
tsne_results3 = tsne3.fit_transform(data3)

# Visualize the results of tSNE with perplexity 50 on the three datasets
axs[2, 1].scatter(tsne_results1[:, 0], tsne_results1[:, 1], c=labels1)
axs[2, 1].set_title("TSNE 3 (perplexity=10)")

axs[2, 2].scatter(tsne_results2[:, 0], tsne_results2[:, 1], c=labels2)
axs[2, 2].set_title("TSNE 3 (perplexity=30)")

axs[2, 3].scatter(tsne_results3[:, 0], tsne_results3[:, 1], c=labels3)
axs[2, 3].set_title("TSNE 3 (perplexity=50)")

fig.tight_layout()
plt.show()
