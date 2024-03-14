from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# %config InlineBackend.figure_format = "svg" # jupyter

def t_SNE(data, label):
    X_tsne = TSNE(perplexity=10, n_components=2, random_state=33).fit_transform(data)
    X_pca = PCA(n_components=2).fit_transform(data)

    font = {"color": "darkred",
            "size": 13,
            "family" : "serif"}

    # plt.style.use("dark_background")
    plt.style.use("default")
    plt.figure(figsize=(8.5, 4))


    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 17)) # 17
    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar(ticks=range(17)) # 17
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 16.5) # (-0.5, 15.5)


    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 17)) # 17
    plt.title("PCA", fontdict=font)
    cbar = plt.colorbar(ticks=range(17)) # 17
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 16.5) # (-0.5, 15.5)
    plt.tight_layout()

    plt.show()
