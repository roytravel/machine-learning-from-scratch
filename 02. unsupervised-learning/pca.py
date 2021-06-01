import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn.decomposition import PCA

"""
Abstract: Find the hyperplane closet to the data, and then projected onto the hyperplane.
1. Find axis which
Reference: https://excelsior-cjh.tistory.com/167
"""

class Handmade_PCA(object):
    '''
    PCA is divided by two method: eigen-value-decomposition, singular-value-decomposition
    scikit learn use SVD because it need not store the covariance matrix on the memory
    '''

    # eigen-value-decomposition
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None


    def fit(self, X):

        self.mean = np.mean(X, axis=0) # 평균을 0으로
        X = X - self.mean

        # 1. calculate covariance
        cov = np.cov(X.T)

        ## use if you want use singular value decomposition
        # singular_value, singular_vector, v_t = np.linalg.svd(X)

        # 2. eigen values, eigen vectors (Weight, Vector)
        eigen_values, eigen_vectors = np.linalg.eig(cov)

        # 3. get the eigen values in descending order
        eigen_vectors = eigen_vectors.T
        idxs = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idxs]
        eigen_vectors = eigen_vectors[idxs]

        # store first n eigen vectors
        self.components = eigen_vectors[0: self.n_components]


    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)


class Library_PCA(object):
    def __init__(self, n_components):
        self.n_components = n_components


    def pca(self, X):
        pca = PCA(self.n_components)
        X_tf = pca.fit_transform(X)
        return X_tf


def visualizer(X_projected, y, flag):
    fig = plt.figure()
    scatter = plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(flag)
    plt.colorbar()
    plt.savefig(flag)
    fig.show()


def main():
    # dataset 1
    # X, y = load_digits(return_X_y=True)
    # ind = np.arange(len(X))
    # np.random.shuffle(ind)
    # X, y = X[ind], y[ind]

    # dataset 2
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Handmade Principal Components Analysis
    handmade_pca = Handmade_PCA(2)
    handmade_pca.fit(X)
    X_projected = handmade_pca.transform(X)
    print(f"[*] Dimension reduction : {X.shape} -> {X_projected.shape}")

    # Library Principal Compoents Analysis
    library_pca = Library_PCA(2)
    X_tf = library_pca.pca(X)

    # Visualization
    visualizer(X_projected, y, "Handmade PCA")
    visualizer(X_tf, y, "Library PCA")
    plt.show()



if __name__ == "__main__":
    sys.exit(main())