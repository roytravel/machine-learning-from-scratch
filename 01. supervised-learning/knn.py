import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_wine, load_iris, load_digits
from sklearn.model_selection import train_test_split

# Reference: https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/knn.py


class KNearestNeighbor(object):
    def __init__(self, k=None):
        self.k = k


    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(np.square(x1 - x2)))


    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X_test):
        tmp = []
        for X in X_test:
            euc_dist = [self._euclidean_distance(X_train, X) for X_train in self.X_train]
            k_idx = np.argsort(euc_dist)[: self.k]
            k_neighbor_labels = [self.y_train[i] for i in k_idx]

            # Get the first order in k_neighbor_labels
            most_common = Counter(k_neighbor_labels).most_common(1)
            tmp.append(most_common[0][0])
        y_pred = np.array(tmp)
        return y_pred


    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)


def main():

    # 1. load dataset 
    X, y = load_digits(return_X_y=True)

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=521)

    # 3. Create model
    knn = KNearestNeighbor(k=15)

    # 4. Training model
    knn.fit(X_train, y_train)

    # 5. Predict data
    y_pred = knn.predict(X_test)

    # 6. Accuracy
    accuracy = knn.accuracy(y_test, y_pred)
    print (f"[*] K-nearest neighbor : {accuracy}")


if __name__ == '__main__':
    sys.exit(main())