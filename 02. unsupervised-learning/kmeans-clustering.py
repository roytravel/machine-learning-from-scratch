import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs

class Kmeans(object):

    # Implementation Order.
    # 01. Initialize random centroids (중심점 랜덤 초기화)
    # 02. Create cluster (가까운 중심점에 샘플을 할당하여 클러스터 생성)
    # 03. Closet centroid to the sample (샘플과 가까운 중심점 찾기)
    # 04. Calcaulte new centroid (클러스터에 속한 샘플들의 거리를 평균으로 새로운 중심점 계산)
    # 05. Get cluster labels to classify the sample (샘플 분류를 위한 클러스터 인덱스 확인)
    # 06. Predict sample (cluster indicies 반환)

    def __init__(self, k=2, max_iters=500, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps


    def _init_random_centroids(self, X):
        # get the number of samples and features from sample X (샘플 개수와 샘플 종류를 전체 샘플 X의 shape으로부터 추출)
        n_samples, n_features = np.shape(X)

        # Initialize centroids point(X, Y) using number of centroid and nubmer of features (중심점 개수와, 피처 개수를 통해 중심점 좌표 초기화)
        centroids = np.zeros((self.k, n_features))

        for i in range(self.k):
            # Choice the random integer within n_sample length. (n_sample 길이내에서 랜덤 정수 추출)
            random_num = np.random.choice(range(n_samples))

            # Get the coordinate(independent variable) from X (X로부터 좌표 추출)
            centroid = X[random_num]

            # Input the coordinate to centroids(list variable). (centroids 리스트 변수에 좌표 입력)
            centroids[i] = centroid

        # Compact version.
        # for i in range(self.k):
        #     centroids[i] = X[np.random.choice(range(n_samples))]

        return centroids

        
    def _create_cluster(self, X):
        
        # Get the number of sample from X (X로부터 샘플 개수 획득)
        n_sample = np.shape(X)[0]

        # Initialize the clusters(list variables) (리스트 변수인 클러스터 초기화)
        clusters = [[] for _ in range(self.k)]

        for idx, sample in enumerate(X):
            closet_centroid_idx = self._get_closet_centroid_idx(sample)
            clusters[closet_centroid_idx].append(idx)

        return clusters


    def _get_closet_centroid_idx(self, sample):

        # # Initialize closet centroid index (가장 가까운 중심점 번호 초기화)
        # closet_centroid_idx = 0

        # # Initialize the distance by infinite, not zero, to beautify the code and convenience
        # distance = float('inf')

        # # Check the which centroids is closet to sample
        # for idx, centroid in enumerate(centroids):

        #     # Get the euclidean distance (유클리디언 거리 계산)
        #     euclidean_distance = np.sqrt(np.sum((centroid - sample) ** 2))
 
        #     # Check the which distance is more closet to centroid (어떤 중심점이 가장 가까운 거리인지 검증)
        #     if euclidean_distance < distance:
        #         distance = euclidean_distance
        #         closet_centroids_idx = idx

        # Easy Version.
        distances = [np.sqrt(np.sum((centroid - sample) ** 2)) for centroid in self.centroids]
        closet_centroid_idx = np.argmin(distances)
            
        return closet_centroid_idx


    def _calc_new_centroid(self, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for idx, cluster in enumerate(self.clusters):
            cluster_mean = np.mean(X[cluster], axis=0)
            centroids[idx] = cluster_mean
        return centroids
        

    def _get_cluster_label(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred


    def predict(self, X):

        # Initialize the centroid (중심점 랜덤 초기화)
        self.centroids = self._init_random_centroids(X)

        # Optimize clusters 
        for _ in range(self.max_iters):
            self.clusters = self._create_cluster(X)

            if self.plot_steps:
                self.plot(X)

            prev_centroids = self.centroids
            self.centroids = self._calc_new_centroid(X)

            diff = self.centroids - prev_centroids
            if not diff.any():
                break

            if self.plot_steps:
                self.plot(X)

        return self._get_cluster_label(self.clusters, X)
    

    def plot(self, X):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = X[index].T
            ax.scatter(*point)
        
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()


def main():
    
    # It usally used to create virtual data on clustering (주로 클러스터링용 가상 데이터 생성에 사용)
    # center: centroids, n_features: independent variable
    X, y = make_blobs(centers=2, n_samples=10000, n_features=2, shuffle=True)

    K = Kmeans(k=2, max_iters=500, plot_steps=True)

    cluster_indexs = K.predict(X)
    print (cluster_indexs)

    K.plot(X)


if __name__ == "__main__":

    sys.exit(main())