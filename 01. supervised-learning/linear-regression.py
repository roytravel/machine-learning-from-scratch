#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


# In[83]:


class LinearRegression(object):
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate # 학습률
        self.n_iters = n_iters # Iteration
        self.weights = None # 가중치
        self.bias = None # 편향

    def fit(self, X, y):
        # (샘플 개수, 피처 개수)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent (경사 하강법)
        for _ in range(self.n_iters):
            # 예측 = 데이터 * 가중치 + 편향
            y_pred = np.dot(X, self.weights) + self.bias

            # compute gradients (기울기 계산)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update parameters (파라미터 업데이트)
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)


    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


# In[84]:


def r2_score(y_true, y_pred):
    """
    # 결정 계수 계산
    It represent the explainablity of regression model. (회귀 모형의 설명력 표현)
    0: Min explainability (설명력 낮음)
    1: Max explainability (설명력 높음)
    """
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def plot(X, X_train, y_train, X_test, y_test)


# In[102]:


def main():
    # 데이터셋 로드
    X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=2021)

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)

    # 회귀 모델 객체 생성
    LR = LinearRegression(learning_rate=0.001, n_iters=1000)

    # 회귀 모델 학습
    LR.fit(X_train, y_train)

    # 학습 예측 값 출력
    y_pred = LR.predict(X_test)

    # 평가 메트릭(mse, r2_score)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = r2_score(y_test, y_pred)
    print (f"{mse}")
    print (f"{accuracy}")

    y_pred_linear = LR.predict(X)

    cmap = plt.get_cmap('viridis')
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color=cmap(1.0), s=10)
    plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_linear, color="green", linewidth=3, label="prediction")
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.savefig('Linear Regression')
    plt.show()


# In[103]:


if __name__ == "__main__":
    main()


# In[ ]:




