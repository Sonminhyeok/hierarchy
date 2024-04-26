# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time
# np.random.seed(int(time.time()))
np.random.seed(42)

def generate_and_plot_clusters(n_big_centers=1,  # 큰 중심점 수
                               num_samples=100,  # Sample 수
                               cluster_std=0.5):  # Cluster 표준 편차 조정 용도
    # 큰 중심점 생성
    big_centers_x, big_centers_y, big_centers = make_blobs(n_samples=n_big_centers,
                                                           n_features=3,
                                                           centers=n_big_centers,
                                                           center_box=(-10, 10),
                                                           return_centers=True)

    # 작은 중심점 생성
    small_centers = []
    fixed_distances = [2.5, 2.5, 2.5]  # 큰 중심점에서 작은 중심점까지 거리 조정용

    for big_center in big_centers:
        for i in range(len(fixed_distances)):
            # 각 축 방향으로 고정된 거리로 작은 중심점 생성
            random_direction = np.zeros(3)
            random_direction[i] = fixed_distances[i]

            # 큰 중심점에 작은 중심점까지의 거리를 일정하게 유지하면서 위치 조정
            small_center = big_center + random_direction
            small_centers.append(small_center)

    # 작은 중심점을 배열로 변환
    small_centers = np.vstack(small_centers)
    # 작은 중심점 기반의 Cluster 생성
    X, labels = make_blobs(n_samples=num_samples * n_big_centers * 3,
                           centers=small_centers,
                           cluster_std=cluster_std)

    # Data 시각화
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111,
    #                      projection='3d')

    # # 각 Cluster에 대해 색상 할당
    # num_clusters = len(np.unique(labels))
    # cmap = plt.get_cmap('tab10')
    # colors = cmap(np.linspace(0, 1, num_clusters))

    # for label in np.unique(labels):
    #     ax.scatter(X[labels == label, 0], X[labels == label, 1], X[labels == label, 2],
    #                c=colors[label],
    #                marker='o',
    #                label=f'Cluster {label}')

    # # 중심점 표시
    # ax.scatter(small_centers[:, 0], small_centers[:, 1], small_centers[:, 2],
    #            c='k',
    #            marker='x',
    #            s=100,
    #            label='Centers')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.title('Nested Cluster')
    # # plt.legend()
    # plt.savefig("2.png")
    # plt.show()

    return X, labels

def draw_graph(X, labels, centers):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 각 Cluster에 대해 색상 할당
    num_clusters = len(np.unique(labels))
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, num_clusters))

    for label in np.unique(labels):
        ax.scatter(X[labels == label, 0], X[labels == label, 1], X[labels == label, 2],
                   c=colors[label],
                   marker='o',
                   label=f'Cluster {label}')

    # 중심점 표시
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
               c='black', marker='x', s=200, label='Centroids')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title(f'Nested Cluster{len(np.unique(labels))}')
    # plt.legend()
    plt.savefig("2.png")
    plt.show()


if __name__ == '__main__':
    pass
    # x, labels = generate_and_plot_clusters(n_big_centers=2,num_samples=10)
    # print(x.shape,labels.shape)