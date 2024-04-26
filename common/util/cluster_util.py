from make_nested_cluster import generate_and_plot_clusters, draw_graph
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def test1(guess):
    
    # import cv2
    x, labels = generate_and_plot_clusters(n_big_centers=guess)
    N = len(x)
    k_min = 1
    k_max = int(np.sqrt(N + 1))
    MD = []
    k_x = []
    k_labels_ = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(k)
        kmeans.fit(x)
        if k == guess * 3:
            k_labels_ = kmeans.labels_
        SSE = kmeans.inertia_
        MD.append(SSE / N)
        if  k== guess or k==guess*3 or k == 8:
            draw_graph(x, kmeans.labels_, kmeans.cluster_centers_)
    n = []

    # from yellowbrick.cluster import KElbowVisualizer
    # visualizer = KElbowVisualizer(KMeans(), k=(k_min, k_max))
    # visualizer.fit(x)
    # visualizer.show()
    def calculate_euclidean_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


    for i in range(0, k_max):
        ni = (MD[i] - min(MD)) * 10 / (max(MD) - min(MD))
        n.append(ni)


    P = []
    for i in range(0, k_max):
        P.append((n[i], i + 1))

    min_alpha = np.pi
    opt_k = 0
    alpha_list = []
    opt_k_list = []
    gradient_list = []
    threshold = 0.5
    for i in range(0, k_max - k_min - 1):
        j = i + 1
        k = i + 2

        a = calculate_euclidean_distance(P[i], P[j])
        b = calculate_euclidean_distance(P[j], P[k])
        c = calculate_euclidean_distance(P[k], P[i])
        theta = (np.power(a, 2) + np.power(b, 2) - np.power(c, 2)) / (2 * a * b)
        alpha = np.arccos(theta)
        alpha_list.append(alpha)
    for i in range(0, len(alpha_list) - 1):
        if alpha_list[i + 1] - alpha_list[i] > np.std(alpha_list)/10 :
            opt_k_list.append((alpha_list[i + 1] - alpha_list[i], i + 2))


    remove_list=[]
    for i in range(0, len(opt_k_list)-1):
        if (opt_k_list[i+1][1]-opt_k_list[i][1])==1:
            remove_list.append(opt_k_list[i])
    for remove in remove_list:
        opt_k_list.remove(remove)

    plt.plot([x for x in range(k_min, k_max - 1)], alpha_list)
    plt.xticks([x for x in range(k_min, k_max - 1)])
    plt.axvline(guess, 0, 1, color='gray', linestyle='--', linewidth=1)
    plt.axvline(guess * 3, 0, 1, color='gray', linestyle='--', linewidth=1)
    plt.show()
    
# test1(1)

class KmeansCluster():    
    def __init__(self, X):    #default number =4 . k 는 클러스터 개수인데 사실 안씀
        self.X = X
        self.opt_k=[]
        self.k_min=1
        self.k_max=int(np.sqrt(len(self.X)+1))
        self.alpha_list=[]
    def calculate_euclidean_distance(self ,point1, point2): #유클리드 거리 계산
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    def return_opt_k(self) -> list : #optimal k list형태로 리턴
        N = len(self.X)
        k_min = 1
        k_max = int(np.sqrt(N + 1))

        MD = []
        for k in range(k_min, k_max + 1):   #MD는 mean distortion
            kmeans = KMeans(k)
            kmeans.fit(self.X)
            SSE = kmeans.inertia_
            MD.append(SSE / N)

        n = []
        for i in range(0, k_max): #n 은 정규화된 MD
            ni = (MD[i] - min(MD)) * 10 / (max(MD) - min(MD))
            n.append(ni)


        P = []
        for i in range(0, k_max):
            P.append((n[i], i + 1))

        alpha_list = []
        opt_k_list = []
        for i in range(0, k_max - k_min - 1): # alpha는 n의 그래프를 그렸을 때, 특정 지점에서의 각도.
            j = i + 1
            k = i + 2

            a = self.calculate_euclidean_distance(P[i], P[j])
            b = self.calculate_euclidean_distance(P[j], P[k])
            c = self.calculate_euclidean_distance(P[k], P[i])

            theta = (np.power(a, 2) + np.power(b, 2) - np.power(c, 2)) / (2 * a * b)
            alpha = np.arccos(theta)
            alpha_list.append(alpha)
        
        for i in range(0, len(alpha_list) - 1):    #경험적 관찰로, alpha 표준편차의 /10 정도 차이 사용
            if alpha_list[i + 1] - alpha_list[i] > np.std(alpha_list)/10 :
                opt_k_list.append((alpha_list[i + 1] - alpha_list[i], i + 2))

        remove_list=[]
        for i in range(0, len(opt_k_list)-1):
            if (opt_k_list[i+1][1]-opt_k_list[i][1])==1:
                remove_list.append(opt_k_list[i])

        for remove in remove_list:
            opt_k_list.remove(remove)
        self.alpha_list= alpha_list
        self.opt_k=opt_k_list
        return [d[1] for d in opt_k_list]
    

    def draw_alpha(self):
        plt.plot([x for x in range(self.k_min, self.k_max - 1)], self.alpha_list)
        plt.xticks([x for x in range(self.k_min, self.k_max - 1)])
        for i in self.opt_k:
            plt.axvline(i[1], 0, 1, color='gray', linestyle='--', linewidth=1)
        plt.show()




if __name__ == '__main__':
    # generate_and_plot_clusters()
    # x, labels = generate_and_plot_clusters(n_big_centers=1)
    # kmeans =KmeansCluster(X=x)
    
    # result = kmeans.return_opt_k()
    # kmeans.draw_alpha()
    # print(result)
    pass