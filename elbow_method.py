# -*- coding: utf-8 -*-
# test in python27

from numpy import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def loadDataMatrix(fileName):
    dataSet = []
    f = open(fileName)
    for line in f.readlines():
        curLine = line.strip().split(',')   # 这里","表示以文件中数据之间的分隔符","分割字符串
        row = []
        for item in curLine:
            row.append(float(item))
        dataSet.append(row)

    return mat(dataSet)

if __name__ == "__main__":
    X = loadDataMatrix('./source_data/1.data').getA()
    K = range(1, 10)

    meandistortions = []
    for k in K:
        kmeans = KMeans(k, 'k-means++')
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    plt.plot(K, meandistortions, 'bx-')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters K')
    plt.ylabel('Average Within-Cluster distance to Centroid')

    imgName = './'+str(int(time.time())) + '_elbow.png'
    #plt.savefig(imgName)
    plt.show()
