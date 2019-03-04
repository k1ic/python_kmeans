# -*- coding: utf-8 -*-
# test in python27

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import sys
from numpy import *
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
    print('用法: python ' + __file__ + ' [待分析的数据文件（列间用逗号分割）] [簇的最大值（从1到最大值之间寻找合适的簇数量值，默认值：12）]')

    if len(sys.argv) == 1:
       exit('Param error, need file to analysis!')
    else:
	dataFilePath = sys.argv[1]

    if len(sys.argv) == 3 and int(sys.argv[2]) > 1:
        kMax = int(sys.argv[2])
    else:
	kMax = 12

    X = loadDataMatrix(dataFilePath).getA()
    K = range(1, kMax)

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
