# -*- coding: utf-8 -*-
# test in python27

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import time

def loadDataSet(fileName):
    dataSet = []
    f = open(fileName)
    for line in f.readlines():
        curLine = line.strip().split(',')   # 这里","表示以文件中数据之间的分隔符","分割字符串
        row = []
        for item in curLine:
            row.append(float(item))
        dataSet.append(row)

    return mat(dataSet)

# 求向量距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 选前k个点作为初始质心
def initCent(dataSet, k):
    data = []
    for i in range(k):
        data.append(dataSet[i].tolist())
    a = array(data)
    centroids = mat(a)
    return centroids

# K均值聚类算法实现
def KMeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0] #数据集的行
    clusterAssment = mat(zeros((m, 2)))
    centroids = initCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m): #遍历数据集中的每一行数据
            minDist = inf
            minIndex = -1
            for j in range(k): #寻找最近质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist: #更新最小距离和质心下标
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2 #记录最小距离质心下标，最小距离的平方
        for cent in range(k): #更新质心位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] #获得距离同一个质心最近的所有点的下标，即同一簇的坐标
            centroids[cent,:] = mean(ptsInClust, axis=0) #求同一簇的坐标平均值，axis=0表示按列求均值

    return centroids, clusterAssment

# 取数据的前两维特征作为该条数据的x , y 坐标，
def getXY(dataSet):
    import numpy as np
    m = shape(dataSet)[0]  # 数据集的行
    X = []
    Y = []
    for i in range(m):
        X.append(dataSet[i,0])
        Y.append(dataSet[i,1])
    return np.array(X), np.array(Y)

# 数据可视化
def showCluster(dataSet, k, clusterAssment, centroids):
    fig = plt.figure() #创建一个图形实例
    plt.title("K-means Send Gift Times Sum Distribution")

    my_y_ticks = np.arange(0, 250000, 30000) #设置y轴初始值、最大值、刻度步长
    plt.yticks(my_y_ticks)

    #参数含义：作为单个整数编码的子绘图网格参数。例如，“111”表示“1×1网格，第一子图”，“234”表示“2×3网格，第四子图”
    ax = fig.add_subplot(111)

    data = []
    counter = []

    for cent in range(k): #提取出每个簇的数据
        ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] #获得属于cent簇的数据
        data.append(ptsInClust)
        counter.append(len(ptsInClust))

    for cent, c, marker in zip( range(k), ['r', 'g', 'b', 'y'], ['^', 'o', '*', 's'] ): #画出数据点散点图
        X, Y = getXY(data[cent])

        #s:点的大小，c:color，marker:点的形状, alpha:点的亮度，label:标签
        ax.scatter(X, Y, s=30, c=c, marker=marker, alpha=0.8, label=counter[cent])

    centroidsX, centroidsY = getXY(centroids)
    ax.scatter(centroidsX, centroidsY, s=200, c='black', marker='+', alpha=1)  # 画出质心点
    ax.set_xlabel('X Label Send Gift Times')
    ax.set_ylabel('Y Label Send Gift Coin Sum')

    #这个必须有，否则label不显示
    plt.legend(loc='upper right')

    imgName = str(int(time.time())) + '_k-means_GT_GS_k4_divided_100_all.png'
    plt.savefig(imgName)

    plt.show()

if __name__ == "__main__":
    cluster_Num = 4
    #data = loadDataSet("20190221_20190227_send_coin_total_sum.csv")
    data = loadDataSet("20190221_20190227_send_coin_total_sum_divided_100.csv")
    centroids, clusterAssment = KMeans(data, cluster_Num)
    showCluster(data, cluster_Num, clusterAssment, centroids)
