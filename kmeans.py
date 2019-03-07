# -*- coding: utf-8 -*-
# create by k1ic at 20190306
# 注：KMeans 会自动使用cpu多核，该脚本只每行只有两列的数据（正好对应二维坐标系x、y轴）
# test in python27

from sklearn.cluster import KMeans
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import time
import decimal

#fix error:RuntimeError: Invalid DISPLAY variable
plt.switch_backend('agg')

#防止数据过长时打印结果有省略
np.set_printoptions(threshold=np.inf)

def loadDataSet(fileName):
    dataSet = []
    f = open(fileName)
    for line in f.readlines():
        curLine = line.strip().split(',')   # 这里","表示以文件中数据之间的分隔符","分割字符串
        row = []
        for item in curLine:
            row.append(float(item))
        dataSet.append(row)

    return np.array(dataSet)

#初始化数据集
data=loadDataSet("./source_data/OpenApp_SendGiftSum.data")

#指定簇的个数，即分成几类
cluster_num = 4
km=KMeans(n_clusters=cluster_num).fit(data)

#标签结果
rs_labels=km.labels_

#每个类别的中心点
rs_center_ids=km.cluster_centers_

# 用 [[]] * cluster_num 方式创建的list，append元素后是添加了整行，不合预期
# 初始化存储各簇信息的list
clusters = [[] for i in range(cluster_num)]

#按数据标签将数据放入各簇
for k,v in enumerate(rs_labels):
    clusters[v].append(data[k].tolist())

#用于按各簇元素个数正序排列，需耗费1/10的时间
#clusters.sort(key = len)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
markers = ['^', 'o', '*', 's', 'd', 'h']

for i in range(cluster_num):
    #画各簇数据散点图
    plt.scatter(np.array(clusters[i])[:,0], np.array(clusters[i])[:,1], s=30, c=colors[i], marker=markers[i], alpha=0.8, label='Cluster_'+str(i)+' '+str(len(clusters[i])))
    #将各簇数据写入文件
    fileName = './target_data_1/OpenApp_SendGiftSum_cluster'+str(i)+"_total_"+str(len(clusters[i]))+".csv"
    f = open(fileName, 'a')

    for item in clusters[i]:
        row = str(int(item[0]))+','+str(decimal.Decimal("%.4f" % float(item[1])))+"\n"
        f.write(row)

    f.close()

#指定标签的显示位置
plt.legend(loc='upper right')

# 画各簇质心点
plt.scatter(rs_center_ids[:,0], rs_center_ids[:,1], s=200, c='black', marker='+', alpha=1)

plt.title('Open-App Send-Gift-Sum Distribution')
plt.xlabel('X_Label Open-App-Times')
plt.ylabel('Y_Label Send-Gift-Sum')

imgName = './target_data_1/'+str(int(time.time())) + '_OpenApp_SendGiftSum.png'
plt.savefig(imgName)
#plt.show()
