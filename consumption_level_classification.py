# -*- coding: utf-8 -*-
# run in python34
import numpy as np
import io
from sklearn.cluster import KMeans

def loadData(filePath):
    """
    读取文件数据并返回消费数据和对应省份名称
    :param filePath: 数据文件路径
    :return: 各省消费数据,省份名称
    """
    file = io.open(filePath, 'r+', encoding='utf-8')  # 注意读文件的编码
    lines = file.readlines()
    fileData = []
    fileCityName = []
    for line in lines:
        items = line.strip().split(',')
        fileCityName.append(items[0])
        fileData.append([float(items[i]) for i in range(1, len(items))])
    file.close()
    return fileData, fileCityName

def saveData(filePath, data):
    """
    保存输出结果到指定路径下
    :param filePath: 保存结果的目的文件路径
    :param data: 结果数据
    :return:
    """
    file = io.open(filePath, 'w+',encoding='utf-8') # 注意编码
    file.write(str(data))
    file.close()


data, cityName = loadData('data.csv')
km = KMeans(n_clusters=3) # 将省份分3类
label = km.fit_predict(data) # 获取各省份所属的类编号
avgExpenses = np.average(km.cluster_centers_, axis=1)  # axis 1按行 2按列 求平均

# 根据label将相同分类省份名放置一起
CityCluster = [[], [], []]
for i in range(len(cityName)):
    CityCluster[label[i]].append(cityName[i])

resultStr = '' # 保存分类结果
# 输出分类结果
for i in range(len(CityCluster)):
    print("平均消费%0.2f" % (avgExpenses[i]))
    print(CityCluster[i])
    # 将同分类省份用,拼接
    resultStr = resultStr + ','.join(CityCluster[i]) + '\n'

# 保存分类结果
saveData('result.csv',resultStr)
