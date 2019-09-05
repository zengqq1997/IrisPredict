#coding:UTF-8
import numpy as np
import csv
import math
#加载数据
#从给定的csv文件加载数据

data = []
with open(r"C:/Users/ZQQ/Desktop/advanced/study/informationAF/classtest/data/winequality-white.csv") as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    h = next(csv_reader)  # 读取第一行每一列的标题
    print(h)
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        data.append(row[1:])

X = [[float(x) for x in row] for row in data]  # 将字符串转成float类型
X = np.array(X)

w, h = X.shape
print("w*h=", w, "*", h)
print("原始数据：")
print(X)

std_2 = np.zeros((h))
# 1.归一化

t = np.zeros((w, h), dtype=float)
for i in range(0, h):
    max1 = max(X[:, i])
    min1 = min(X[:, i])
    for j in range(0, w):
        t[j, i] = (X[j, i] - min1) / (max1 - min1)
        # t[j,i] = math.log(X[j,i])/math.log(max1)
print("归一化后的数据：")
print(t)

# 2.计算协方差

r = np.zeros((h, h), dtype=float)
for i in range(0, h):
    for j in range(0, h):
        avg_xi = np.mean(t[:, i])  # 第i列
        avg_xj = np.mean(t[:, j])  # 第j列
        num1 = np.dot(t[:, i] - avg_xi, t[:, j] - avg_xj)  # 两列矩阵相乘
        num2pf = np.dot(t[:, i] - avg_xi, t[:, i] - avg_xi)
        num3pf = np.dot(t[:, j] - avg_xj, t[:, j] - avg_xj)
        r[i, j] = num1 / math.sqrt(num2pf * num3pf)
print("相关系数矩阵：")
print(r)

# 3.计算机特征值和特征向量
eigenvalue, featurevector = np.linalg.eig(r)
print("原始矩阵的特征值：")
print("eigenvalue=", eigenvalue)
print("原始矩阵的特征向量：")
print("featurevector=", featurevector)

# 4.计算主成分贡献率

indexs = np.argsort(eigenvalue)[::-1]
print("特征向量从小到大排序后原来的索引：")
print(indexs)
eig_values = eigenvalue[indexs]
eig_vectors = featurevector[:, indexs]

explained_rate = eig_values / sum(eig_values)  # 计算贡献率
explained_rate_sum = np.cumsum(explained_rate)  # 累计计算贡献率
print("贡献率")
print(explained_rate)
print("累计计算贡献率")
print(explained_rate_sum)
# 5.确定主成分
count = 0
for i in range(0, h):
    if (explained_rate_sum[i] >= 0.95):
        count = i + 1
        break
# 因为累计贡献率越来越大，所以是递增的，所以出现一个累计大于0.95的加一
Q = np.zeros((count, h), dtype=float)
for i in range(0, count):
    Q[i] = eig_vectors[i]
P = Q.T  # 转置
print("主成分特征向量矩阵：")
print(P)

Y = np.dot(X, P)
print("最终降维数据Y：")
print(Y)





