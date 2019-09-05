import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



def iris_type(s):+
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    #mpl.rcParams['font.sans-serif'] = [u'SimHei']  
    #mpl.rcParams['axes.unicode_minus'] = False

    path = 'iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x_prime, y = np.split(data, (4,), axis=1)
    unif_trainX, unif_testX, train_Y, test_Y = train_test_split(x_prime, y, test_size=0.2, random_state=0)

    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]].
    plt.figure(figsize=(10, 9), facecolor='#FFFFFF')
    # 准备数据
    # x = x_prime[:, pa]
    #print(pair)
    #print (x)

    # 决策树学习
    clf = DecisionTreeClassifier()
    dt_clf = clf.fit(unif_trainX, train_Y)

       

  
    y_hat = dt_clf.predict(unif_testX)
    print(y_hat)
    y= test_Y.reshape(-1)
    print(y)
    c = np.count_nonzero(y_hat == y)    # 统计预测正确的个数
    #print('特征：  ', iris_feature[pair[0]], ' + ', iris_feature[pair[1]])
    print('\t预测正确数目：', c)
    print('\t准确率: %.2f%%' % (100 * float(c) / float(len(y))))
    c00 = np.count_nonzero(y_hat == 1)
    c11 = np.count_nonzero(y_hat == 2)    
    c22 = np.count_nonzero(y_hat == 3)
    c0 = np.count_nonzero(y == 1)
    c1 = np.count_nonzero(y == 2)
    c2 = np.count_nonzero(y == 3)

