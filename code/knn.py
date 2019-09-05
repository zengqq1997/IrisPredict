#first read in the data
import pandas as pd
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import numpy as np

linearaccuracy=0

def feature_label_split(pd_data):
    # 行数、列数
    row_cnt, column_cnt = pd_data.shape
    # 生成新的X、Y矩阵
    X = np.empty([row_cnt, column_cnt - 1])  # 生成两个随机未初始化的矩阵
    Y = np.empty([row_cnt, 1])
    for i in range(0, row_cnt):
        row_array = pd_data.iloc[i,]
        X[i] = np.array(row_array[0:-1])
        Y[i] = np.array(row_array[-1])
    return X, Y
# 把特征数据进行标准化为均匀分布
def uniform_norm(X_in):
    X_max = X_in.max(axis=0)
    X_min = X_in.min(axis=0)
    X = (X_in - X_min) / (X_max - X_min)
    return X
def knn(path):
    global linearaccuracy
    # the parameter is the directory you store the winequality-red.csv file, make sure use delimiter ;
    wine = pd.read_csv(path)

    # second get labels and features
    features, wine_labels = feature_label_split(wine)  # the correct label of red wine 1599 labels
    # wine_features= wine.drop('class', axis=1) #the features of each wine 1599 rows by 11 columns
    wine_features = uniform_norm(features)

    # split data into training and testing data
    test_size = 0.20  # testing size propotional to wht whole size
    seed = 0  # random number, whatever you like
    features_train, x_test, labels_train, y_test = model_selection.train_test_split(wine_features, wine_labels,
                                                                                    test_size=test_size,
                                                                                    random_state=seed)

    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(features_train,
                                                                                    labels_train, test_size=test_size,
                                                                                    random_state=seed)
    print(y_train)
    # subsets for training models
    # subsets for validation
    # Fit the KNN Model
    k_range = range(1, 100, 5)  #
    KNN_k_error = []
    for k_value in k_range:
        clf = KNeighborsClassifier(n_neighbors=k_value)
        clf.fit(x_train, y_train)
        error = 1. - clf.score(x_validation, y_validation)
        KNN_k_error.append(error)
    # plt.figure(figsize=(10,5))
    # plt.plot(k_range, KNN_k_error)
    # plt.title('auto KNN')
    # plt.xlabel('k values')
    # plt.ylabel('error')
    # plt.xticks(k_range)
    # plt.show()

    algorithm_types = ['ball_tree', 'kd_tree', 'brute']
    KNN_algorithm_error = []
    for algorithm_value in algorithm_types:
        clf = KNeighborsClassifier(algorithm=algorithm_value, n_neighbors=1)
        clf.fit(x_train, y_train)
        error = 1. - clf.score(x_validation, y_validation)
        KNN_algorithm_error.append(error)
    # plt.plot(algorithm_types, KNN_algorithm_error)
    # plt.title('KNN algorithm')
    # plt.xlabel('algorithm')
    # plt.ylabel('error')
    # plt.xticks(algorithm_types)
    # plt.show()

    index_algorithm = 0
    minmum_algorithm = max(KNN_algorithm_error)
    for i, value in enumerate(KNN_algorithm_error):
        if value <= minmum_algorithm:
            minmum_algorithm = value
            index_algorithm = i

    index_k = 0
    minmum_k = max(KNN_k_error)
    for i, value in enumerate(KNN_k_error):
        if value <= minmum_k:
            minmum_k = value
            index_k = i

    ## step 4 Select the best model and apply it over the testing subset
    best_algorithm = algorithm_types[index_algorithm]
    best_k = k_range[index_k]  # poly had many that were the "best"
    model = KNeighborsClassifier(algorithm=best_algorithm, n_neighbors=best_k)
    model.fit(X=x_train, y=y_train)
    score = model.score(x_test, y_test)
    test_pred = model.predict(x_test)

    l = []
    for m in range(y_test.shape[0]):
        for i in y_test[m]:
            l.append(i)
    sum = 0
    for j in range(y_test.shape[0]):
        if (l[j] == test_pred[j]):
            sum = sum + 1
    score = sum / y_test.shape[0]
    linearaccuracy=score
    print("正确率为 %.6f" % score)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(l, lw=2.5, label="真实值", color='blue')
    plt.plot(test_pred, lw=2.5, label="预测值", color='red')
    plt.legend(loc="best")
    plt.plot(l, 'r*')
    plt.plot(test_pred, 'ro', color='yellow')
    plt.ylabel("class")
    plt.xlabel("num")
    plt.title("predict result and true data")
    plt.savefig('test.jpg')













