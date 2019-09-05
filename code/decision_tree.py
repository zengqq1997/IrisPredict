#first read in the data
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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

#the parameter is the directory you store the winequality-red.csv file, make sure use delimiter ;
wine = pd.read_csv(r"C:\Users\ZQQ\Desktop\advanced\study\informationAF\classtest\data\flower.csv")


 # second get labels and features
wine_features, wine_labels = feature_label_split(wine)  # the correct label of red wine 1599 labels
print(wine_labels)
# wine_features= wine.drop('class', axis=1) #the features of each wine 1599 rows by 11 columns
print(wine_features)
#split data into training and testing data
test_size = 0.20 #testing size propotional to wht whole size
seed = 7 #random number, whatever you like
features_train, x_test, labels_train, y_test = model_selection.train_test_split(wine_features, wine_labels,
                                                                                test_size=test_size, random_state=seed)

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(features_train, labels_train,
                                                                                test_size=test_size, random_state=seed)

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
score = decision_tree.score(x_validation, y_validation)
print (score)


    
  
    
max_depth = [int(x) for x in range(1,200,10)]
max_depth_error = []
previous = 1
best_depth = 0
for depth in max_depth:
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    if previous > error:
        previous = error
        best_depth = depth
    max_depth_error.append(error)

plt.plot(max_depth, max_depth_error)
plt.title('RF max depth')
plt.xlabel('max depth')
plt.ylabel('error')
plt.xticks(max_depth)
plt.show()
print(best_depth)

min_samples_split = [2,5,10,25,50,70,90,100]
min_samples_error = []
previous = 1
best_num = 0
for num in min_samples_split:
    clf = tree.DecisionTreeClassifier(min_samples_split=num)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    if previous > error:
        previous = error
        best_num = num
    min_samples_error.append(error)

plt.plot(min_samples_split, min_samples_error)
plt.title('RF min samples')
plt.xlabel('min samples')
plt.ylabel('error')
plt.xticks(min_samples_split)
plt.show() 
print(best_num)

# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,4,10,20,50,70,90,100]
min_leaf_error = []
previous = 1
best_leaf = 0
for num in min_samples_leaf:
    clf = tree.DecisionTreeClassifier(min_samples_leaf=num)
    clf.fit(x_train,y_train)
    error = 1. - clf.score(x_validation, y_validation)
    if previous > error:
        previous = error
        best_leaf = num
    min_leaf_error.append(error)

plt.plot(min_samples_leaf, min_leaf_error)
plt.title('RF min samples in leaf')
plt.xlabel('min samples')
plt.ylabel('error')
plt.xticks(min_samples_leaf)
plt.show()
print(best_leaf)

decision_tree = tree.DecisionTreeClassifier(min_samples_leaf=best_leaf,min_samples_split=best_num,max_depth=best_depth)
decision_tree.fit(x_train, y_train)
score = decision_tree.score(x_test,y_test)
test_pred = decision_tree.predict(x_test)
l=[]
for m in range(y_test.shape[0]):
        for i in y_test[m]:
            l.append(i)
print("正确率为 %.6f" % score)
print(l)
print(test_pred)
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
plt.show()


