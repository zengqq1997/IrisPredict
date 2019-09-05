# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\GGboom\PycharmProjects\endwork\test.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import csv
import numpy as np
from PIL import Image
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt#约定俗成的写法plt
import matplotlib
import Linear
import knn
import random_forest

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(968, 803)
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(8)
        MainWindow.setFont(font)
        MainWindow.setWindowOpacity(2.0)

        self.modelclass=0;
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.data = QtWidgets.QLabel(self.centralwidget)
        self.data.setGeometry(QtCore.QRect(10, 70, 61, 261))
        self.data.setFrameShape(QtWidgets.QFrame.Box)
        self.data.setFrameShadow(QtWidgets.QFrame.Raised)
        self.data.setObjectName("data")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 360, 941, 16))
        self.line.setLineWidth(2)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.yuce = QtWidgets.QLabel(self.centralwidget)
        self.yuce.setGeometry(QtCore.QRect(10, 380, 61, 401))
        self.yuce.setFrameShape(QtWidgets.QFrame.Box)
        self.yuce.setFrameShadow(QtWidgets.QFrame.Raised)
        self.yuce.setObjectName("yuce")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(70, 70, 601, 261))
        self.scrollArea.setMinimumSize(QtCore.QSize(601, 261))
        self.scrollArea.setMaximumSize(QtCore.QSize(1000, 1000))
        self.scrollArea.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 578, 3000))
        self.scrollAreaWidgetContents.setMinimumSize(QtCore.QSize(500, 3000))
        self.scrollAreaWidgetContents.setMaximumSize(QtCore.QSize(601, 261))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.datacont = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.datacont.setGeometry(QtCore.QRect(0, 0, 601, 1000))
        self.datacont.setMinimumSize(QtCore.QSize(500, 10000))
        self.datacont.setText("")
        self.datacont.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignTop)
        self.datacont.setObjectName("datacont")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 20, 91, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(610, 20, 81, 28))
        self.pushButton_5.setObjectName("pushButton_5")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(750, 40, 171, 211))
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.regression = QtWidgets.QPushButton(self.groupBox)
        self.regression.setGeometry(QtCore.QRect(30, 20, 111, 41))
        self.regression.setFlat(False)
        self.regression.setObjectName("regression")
        self.knn = QtWidgets.QPushButton(self.groupBox)
        self.knn.setGeometry(QtCore.QRect(30, 80, 111, 41))
        self.knn.setObjectName("knn")
        self.randforest = QtWidgets.QPushButton(self.groupBox)
        self.randforest.setGeometry(QtCore.QRect(30, 140, 111, 41))
        self.randforest.setObjectName("randforest")
        self.rate = QtWidgets.QLabel(self.centralwidget)
        self.rate.setGeometry(QtCore.QRect(670, 490, 91, 41))
        self.rate.setObjectName("rate")
        self.accuracy = QtWidgets.QLabel(self.centralwidget)
        self.accuracy.setGeometry(QtCore.QRect(750, 480, 131, 61))
        self.accuracy.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.accuracy.setFrameShadow(QtWidgets.QFrame.Raised)
        self.accuracy.setObjectName("accuracy")
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(110, 20, 491, 31))
        self.label1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label1.setLineWidth(5)
        self.label1.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.label1.setObjectName("label1")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(720, 270, 231, 81))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(15, 25, 211, 41))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(70, 380, 531, 401))
        self.label_2.setLineWidth(3)
        self.label_2.setText("")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(70, 380, 531, 401))
        self.textEdit.setObjectName("textEdit")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(680, 580, 201, 71))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(650, 460, 261, 91))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(650, 390, 111, 51))
        self.label_4.setObjectName("label_4")
        self.model = QtWidgets.QLabel(self.centralwidget)
        self.model.setGeometry(QtCore.QRect(750, 390, 151, 51))
        self.model.setFrameShape(QtWidgets.QFrame.Box)
        self.model.setFrameShadow(QtWidgets.QFrame.Raised)
        self.model.setText("")
        self.model.setObjectName("model")
        self.textEdit.raise_()
        self.label_3.raise_()
        self.label_2.raise_()
        self.data.raise_()
        self.line.raise_()
        self.yuce.raise_()
        self.scrollArea.raise_()
        self.pushButton.raise_()
        self.pushButton_5.raise_()
        self.groupBox.raise_()
        self.rate.raise_()
        self.accuracy.raise_()
        self.groupBox_2.raise_()
        self.pushButton_2.raise_()
        self.label1.raise_()
        self.label_4.raise_()
        self.model.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.actiondaoru = QtWidgets.QAction(MainWindow)
        self.actiondaoru.setObjectName("actiondaoru")

        self.retranslateUi(MainWindow)
        self.pushButton_5.clicked.connect(self.choose)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.read)
        self.pushButton_2.clicked.connect(self.showpicture)
        self.regression.clicked.connect(self.linearmodel)
        self.knn.clicked.connect(self.knnmodel)
        self.randforest.clicked.connect(self.randforestmodel)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "莺尾花的种类智能预测系统"))
        self.data.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">数据：</span></p><p><span style=\" font-weight:600;\"><br/></span></p></body></html>"))
        self.yuce.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">预测</span></p><p><span style=\" font-size:12pt;\">结果 ：</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "读取数据"))
        self.pushButton_5.setText(_translate("MainWindow", "选择文件"))
        self.groupBox.setTitle(_translate("MainWindow", "模型"))
        self.regression.setText(_translate("MainWindow", "多元回归分析"))
        self.knn.setText(_translate("MainWindow", "KNN"))
        self.randforest.setText(_translate("MainWindow", "随机森林"))
        self.rate.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">准确率：</span></p></body></html>"))
        self.accuracy.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.label1.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.groupBox_2.setTitle(_translate("MainWindow", "状态栏"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.pushButton_2.setText(_translate("MainWindow", "预测分析"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">当前模型：</span></p></body></html>"))
        self.actiondaoru.setText(_translate("MainWindow", "daoru"))

    def choose(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self, "选取文件", "C:/Users/GGboom/PycharmProjects/endwork",
                                                          "All Files (*);;Text Files (*.csv)")  # 设置文件扩展名过滤,注意用双分号间隔
        print(fileName1, filetype)
        self.label1.setText(fileName1)
        self.path = fileName1

    def read(self):
        data = []
        ndata = ""
        print(self.path)
        data = np.loadtxt(self.path, delimiter=",", skiprows=1, dtype=np.str)  # 用numpy读取数据
        data = data[0:150]
        with open(self.path, "r") as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            h = next(csv_reader)  # 读取第一行每一列的标题
            title = "               Number" + "         " + "sepal_length" + "        " + h[1] + "        " + h[
                2] + "        " + h[3] + "         " + h[4] + "\n"
        w, e = data.shape
        number = np.arange(1, 1 + w)
        col = number.reshape(-1, 1)
        # for i in range(0,w):
        #     col[i]=str(col[i]).zfill(3)
        data = np.hstack((col, data))
        print(data)
        w, e = data.shape

        for i in range(0, w):  # 将csv 文件中的数据保存到birth_data中
            for j in range(0, e):
                ndata = ndata + "                     " + str(data[i][j]).zfill(3)
                if j % 5 == 0 and j != 0:
                    ndata = ndata + "\n"
        all = title + ndata
        self.datacont.setText(all)

    def showpicture(self):
        _translate = QtCore.QCoreApplication.translate
        image2 = Image.open('test.jpg')
        pix = QPixmap('test.jpg')
        pixmap = QtGui.QPixmap(pix)
        self.label_2.setPixmap(pixmap)
        self.label_2.setScaledContents(True)

        if self.modelclass==1:
             rate = Linear.linearaccuracy
        elif self.modelclass == 2:
            rate = knn.linearaccuracy
        elif self.modelclass==3:
            rate = random_forest.linearaccuracy

        rate = str(round(100 * rate, 2)) + "%"
        self.accuracy.setText(_translate("MainWindow",
                                         "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">" + rate + "</span></p></body></html>"))

    def draw(self):
        X = np.linspace(-np.pi, np.pi, 256, endpoint=True)  # -π to+π的256个值
        C, S = np.cos(X), np.sin(X)
        plt.plot(X, C)
        plt.plot(X, S)
        # 在ipython的交互环境中需要这句话才能显示出来
        plt.savefig('test.jpg')
        plt.show()

    def state(self):
        self.label.setText("已经读取数据")

    def linearmodel(self):
        _translate = QtCore.QCoreApplication.translate
        Linear.Linear(self.path)
        self.modelclass=1
        self.label.setText(_translate("MainWindow",
                                      "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">" + "  多元回归分析模型训练完毕!" + "</span></p></body></html>"))
        self.model.setText(_translate("MainWindow",
                                         "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">" + "多元回归分析模型 "+ "</span></p></body></html>"))

    def knnmodel(self):
        _translate = QtCore.QCoreApplication.translate
        knn.knn(self.path)
        self.modelclass = 2
        self.label.setText(_translate("MainWindow",
                                      "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">" + "  KNN模型训练完毕!" + "</span></p></body></html>"))
        self.model.setText(_translate("MainWindow",
                                      "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">" + "KNN模型 " + "</span></p></body></html>"))
    def randforestmodel(self):
        _translate = QtCore.QCoreApplication.translate
        random_forest.random_forest(self.path)
        self.modelclass =3
        self.label.setText(_translate("MainWindow",
                                      "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">" + "  随机森林模型训练完毕!" + "</span></p></body></html>"))
        self.model.setText(_translate("MainWindow",
                                      "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">" + "随机森林模型 " + "</span></p></body></html>"))