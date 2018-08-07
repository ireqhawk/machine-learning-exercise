# 实现了多项式朴素贝叶斯,精度和sklearn接近
# 2018.08.07
import numpy as np
import matplotlib.pyplot as plot
import matplotlib as mpl
import sklearn.model_selection
import sklearn.datasets
from sklearn import tree
import time
from abc import ABCMeta, abstractmethod
from functools import reduce
from sklearn.naive_bayes import MultinomialNB
class NB():
    def __init__(self):
        self.pro_table = None

    @abstractmethod
    def fit(self,X,Y):
        pass

    def predict(self,X):
        predicts = []
        predict_num = X.shape[0]
        feature_num = X.shape[1]
        for i in range(predict_num):
            probas = []
            for y in range(self.pro_table.shape[0]):
                probas.append(reduce(lambda x,y:x*y,[self.pro_table[y,j,X[i,j]] for j in range(feature_num)]))
            predicts.append(np.argmax(probas))
        return predicts
class My_MultinomialNB(NB):
    def fit(self,X,Y,feature_max,class_num):
        feature_num = X.shape[1]
        train_num = X.shape[0]
        self.pro_table = np.zeros((class_num,feature_num + 1,feature_max))#shape1 加1用来存放P(y)
        for y in np.unique(Y):
            self.pro_table[y][feature_num][0] = (np.sum(Y == y) + 1) / (train_num + class_num)
            for feature in range(feature_num):
                self.pro_table[y,feature,:] = np.array([ (np.sum((Y == y) & (X[:,feature] == val)) + 1) / (train_num + class_num)             for val in range(feature_max)])

# class GaussianNB(NB):
#     def cal_pro(self):
#         return 2


x,y = sklearn.datasets.load_digits(return_X_y=True)
x = x.astype(np.int64)
y = y.astype(np.int64)

feature_max = np.max([np.unique(x[:,i]).shape[0] for i in range(x.shape[1])])
class_num = np.unique(y).shape[0]
x,xt,y,yt = sklearn.model_selection.train_test_split(x,y)

def show_info(clf,arr = True):
    predicts = []
    for i in range(xt.shape[0]):
        if arr:
            proba = clf.predict(xt[i].reshape(1,-1))
            predicts.append(proba[0])
        else:
            predicts.append(clf.predict(xt[i]))
        #print('is {} predict {}'.format(yt[i], predicts[i]))
    total = yt.shape[0]
    corrects = np.sum(predicts == yt)
    print('total = {} correct = {} accuracy = {}'.format(total, corrects, corrects / total))

multionmial_nb = My_MultinomialNB()
multionmial_nb.fit(x,y,feature_max,class_num)
show_info(multionmial_nb)

clf = MultinomialNB()
clf.fit(x,y)
show_info(clf)


