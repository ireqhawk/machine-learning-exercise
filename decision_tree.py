# 2018.08.07 参考sklearn的方式实现了CART-GINI决策树
# 精度与Sklearn相近,耗时约为100倍,待优化
#2018.08.07 添加随机森林,精度同样与sklearn接近

import numpy as np
import matplotlib.pyplot as plot
import matplotlib as mpl
import sklearn.model_selection
import sklearn.datasets
from sklearn import tree
import time
from sklearn.ensemble import RandomForestClassifier

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
class Node:
    def __init__(self,judge_val,judge_result,childs,judge_index):
        self.judge_val = judge_val
        self.judge_index = judge_index
        self.childs = childs
        self.judge_result = judge_result

class Decision_tree:
    def __init__(self):
        self.root = None

    def cal_gini(self,Y):#计算基尼指数
        unique_Y = np.unique(Y)
        gini = 1
        for y in unique_Y:
            py = np.sum(Y == y) / len(Y)
            gini = gini - py * py
        return gini

    def get_split_property_cart(self,X,Y,A):#用基尼系数算最佳分割属性
        min_gini = None
        gini_idx = None
        gini_val = None
        for a in A:
            for av in np.unique(X[:,a]):
                select_idxs_left = X[:, a] <= av
                select_idxs_right = ~select_idxs_left
                YLeft = Y[select_idxs_left]
                YRight = Y[select_idxs_right]

                gini_a = self.cal_gini(YLeft) * YLeft.shape[0] / Y.shape[0] + self.cal_gini(YRight) * YRight.shape[0] / Y.shape[0]
                if min_gini is None or gini_a < min_gini:
                    min_gini = gini_a
                    gini_idx = a
                    gini_val = av
        return gini_idx,gini_val

    def build_node(self,X,Y,A,split_algo):
        if np.sum(Y == Y[0]) == len(Y):
            return Node(-1,Y[0],None,-1)
        elif len(A) == 0 or np.sum([np.sum(X[:,a] == X[0,a]) for a in A]) == len(A) * X.shape[0]:
            return Node(-1,np.argmax(np.bincount(Y)),None,-1)
        else:
            # if split_algo == 'id3':
            #     split_a = self.get_spilt_propertys_id3(X,Y,A)
            # elif split_algo == 'cart':


            #预留随机森林接口
            if self.max_feature is None:
                split_property = A
            else:
                split_property = np.random.choice(A,min(self.max_feature,len(A)),replace= False)

            split_a,split_val = self.get_split_property_cart(X,Y,split_property)

            #unique_a = self.feature_ables[split_a]

            node = Node(split_val,-1,[],split_a)

            select_idxs_left = X[:,split_a] <= split_val
            select_idxs_right = ~select_idxs_left
            DXS = [X[select_idxs_left,:],X[select_idxs_right,:]]
            DYS = [Y[select_idxs_left],Y[select_idxs_right]]

            for DX,DY in zip(DXS,DYS):
                if DX.shape[0] < self.min_split:
                    node.childs.append(Node(-1,np.argmax(np.bincount(Y)),None,-1))
                else:
                    node.childs.append(self.build_node(DX,DY,A,split_algo))
        return node

    def rebuild(self,X,Y,A,split_algo,min_split,max_feature = None):
        assert len(y) > 0
        self.min_split = min_split
        self.max_feature = max_feature
        self.root = self.build_node(X,Y,A,split_algo)

    def predict(self,X):
        p = self.root
        rs = -1
        while p is not None:
            if p.childs is None:
                rs = p.judge_result
                break
            else:
                if X[p.judge_index] <= p.judge_val:
                    idx = 0
                else:
                    idx = 1
                p = p.childs[idx]

        return rs

    def output(self):
        self.print(self.root)

    def print(self,node):

        p = node
        if p is None:
            return

        print(p.judge_val,p.judge_result,p.childs if p.childs is None else len(p.childs))

        if p.childs is None:
            return
        for child in p.childs:
            self.print(child)

class Random_tree():#默认使用sqrt(feature_num)作为随机森林K值
    def rebuild(self,X,Y,A,min_split,max_feature = None,tree_num = 10):
        feature_num = np.int64(np.sqrt(x.shape[1])) if max_feature is None else max_feature
        self.trees = []
        for i in range(tree_num):
            select_idxs = np.random.choice(range(X.shape[0]),X.shape[0])
            TX = X[select_idxs,:]
            TY = Y[select_idxs]
            my_tree = Decision_tree()
            my_tree.rebuild(TX,TY,A,'cart',min_split,feature_num)
            self.trees.append(my_tree)

    def predict(self,X):
        rs = []
        for my_tree in self.trees:
            rs.append(my_tree.predict(X))
        return np.argmax(np.bincount(rs))

x,y = sklearn.datasets.load_digits(return_X_y=True)
x = x.astype(np.int64)
y = y.astype(np.int64)
feature_ables = [np.unique(x[:,i]) for i in range(x.shape[1])]
x,xt,y,yt = sklearn.model_selection.train_test_split(x,y)
A = list(range(x.shape[1]))


def show_info(clf,arr = True):
    predicts = []
    for i in range(xt.shape[0]):
        if arr:
            proba = clf.predict(xt[i].reshape(1,-1))
            assert proba.shape[0] == 1
            predicts.append(proba[0])
        else:
            predicts.append(clf.predict(xt[i]))
        #print('is {} predict {}'.format(yt[i], predicts[i]))
    total = yt.shape[0]
    corrects = np.sum(predicts == yt)
    print('total = {} correct = {} accuracy = {}'.format(total, corrects, corrects / total))

begin = time.time()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x,y)
show_info(clf)
print('sklearn dt cost {} s'.format(time.time() - begin))

begin = time.time()
my_tree = Decision_tree()
my_tree.rebuild(x,y,A,'cart',1)
show_info(my_tree,False)
print('my dt cost {} s'.format(time.time() - begin))

clf_rf = RandomForestClassifier(min_samples_split=2)
clf_rf.fit(x,y)
show_info(clf_rf)whwhich


my_rf = Random_tree()
my_rf.rebuild(x,y,A,1)
show_info(my_rf,False)





