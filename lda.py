import numpy as np
import matplotlib.pyplot as plot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#保留最后三列，两个特征和预测结果
data = np.loadtxt('./watermelon.csv',delimiter='\t')[:,-3:]
x = data[:,0:2]
y = data[:,-1:]

# ones = np.ones((x.shape[0],1))
# x = np.hstack((x,ones))
#w = np.zeros((x.shape[1]))


x0 = np.array([x[i] for i in range(x.shape[0]) if y[i] == 0.0])
x1 = np.array([x[i] for i in range(x.shape[0]) if y[i] == 1.0])
u0 = np.mean(x0)
u1 = np.mean(x1)

sw = np.zeros(2)
for i in range(x0.shape[0]):
    z = x0[i] - u0
    sw = sw + z.dot(z)

for i in range(x1.shape[0]):
    z = x1[i] - u1
    sw = sw + z.dot(z)

w = np.mat(sw).I * (u0 - u1)

for i in range(x.shape[0]):
    print('is {} predict {}'.format(y[i],1 - 1 / (1 + np.exp(x[i].dot(w))) ))


clf = LDA()
clf.fit(x,y)
for i in range(x.shape[0]):
    print(clf.predict([x[i],]))
