import numpy as np
import matplotlib.pyplot as plot
#保留最后三列，两个特征和预测结果
data = np.loadtxt('./watermelon.csv',delimiter='\t')[:,-3:]
x = data[:,0:2]
y = data[:,-1:]

ones = np.ones((x.shape[0],1))
x = np.hstack((x,ones))
w = np.zeros((x.shape[1]))

def update_w_loss(x,w,y):
    num = x.shape[0]
    loss = np.sum([-y[i] * w.dot(x[i]) + np.log(1+np.exp(w.dot(x[i]))) for i in range(num)])
    d1 = 0
    d2 = 0
    for i in range(num):
        cal_val = w.dot(x[i])
        cal_pro0 = 1 / (1 + np.exp(cal_val))
        cal_pro1 = 1 - cal_pro0
        d1 = d1 - x[i] * (y[i] - cal_pro1)
        d2 = d2 + x[i].dot(x[i]) * cal_pro1 * cal_pro0
    w = w - np.power(d2,-1) * d1

    return loss,w

iter = 3000
losses = []
for i in range(iter):
    loss,w = update_w_loss(x,w,y)
    losses.append(loss)
    print(loss)

plot.subplot(1,2,1)
plot.plot(range(1,iter +1),losses)
plot.xlabel('iter')
plot.ylabel('loss')
plot.title('loss figure')


plot.subplot(1,2,2)
plot.scatter(x[:,0],x[:,1],10,c=np.reshape(y,17))
plot.xlabel('param1')
plot.xlabel('param2')
plot.colorbar()
plot.plot()
plot.show()

for i in range(x.shape[0]):
    print('is {} predict {}'.format(y[i],1 - 1 / (1 + np.exp(w.dot(x[i])) )))


