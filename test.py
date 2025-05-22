import numpy as np
import matplotlib.pyplot as plt
from trainer import neural_network as nn
from matplotlib.colors import ListedColormap


def create_inputs():
    train_f = open("data/train.d")
    xtrain = []
    ytrain = []
    for line in train_f:
        line = line[:-1]
        arr = line.split(' ')
        xtrain.append(np.array([float(arr[0]),float(arr[1])]))
        # ytrain.append(np.array([int(arr[2])]))
        if(arr[2] == '-1'):
            ytrain.append(np.array([0]))
        else:
            ytrain.append(np.array([1]))


    test_f = open("data/test.d")
    xtest = []
    ytest = []
    for line in test_f:
        line = line[:-1]
        arr = line.split(' ')
        xtest.append(np.array([float(arr[0]),float(arr[1])]))
        # ytest.append(np.array([int(arr[2])]))
        if(arr[2] == '-1'):
            ytest.append(np.array([0]))
        else:
            ytest.append(np.array([1]))
    return np.array(xtrain), np.array(ytrain), np.array(xtest), np.array(ytest)

def plot(x, y, model):
    one_i,one_s = [],[]
    n_i,n_s = [],[]
    for i in range(len(x)):
        if(y[i] == 1):
            one_i.append(x[i][0])
            one_s.append(x[i][1])
        else:
            n_i.append(x[i][0])
            n_s.append(x[i][1])

    x1 = np.linspace(-1,1,50)
    x2 = np.linspace(-1,1,50)
    X,Y = np.meshgrid(x1,x2)
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    res = model.forwardProp(grid)
    r = []
    for i in range(len(grid)):
        r.append(res[i])
    Z = np.array(r).reshape(X.shape)
    m = ListedColormap(['#C0C0C0','#808080'])
    plt.pcolormesh(X,Y,Z,cmap=m)
    plt.scatter(one_i,one_s,color='blue',marker='x')
    plt.scatter(n_i,n_s,color='red',marker='x')
    plt.xlabel("Insenity")
    plt.ylabel("Symmetry")
    plt.grid(True)
    plt.show()

xtrain, ytrain, xtest, ytest = create_inputs()
m,n = (xtrain.shape)

model = nn(n,2,1,0.1,act_func="sigmoid")
_,_,_,_,test_output = model.train(1000000,xtrain,ytrain,xtest=xtest,ytest=ytest)
plot(xtest,ytest,model)
plot(xtrain,ytrain,model)
