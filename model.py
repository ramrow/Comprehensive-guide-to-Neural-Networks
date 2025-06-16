import numpy as np
import matplotlib.pyplot as plt

class neural_network():
    # input, hidden, and output sizes are all vertical length of layers
    def __init__(self, input_dimension, hidden_dimension, output_dimension, learning_rate, loss_func="mse", act_func="sigmoid"):

        self.weights1 = np.random.randn(input_dimension, hidden_dimension)
        self.bias1 = np.zeros((1, hidden_dimension))
        print(self.bias1.shape)
        self.weights2 = np.random.randn(hidden_dimension, output_dimension)
        self.bias2 = np.zeros((1, output_dimension))

        self.loss_function = loss_func
        self.act_function = act_func
        self.lr = learning_rate

        self.train_loss = []
        self.validation_loss = []

    def forwardProp(self,inputs):
        #((300,2) * (2,2) = (300,2)) + (1,2)
        self.r1 = np.dot(inputs, self.weights1) + self.bias1
        # self.r1.shape = (300,2)
        self.a1 = self.activation(self.r1)
        # self.a1.shape = (300,2)
        r2 = np.dot(self.a1, self.weights2) + self.bias2
        #((300,2) * (2,1) = (300,1)) + (1,1)
        # r2.shape = (300,1)
        self.a2 = self.activation(r2)
        # self.a2.shape = (300,1)
        # a2 is the results of the model when called, forward function can be used to calculate results when used once.
        return self.a2
    
    def backwardProp(self,input,label):
        
        m,_ = input.shape
        tmp = self.calculate_gradient(label)
        old_weights = self.weights2
        self.weights2 -= self.lr * ((1 / m) * np.dot(self.a1.T, tmp))
        self.bias2 -= self.lr * ((1 / m) * np.sum(tmp, axis=0, keepdims=True))
        temp = np.dot(tmp, old_weights.T) * self.activation_derivative(self.a1)
        self.weights1 -= self.lr * ((1 / m) * np.dot(input.T, temp))
        self.bias1 -= self.lr * ((1 / m) * np.sum(temp, axis=0, keepdims=True))

    def calculate_error(self,predicted,measured):

        if self.loss_function == 'mse':
            return np.mean((predicted - measured)**2)
        elif self.loss_function == 'log':
            return -np.mean(predicted*np.log(predicted) + (1-measured)*np.log(1-predicted))
        else:
            raise ValueError('Invalid loss function')

    def calculate_gradient(self, label):

        if self.loss_function == 'mse':
            return self.a2 - label
        elif self.loss_function == 'log':
            return -(label/self.a2 - (1-label)/(1-self.a2))
        else:
            raise ValueError('Invalid loss function')
        
    def activation_derivative(self, c):
        
        if self.act_function == "sigmoid":
            return c * (1 - c)
        elif self.act_function == "tan":
            return 1 / (np.cosh(c))**2
        elif self.act_function == "reLU":
            return 1. * (c > 0)
        else:
            return ValueError("Invalid activation function")
        

    def activation(self, c):

        if self.act_function == "sigmoid":
            return 1 / (1 + np.exp(-c))
        elif self.act_function == "tan":
            return np.tanh(c)
        elif self.act_function == "reLU":
            return (c * (c > 0))
        else:
            return ValueError("Invalid activation function")
        
    def train(self, epoch, xtrain, ytrain, xval=None, yval=None, xtest=None, ytest=None):

        for i in range(epoch):
            self.forwardProp(xtrain)
            self.backwardProp(xtrain,ytrain)
            self.train_loss.append(self.calculate_error(self.a2, ytrain))

            if(np.any(xval) != None and np.any(yval) != None):
                self.forwardProp(xval)
                self.validation_loss.append(self.calculate_error(self.a2,yval))
            
                print('Epoch for %d/%d : Train data loss = %f - Validation data loss = %f' % (i, epoch, self.train_loss[i], self.validation_loss[i]))
            else:
                print('Epoch for %d/%d : Train data loss = %f' % (i, epoch, self.train_loss[i]))

        
        x = np.linspace(0,epoch+1,len(self.train_loss))

        if(np.any(xtest) != None and np.any(ytest) != None):
            self.forwardProp(xtest)
        if(np.any(xval) != None and np.any(yval) != None):
            plt.scatter(x,self.validation_loss,color='red', label="validation_loss")
        
        plt.plot(x,self.train_loss,color='blue',label="train_loss")
        plt.legend()
        plt.xlabel("Epoches")
        plt.ylabel("Training Loss")
        plt.title("Training Loss vs Epoches")
        plt.show()