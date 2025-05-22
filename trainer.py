import numpy as np
import matplotlib.pyplot as plt

class neural_network():
    """
    input, hidden, and output sizes are all vertical length of layers
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate, loss_func="mse", act_func="sigmoid"):

        #Here we have two hidden layers
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

        self.loss_function = loss_func
        self.act_function = act_func

        self.lr = learning_rate

        self.train_loss = []
        self.validation_loss = []

    def forwardProp(self,inputs):
        
        self.r1 = np.dot(inputs, self.weights1) + self.bias1
        self.a1 = self.activation(self.r1)
        r2 = np.dot(self.a1, self.weights2) + self.bias2
        if self.loss_function == 'categorical_crossentropy':

            if(self.act_function != "soft_max"):
                raise ValueError("Invalid combination of activation and loss functions")
            
            tmp = np.exp(r2 - np.max(r2, axis=1, keepdims=True))
            self.a2 = tmp / np.sum(tmp, axis=1, keepdims=True)
        else:
            self.a2 = self.activation(r2)
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
        elif self.loss_function == 'log_loss':
            return -np.mean(predicted*np.log(predicted) + (1-measured)*np.log(1-predicted))
        elif self.loss_function == 'categorical_crossentropy':
            return -np.mean(measured*np.log(predicted))
        else:
            raise ValueError('Invalid loss function')
        
    def calculate_gradient(self, label):

        if self.loss_function == 'mse':
            return self.a2 - label
        elif self.loss_function == 'log_loss':
            return -(label/self.a2 - (1-label)/(1-self.a2))
        elif self.loss_function == 'categorical_crossentropy':
            return self.a2 - label
        else:
            raise ValueError('Invalid loss function')
        
    def activation_derivative(self, c):
        
        if self.act_function == "sigmoid" or self.act_function == "soft_max":
            return c * (1 - c)
        elif self.act_function == "tan":
            return 1 / (np.cosh(c))**2
        else:
            return ValueError("Invalid activation function")

    def activation(self, c):

        if self.act_function == "sigmoid":
            return 1 / (1 + np.exp(-c))
        elif self.act_function == "tan":
            return np.tanh(c)
        elif self.act_function == "soft_max":

            if(self.loss_function != "categorical_crossentropy"):
                raise ValueError("Invalid combination of activation and loss functions")

            tmp = np.exp(c - np.max(c, axis=1, keepdims=True))
            return tmp / np.sum(tmp, axis=1, keepdims=True)
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
            # print("Test accuracy = %f" % (np.mean(self.a2 == ytest)))
        if(np.any(xval) != None and np.any(yval) != None):
            plt.scatter(x,self.validation_loss,color='red', label="validation_loss")
        
        plt.plot(x,self.train_loss,color='blue',label="train_loss")
        plt.legend()
        plt.xlabel("Epoches")
        plt.ylabel("Training Cost")
        plt.title("Training Error vs Epoches")
        plt.show()


        return self.weights1, self.bias1, self.weights2, self.bias2, self.a2
