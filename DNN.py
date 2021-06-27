import matplotlib
matplotlib.use('Agg')

import time
from math import log
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_val, y_val) = mnist.load_data()

x_train = np.array(x_train, dtype = "float32") / 255.0
x_val = np.array(x_val, dtype = "float32") / 255.0
x_train = x_train.reshape(x_train.shape[0], 784)
x_val = x_val.reshape(x_val.shape[0], 784)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_val = lb.transform(y_val)

y_train = np.array(y_train)
y_val = np.array(y_val)

class DeepNueralNetwork:
    def __init__(self, sizes, epochs = 10, l_rate = 0.01):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.train_log_loss = []
        self.val_log_loss = []

        #Activation parameters
        self.K0_list = []
        self.K1_list = []

        #We will save all the parameters of the neural network in a dictionary.
        self.params = self.initialization()



    def initialization(self):
        #Number of nodes in each layers
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]
        k0 = np.random.randn()
        k1 = np.random.randn()

        params = {'W1' : np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1), 'W2' : np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2), 'W3' : np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)}
        params['K1'] = k1
        params['K0'] = k0
        self.K0_list.append(k1)
        self.K1_list.append(k0)

        return params

    def activation_(self, x, derivative = False):
        if derivative:
            return np.array([self.params['K1']] * x.shape[0])
        return self.params['K0'] + self.params['K1'] * x

    def softmax(self, x, derivative = False):
        #Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis = 0)

    def forward_pass(self, x_train):
        params = self.params

        #input layer sample becomes the sample
        params['A0'] = x_train

        #input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params["A0"])
        params['A1'] = self.activation_(params['Z1'])

        #hidden layer to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params["A1"])
        params['A2'] = self.activation_(params["Z2"])

        #hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params["A2"])
        params['A3'] = self.softmax(params["Z3"])

        return params['A3']

    def backward_pass(self, y_train, output):
        """
        This is backpropagation algorithm, for calculating the updates of nueral network's parameters.
        """
        params = self.params
        change_p = {}

        #calculating W3 update
        error = output - y_train
        change_p['W3'] = np.outer(error, params["A2"])

        #calculating W2 update
        # Here we are calculating the Hadamard product between gradient of error function w.r.t activation of second layer and derivative of the activation function w.r.t. pre activations of the same second layer.
        error_2 = np.dot(params['W3'].T, error) * self.activation_(params['Z2'], derivative = True)
        change_p['W2'] = np.outer(error_2, params['A1'])

        #calculating W1 updates
        error_3 = np.dot(params['W2'].T, error_2) * self.activation_(params['Z1'], derivative = True)
        change_p['W1'] = np.outer(error_3, params['A0'])

        # updating K0 and K1.
        change_p['K0'] = np.mean(np.dot(params['W2'].T, error_2))
        change_p['K1'] = np.mean(np.dot(params['W2'].T, error_2) * params['Z1'])

        return change_p

    def update_network_parameters(self, changes_to_p, epoch_number):
        """
        Update network according to the update rule from stochastic gradient descent.
        θ = θ - η * ∇J(x, y),
            theta θ:            a network parameter (e.g. a weight w)
            eta η:              the learning rate
            gradient ∇J(x, y):  the gradient of the objective function,
                                i.e. the change for a specific theta θ
        """

        for key, value in changes_to_p.items():
            self.params[key] -= self.l_rate * value


    def compute_accuracy(self, x_val, y_val):
        """
        This function does a forward pass of x, then checks whether the indices of the maximum value of x is equal to the indices of the label y. Then it sums over each prediction and calculate the accuracy. Accuracy is calculated by checking the correct number of output out of total samples.
        """
        predictions = []
        log_loss = 0

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
            log_loss += -log(output[np.argmax(y)])

        self.val_log_loss.append(log_loss /len(x_val))

        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()

        for iteration in range(self.epochs):
            train_predictions = []
            log_loss = 0
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_p = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_p, iteration + 1)

                log_loss += -log(output[np.argmax(y)])

                pred = np.argmax(output)
                train_predictions.append(pred == np.argmax(y))

            self.train_log_loss.append(log_loss/len(x_train))
            train_accuracy = np.mean(train_predictions)
            val_accuracy = self.compute_accuracy(x_val, y_val)

            self.K0_list.append(self.params['K0'])
            self.K1_list.append(self.params['K1'])

            print('Epoch: {0}, Time Spent: {1:.2f}s, Val_Accuracy: {2:.2f}%, Train_Accuracy: {3:.2f}%'.format(iteration+1, time.time() - start_time, val_accuracy * 100, train_accuracy * 100))


dnn = DeepNueralNetwork(sizes = [784, 128, 64, 10])
dnn.train(x_train, y_train, x_val, y_val)


plt.style.use("ggplot")
plt.figure()
N = np.arange(0, 10)

plt.plot(N, dnn.train_log_loss, label = "traning log loss")
plt.plot(N, dnn.val_log_loss, label = "validation log loss")
plt.xlabel("Epochs count")
plt.ylabel("Train/Val log loss")
plt.legend()
plt.plot()
plt.savefig("Log_loss.png")
