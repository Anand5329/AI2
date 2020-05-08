import src.NeuralArchitecture.NeuralNetworks as NN
import src.NeuralArchitecture.Neurons as N
import src.NeuralArchitecture.Layer as L
import src.NeuralArchitecture.verify as V
import numpy as np
import random
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time
import h5py

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X = X_train.reshape(X_train.shape[0],-1).T/255
Y = np.zeros((10,Y_train.size))


for y in range(len(Y_train)):
    Y[Y_train[y]][y]=1

ann = NN.NeuralNetwork(inp = 784, hidden = [16,16], output = 10)
epochs = 2000
learning_rate = 0.075
activation = 'relu'

X_T = X_test.reshape(X_test.shape[0], -1).T/255
Y_T = np.zeros((10, Y_test.size))

for y in range(len(Y_test)):
    Y_T[Y_test[y]][y] = 1

# print(ann.measure_accuracy(X_T,Y_T))
def compare():
    print("Comparing:")
    Y_P = ann.predict(X_T)
    Y_predict = []
    for i in range(Y_P.shape[1]):
        x = Y_P[:, i]
        for i in range(x.size):
            if x[i] == 1:
                Y_predict.append(i)

    for x, y in zip(Y_test, Y_predict):
        if x!=y:
            print(str(x) + ", " + str(y))
    return


def my_model():
    tic = time.time()
    costs, accuracies = ann.train(X, Y, epochs=epochs, learning_rate=learning_rate, measure_accuracy=True, X_test=X_T,
                                  Y_test=Y_T, activation=activation)
    toc = time.time()
    print("Time taken: " + str(toc - tic) + " s")
    print("Time per epoch: " + str((toc - tic) / epochs))

    plt.plot(range(1, epochs + 1), costs)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Learning rate: " + str(learning_rate))
    plt.show()

    # plt.plot(range(1,epochs+1), accuracies["train"], 'b', label='Train')
    plt.plot(range(1, epochs + 1), accuracies["test"], 'r', label='Test')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

    compare()

    print(ann.measure_accuracy(X_T, Y_T))
    ann.save_model('o_point_4', 2000, 0.4)
    return tic, toc


def verify(layers_dim):
    return V.L_layer_model(X, Y, layers_dim, learning_rate, epochs, True, X_T, Y_T, print_acc=True)


para = verify([784,16,16,10])


#TODO: use very simple architecture


