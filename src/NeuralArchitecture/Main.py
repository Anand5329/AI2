import src.NeuralArchitecture.NeuralNetworks as NN
import src.NeuralArchitecture.Neurons as N
import src.NeuralArchitecture.Layer as L
import numpy as np
import random
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X = X_train.reshape(X_train.shape[0],-1).T/255
Y = np.zeros((10,Y_train.size))


for y in range(len(Y_train)):
    Y[Y_train[y]][y]=1

ann = NN.NeuralNetwork(inp = 784, hidden = [16,16], output = 10)
epochs = 2000
learning_rate = 0.5

X_T = X_test.reshape(X_test.shape[0], -1).T/255
Y_T = np.zeros((10, Y_test.size))

for y in range(len(Y_test)):
    Y_T[Y_test[y]][y] = 1

# print(ann.measure_accuracy(X_T,Y_T))

tic = time.time()

costs, accuracies = ann.train(X, Y,epochs=epochs,learning_rate=learning_rate, measure_accuracy=True, X_test=X_T, Y_test=Y_T)

toc = time.time()

print("Time taken: "+str(toc-tic)+ " s")



plt.plot(range(1,epochs+1),costs)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Learning rate: "+str(learning_rate))
plt.show()

plt.plot(range(1,epochs+1), accuracies)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

print(ann.measure_accuracy(X_T,Y_T))