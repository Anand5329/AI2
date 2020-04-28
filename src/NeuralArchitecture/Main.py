import src.NeuralArchitecture.NeuralNetworks as NN
import src.NeuralArchitecture.Neurons as N
import src.NeuralArchitecture.Layer as L
import numpy as np
import random
from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X = X_train.reshape(X_train.shape[0],-1).T
Y = Y_train.reshape(Y_train.shape[0],1).T
ann = NN.NeuralNetwork(inp = 784, hidden = [16, 16], output = 10)
costs = ann.train(X, Y)
