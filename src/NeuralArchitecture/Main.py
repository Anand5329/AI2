import src.NeuralArchitecture.NeuralNetworks as NN
import src.NeuralArchitecture.Neurons as N
import src.NeuralArchitecture.Layer as L
import numpy as np
import random
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X = X_train.reshape(X_train.shape[0],-1).T
Y = np.zeros((10,Y_train.size))

for y in range(len(Y_train)):
    Y[Y_train[y]][y]=1

ann = NN.NeuralNetwork(inp = 784, hidden = [4,4], output = 10)
epochs = 50
costs = ann.train(X, Y,epochs=epochs,learning_rate=0.5)

Y_predict = ann.predict(X_test.reshape(X_test.shape[0],-1).T)
Y_T = np.zeros((10,Y_test.size))

for y in range(len(Y_test)):
    Y_T[Y_test[y]][y]=1
ctr=0
for i in range(Y_T.shape[1]):
    if (Y_T[:,i]==Y_predict[:,i]).all:
        ctr +=1
print("Accuracy: "+str(ctr/Y_T.shape[1]))

plt.plot(range(1,epochs+1),costs)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()