import h5py
import numpy as np
import src.NeuralArchitecture.NeuralNetworks as NN
import matplotlib.pyplot as plt
import src.NeuralArchitecture.verify as V

file = h5py.File("C:/Users/KD SP2/PycharmProjects/ArtificialIntelligence/src/Datasets/train_catvnoncat.h5", 'r')
test = h5py.File("C:/Users/KD SP2/PycharmProjects/ArtificialIntelligence/src/Datasets/test_catvnoncat.h5", 'r')
print(list(file.keys()))

X_train = file['train_set_x']
Y_train = file['train_set_y']

X_test = test['test_set_x']
Y_test = test['test_set_y']
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0],-1).T/255
Y_test = np.array(Y_test)
Y_test = Y_test.reshape(Y_test.size,1).T


X_t = np.array(X_train)

X = X_t.reshape(X_t.shape[0],-1)/255


Y = np.array(Y_train)
Y = Y.reshape(Y.size,1)

X = X.T
Y = Y.T
epochs = 2500
learning_rate = 0.0075
layer_dims = [X.shape[0], 20,7,5,1]


def my_model():
    model = NN.NeuralNetwork(X.shape[0], [20, 7, 5], 1)
    costs, accuracies = model.train(X, Y, epochs, learning_rate, measure_accuracy=True, X_test=X_test, Y_test=Y_test,
                                    activation='relu')

    plt.plot(range(1, epochs + 1), costs)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Learning rate: " + str(learning_rate))
    plt.show()

    plt.plot(range(1, epochs + 1), accuracies['test'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Learning rate: " + str(learning_rate))
    plt.show()

print(np.sum(89))
para = V.L_layer_model(X,Y,layer_dims,learning_rate,epochs,True,X_test,Y_test)
print(V.measure_accuracy(X_test,Y_test,para))