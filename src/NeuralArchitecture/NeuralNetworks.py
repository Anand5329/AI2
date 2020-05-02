import src.NeuralArchitecture.Neurons
import src.NeuralArchitecture.Layer as L
import numpy as np
import math
import sklearn.utils as sku

class NeuralNetwork:

    def __init__(self, inp=1, hidden=[1], output=1):
        self.input = inp
        self.hidden = hidden
        self.output = output
        self.L = len(hidden)+2
        self.input_layer = L.Layer(0, inp, values=[])
        self.hidden_layers = []
        self.hidden_layers.append(L.Layer(inp, hidden[0]))
        self.training_data = []
        for i in range(1, len(hidden)):
            self.hidden_layers.append(L.Layer(hidden[i - 1], hidden[i]))
        self.output_layer = L.Layer(hidden[len(hidden) - 1], output)

    def update_input(self, inputs):
        self.input_layer.values = inputs
        return

    def forward_propagation(self, inputs):  # forward propagation (calculating values)
        # for hidden layers:
        self.update_input(inputs)
        for i in range(len(self.hidden)):
            if i == 0:
                layer = self.input_layer
            else:
                layer = self.hidden_layers[i - 1]
            self.hidden_layers[i].z = NeuralNetwork.feed_forward(layer, self.hidden_layers[i])
            for j in range(len(self.hidden_layers[i].z)):
                self.hidden_layers[i].values[j] = NeuralNetwork.sigmoid(self.hidden_layers[i].z[j])
            self.hidden_layers[i].update_values()

        # for output layer:
        layer = self.hidden_layers[len(self.hidden) - 1]
        self.output_layer.z = NeuralNetwork.feed_forward(layer, self.output_layer)
        for i in range(len(self.output_layer.z)):
            self.output_layer.values[i] = NeuralNetwork.sigmoid(self.output_layer.z[i])
        self.output_layer.update_values()
        return

    def forward_prop_vec(self, inputs):
        self.update_input(inputs)
        A = inputs
        Z_cache = []
        for i in range(1, self.L-1):
            W = self.hidden_layers[i-1].weights
            b = self.hidden_layers[i-1].biases
            self.hidden_layers[i-1].z = np.dot(W,A) + b
            A = NeuralNetwork.sigmoid(self.hidden_layers[i-1].z)
            self.hidden_layers[i-1].values = A
        self.output_layer.z = np.dot(self.output_layer.weights,A) + self.output_layer.biases
        self.output_layer.values = NeuralNetwork.sigmoid(self.output_layer.z)
        return

    @staticmethod
    def get_weights(layer):
        weights = []
        for i in range(len(layer.neurons)):
            weights.append(layer.neurons[i].weights)
        return weights

    @staticmethod
    def sigmoid(x):
        x = x.astype('float64')
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def feed_forward(previous_layer, current_layer):
        biases = current_layer.biases
        values = previous_layer.values
        weights = NeuralNetwork.get_weights(current_layer)
        current_layer.weights = weights
        np_weights = np.array(weights).transpose()
        np_values = np.array(values)
        np_x = np.dot(np_values, np_weights)

        for i in range(len(np_x)):
            np_x[i] = np_x[i] + biases[i]
        return list(np_x)

    @staticmethod
    def sigmoid_derivative(x):
        y = NeuralNetwork.sigmoid(x)
        return np.multiply(1-y,y)

    def calculate_last_layer_error_mse(self, needed_output):
        error = []
        for i in range(len(needed_output)):
            d = (NeuralNetwork.sigmoid(self.output_layer.z[i]) - needed_output[i]) * NeuralNetwork.sigmoid_derivative(self.output_layer.z[i])
            error.append(d)
        return error

    @staticmethod
    def calculate_layer_error(next_layer, current_layer):
        error = []
        np_weights = np.array(next_layer.weights)
        np_error = np.array(next_layer.error)
        np_w_dot_e = np.dot(np_weights.transpose(), np_error)
        y_list = list(np_w_dot_e)
        z_list = current_layer.z
        for z, y in zip(z_list, y_list):
            error.append(NeuralNetwork.sigmoid_derivative(z) * y)
        return error

    @staticmethod
    def mean_squared_error(needed_output, existing_output_layer):
        sum = 0
        for x, y in zip(needed_output, existing_output_layer.values):
            sum = sum + (x - y) ** 2
        return sum / existing_output_layer.size

    @staticmethod
    def cross_entropy_cost(AL, Y): #TODO: change cost
        return (np.sum((- np.dot(Y,np.log(AL).T) - np.dot(1-Y,np.log(1-AL).T))/Y.shape[1]))

    def calculate_errors(self, needed_output):
        self.output_layer.error = self.calculate_last_layer_error_mse(needed_output)  # calculating last layer errors
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            if i == len(self.hidden_layers) - 1:
                next_layer = self.output_layer
            else:
                next_layer = self.hidden_layers[i + 1]

            self.hidden_layers[i].error = NeuralNetwork.calculate_layer_error(next_layer, self.hidden_layers[
                i])  # calculating other layers' errors
        return

    def calculate_gradient(self):
        L = len(self.hidden_layers) + 2
        w_gradient = np.empty(shape=(L,self.hidden[1],self.input), dtype='float')
        b_gradient = np.empty(shape=(L,self.hidden[1]), dtype='float')

        for l in range(1, L - 1):  # not last layer, that's why L-1
            for j in range(self.hidden[l - 1]):
                if l == 1:
                    size = self.input
                else:
                    size = self.hidden[l - 2]
                for i in range(size):
                    w_gradient[l-1][j][i] = self.partial_derivative_weight(l, j, i)
                    b_gradient[l-1][j] = self.hidden_layers[l - 1].error[j]

        for j in range(self.output):  # last layer
            for i in range(self.hidden[L - 3]):
                w_gradient[L-1][j][i] = self.partial_derivative_weight(L, j, i)
            b_gradient[L-1][j] = self.output_layer.error[j]
        return w_gradient, b_gradient

    def back_propagation(self, needed_output):
        self.calculate_errors(needed_output)
        return self.calculate_gradient()

    def partial_derivative_weight(self, n_layer, j, i):
        if n_layer == len(self.hidden_layers) + 2:
            k = self.output_layer
            prev_k = self.hidden_layers[n_layer - 3]
        elif n_layer == 1:
            k = self.hidden_layers[n_layer - 1]
            prev_k = self.input_layer
        else:
            k = self.hidden_layers[n_layer - 1]
            prev_k = self.hidden_layers[n_layer - 2]
        return k.error[j] * prev_k.values[i]

    def sigmoid_derivative(self, layer):
        return np.multiply(1-layer.values, layer.values)

    def back_step(self, dA, layer, prev_layer, m):
        dZ = np.multiply(dA, self.sigmoid_derivative(layer))
        layer.grad["W"] = np.dot(dZ, prev_layer.values.T) / m
        layer.grad["b"] = np.sum(dZ, axis=1, keepdims=True) / m
        return np.dot(layer.weights.T, dZ)

    def backward_prop_vec(self, Y, learning_rate):
        m = Y.shape[1]
        dAL = -np.divide(Y, self.output_layer.values) + np.divide(1 - Y, 1 - self.output_layer.values)
        dA = self.back_step(dAL, self.output_layer, self.hidden_layers[self.L - 2-1], m)
        for l in reversed(range(1, self.L - 2)):
            dA = self.back_step(dA, self.hidden_layers[l], self.hidden_layers[l - 1], m)
        dA = self.back_step(dA, self.hidden_layers[0], self.input_layer, m)
        self.update(learning_rate)
        return

    def update(self, learning_rate):
        for l in range(self.L - 2):
            self.hidden_layers[l].weights = self.hidden_layers[l].weights - learning_rate * self.hidden_layers[l].grad[
                "W"]
            self.hidden_layers[l].biases = self.hidden_layers[l].biases - learning_rate * self.hidden_layers[l].grad[
                "b"]
        self.output_layer.weights = self.output_layer.weights - learning_rate * self.output_layer.grad["W"]
        self.output_layer.biases = self.output_layer.biases - learning_rate * self.output_layer.grad["b"]
        return

    def predict(self, X):
        self.forward_prop_vec(X)
        prediction = np.zeros((self.output,X.shape[1]))
        for i in range(X.shape[1]):
            y = self.output_layer.values[:,i].reshape(self.output, 1)
            max = np.max(y)
            for j in range(y.size):
                if y[j] == max:
                    prediction[j][i] = 1
                else:
                    prediction[j][i] = 0
        return prediction

    def measure_accuracy(self, X_test, Y_test):
        Y_predict = self.predict(X_test)

        ctr = 0
        for i in range(Y_test.shape[1]):
            check = Y_test[:, i] == Y_predict[:,i]
            # print(check.all)
            if check.all():
                ctr += 1
            else:
                pass
        # print("Accuracy: " + str(ctr / Y_T.shape[1]))
        return ctr/Y_test.shape[1]

    def print_cost(self, inputs, needed_output):
        self.forward_propagation(inputs)
        print(self.mean_squared_error(needed_output, self.output_layer))
        return

    def input_training_data(self, training_data):
        self.training_data = training_data

    def gradient_descent(self, w_gradient, b_gradient):
        L = len(self.hidden_layers) + 2
        for l in range(1, L - 1):  # not last layer, that's why L-1
            for j in range(self.hidden[l - 1]):
                if l == 1:
                    size = self.input
                else:
                    size = self.hidden[l - 2]
                for i in range(size):
                    self.hidden_layers[l-1].weights[j][i] = self.hidden_layers[l-1].weights[j][i] - w_gradient[l-1][j][i]
                self.hidden_layers[l-1].biases[j] = self.hidden_layers[l-1].biases[j] - b_gradient[l-1][j]
            self.hidden_layers[l-1].update_biases()
            self.hidden_layers[l-1].update_weights()

        for j in range(self.output):  # last layer
            for i in range(self.hidden[L - 3]):
                self.output_layer.weights[j][i] = self.output_layer.weights[j][i] - w_gradient[L-1][j][i]
            self.output_layer.biases[j] = self.output_layer.biases[j] - b_gradient[L-1][j]
        self.output_layer.update_weights()
        self.output_layer.update_biases()
        return

    def train(self, X, Y, epochs = 2000, learning_rate = 1.2, measure_accuracy = False, X_test = [], Y_test = []):
        costs = []
        accuracies = {"train" : [], "test" : []}
        for i in range(epochs):
            X,Y = sku.shuffle(X.T,Y.T)
            X = X.T
            Y = Y.T
            self.forward_prop_vec(X)
            cost = NeuralNetwork.cross_entropy_cost(self.output_layer.values, Y)
            print("Cost after epoch {0}: {1}".format(i+1,cost))
            costs.append(cost)
            self.backward_prop_vec(Y, learning_rate)
            if(measure_accuracy):
                accuracies["test"].append(self.measure_accuracy(X_test,Y_test))
                # accuracies["train"].append(self.measure_accuracy(X, Y))
        return costs,accuracies

    def stochastic_GD(self, batch_size):
        L = len(self.hidden_layers) + 2
        w_gradient = np.empty(shape=(L, self.hidden[1], self.input), dtype='float')
        b_gradient = np.empty(shape=(L, self.hidden[1]), dtype='float')
        for i in range(0,len(self.training_data),batch_size):
            batch = self.training_data[i:i+batch_size]
            for x, y in zip(batch[0], batch[1]):
                self.forward_propagation(x)
                w_g, b_g = self.back_propagation(y)
                for id, x in np.ndenumerate(w_g):
                    w_gradient[id] = w_gradient[id] + x
                for id, x in np.ndenumerate(b_g):
                    b_gradient[id] = b_gradient[id] + x
            for id, w in np.ndenumerate(w_gradient):
                w_gradient[id] = w/batch_size
            for id, b in np.ndenumerate(b_gradient):
                b_gradient[id] = b/batch_size
            self.gradient_descent(w_gradient, b_gradient)

    def save_model(self, name, epochs, learning_rate):
        file = open(name + ".txt", 'wt')
        try:
            file.write("Model name: " + name + "\n")
            file.write("Epochs: " + str(epochs) + "\n")
            file.write("Learning Rate: " + str(learning_rate) + "\n")
            file.write("Input Layer Size: " + str(self.input) + "\n")
            file.write("Hidden Layers': " + str(self.hidden) + "\n")
            file.write("Output Layer Size: " + str(self.output) + "\n")
            file.write("Weights and Biases:" + "\n")
            for id, x in np.ndenumerate(self.hidden_layers):
                file.write("Size: " + str(self.hidden_layers[id[0]].size) + "\n")
                file.writelines(str(x.weights) + "\n")
                file.writelines(str(x.biases) + "\n")
            file.write("Size: " + str(self.output) + "\n")
            file.writelines(str(self.output_layer.weights) + "\n")
            file.writelines(str(self.output_layer.biases) + "\n")
        finally:
            file.close()
        return

    def load_model(self, name):
        file = open(name+".txt", 'rt')
        file.readline()
        print(file.readline())
        print(file.readline())
        self.input = int(file.readline().split()[-1])
        self.hidden = list(file.readline().split()[-1])
        self.output = int(file.readline().split()[-1])
        file.readline()
        for i in self.hidden:
            pass #TODO: complete load_model


    #TODO: implement ReLU
    #TODO: try to make it faster