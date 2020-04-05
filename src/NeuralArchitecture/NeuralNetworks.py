import src.NeuralArchitecture.Neurons
import src.NeuralArchitecture.Layer as L
import numpy as np
import math


class NeuralNetwork:

    def __init__(self, inp=1, hidden=[1], output=1, inputs=[]):
        self.input = inp
        self.hidden = hidden
        self.output = output
        self.input_layer = L.Layer(0, inp, values=inputs)
        self.hidden_layers = []
        self.hidden_layers.append(L.Layer(inp, hidden[0]))
        for i in range(1, len(hidden)):
            self.hidden_layers.append(L.Layer(hidden[i - 1], hidden[i]))
        self.output_layer = L.Layer(hidden[len(hidden) - 1], output)

    def forward_propagation(self):  # forward propagation (calculating values)
        # for hidden layers:
        for i in range(len(self.hidden)):
            if i == 0:
                layer = self.input_layer
            else:
                layer = self.hidden_layers[i - 1]
            self.hidden_layers[i].z = NeuralNetwork.feed_forward(layer, self.hidden_layers[i])
            for j in range(len(self.hidden_layers[i].z)):
                self.hidden_layers[i].values[j] = NeuralNetwork.activation_sigmoid(self.hidden_layers[i].z[j])
            self.hidden_layers[i].update_values()

        # for output layer:
        layer = self.hidden_layers[len(self.hidden) - 1]
        self.output_layer.z = NeuralNetwork.feed_forward(layer, self.output_layer)
        for i in range(len(self.output_layer.z)):
            self.output_layer.values[i] = NeuralNetwork.activation_sigmoid(self.output_layer.z[i])
        self.output_layer.update_values()
        return

    @staticmethod
    def get_weights(layer):
        weights = []
        for i in range(len(layer.neurons)):
            weights.append(layer.neurons[i].weights)
        return weights

    @staticmethod
    def activation_sigmoid(x):
        return 1 / (1 + math.e ** (-x))

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
        y = NeuralNetwork.activation_sigmoid(x)
        return (1 - y) * y

    def calculate_last_layer_error_mse(self, needed_output):
        error = []
        for i in range(len(needed_output)):
            d = (NeuralNetwork.activation_sigmoid(self.output_layer.z[i]) - needed_output[i]) * NeuralNetwork.sigmoid_derivative(self.output_layer.z[i])
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
    def mean_squared_error(needed_output, existing_output):
        sum = 0
        for x, y in zip(needed_output, existing_output.values):
            sum = sum + (x - y) ** 2
        return sum / existing_output.size

    def back_propagation(self, needed_output):

        self.output_layer.error = self.calculate_last_layer_error_mse(needed_output)  # calculating last layer errors
        for i in range(len(self.hidden_layers)-1,-1,-1):
            if i == len(self.hidden_layers) - 1:
                next_layer = self.output_layer
            else:
                next_layer = self.hidden_layers[i + 1]

            self.hidden_layers[i].error = NeuralNetwork.calculate_layer_error(next_layer, self.hidden_layers[
                i])  # calculating other layers' errors

        L = len(self.hidden_layers) + 2

        for l in range(1, L - 1):  # not last layer, that's why L-1
            for j in range(self.hidden[l - 1]):
                if l == 1:
                    size = self.input
                else:
                    size = self.hidden[l - 2]
                for i in range(size):
                    self.hidden_layers[l - 1].weights[j][i] = self.hidden_layers[l - 1].weights[j][
                                                                  i] - self.partial_derivative_weight(l, j, i)
                self.hidden_layers[l - 1].biases[j] = self.hidden_layers[l - 1].biases[j] - \
                                                      self.hidden_layers[l - 1].error[j]
            self.hidden_layers[l - 1].update_weights()
            self.hidden_layers[l - 1].update_biases()

        for j in range(self.output):  # last layer
            for i in range(self.hidden[L - 3]):
                self.output_layer.weights[j][i] = self.output_layer.weights[j][i] - self.partial_derivative_weight(L, j,
                                                                                                                   i)
            self.output_layer.biases[j] = self.output_layer.biases[j] - self.output_layer.error[j]
        self.output_layer.update_weights()
        self.output_layer.update_biases()

        return

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

    def print_cost(self, needed_output):
        print(NeuralNetwork.mean_squared_error(needed_output, self.output_layer))
        return
