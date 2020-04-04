import Neurons
import Layer as L
import numpy as np
import math

class NeuralNetwork:

    def __init__(self, inp = 1, hidden = [1], output = 1, inputs = []):
        self.input = inp
        self.hidden = hidden
        self.output = output
        self.input_layer = L.Layer(0, inp, values = inputs)
        self.hidden_layers = []
        self.hidden_layers.append(L.Layer(inp, hidden[0]))
        for i in range(1, len(hidden)):
            self.hidden_layers.append(L.Layer(hidden[i-1], hidden[i]))
        self.output_layer = L.Layer(hidden[len(hidden)-1], output)


    def forward_propagation(self):  # forward propagation (calculating values)
        # for hidden layers:
        for i in range(len(self.hidden)):
            if i == 0:
                layer = self.input_layer
            else:
                layer = self.hidden_layers[i-1]
            self.hidden_layers[i].values = NeuralNetwork.feed_forward(layer, self.hidden_layers[i])
            self.hidden_layers[i].update_values()

        # for output layer:
        layer = self.hidden_layers[len(self.hidden)-1]
        self.output_layer.values = NeuralNetwork.feed_forward(layer, self.output_layer)
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
        return 1/(1+math.e**(-x))

    @staticmethod
    def feed_forward(previous_layer, current_layer):
        biases = current_layer.biases
        values = previous_layer.values
        weights = NeuralNetwork.get_weights(current_layer)
        np_weights = np.array(weights).transpose()
        np_values = np.array(values)
        np_x = np.dot(np_values, np_weights)

        for i in range(len(np_x)):
            np_x[i] = np_x[i] + biases[i]
            np_x[i] = NeuralNetwork.activation_sigmoid(np_x[i])
        return list(np_x)
