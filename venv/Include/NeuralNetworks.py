import Neurons
import Layer as L
import numpy as np

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
            biases = layer.biases
            values = layer.values
            weights = NeuralNetwork.get_weights(self.hidden_layers[i])
            np_weights = np.array(weights).transpose()
            np_values = np.array(values)
            np_x = np.dot(np_values, np_weights)  # values for the current layer
            self.hidden_layers[i].values = list(np_x)

        # for output layer:
        layer = self.hidden_layers[len(self.hidden)-1]
        np_weights = np.array(NeuralNetwork.get_weights(self.output_layer)).transpose()
        np_values = np.array(layer.values)
        np_x = np.dot(np_values, np_weights)
        self.output_layer.values = list(np_x)
        return

    @staticmethod
    def get_weights(layer):
        weights = []
        for i in range(len(layer.neurons)):
            weights.append(layer.neurons[i].weights)
        return weights
