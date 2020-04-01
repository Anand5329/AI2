import Neurons
import Layer as L
import numpy as np

class NeuralNetwork:
    def __init__(self, inp = 1, hidden = [1], output = 1):  #input_layer & output_layer are a Layer. hidden_layers is a list of Layer()
        self.input = inp
        self.hidden = hidden
        self.output = output
        self.input_layer = L.Layer(0, inp)
        self.hidden_layers = []
        self.hidden_layers.append(L.Layer(inp, hidden[0]))
        for i in range(1, len(hidden)):
            self.hidden_layers.append(L.Layer(i-1, i))
        self.output_layer = L.Layer(hidden[len(hidden)-1], output)
        #self.create()
    """
    def create(self):
       self.input_layer

        for i in range(self.hidden):
            neurons = []
            for j in range(self.hidden[i]):
                neurons.append(Neurons.HiddenNeuron())
                if i == 0:
                    neurons[j].previous_layer_size = self.input
                else:
                    neurons[j].previous_layer_size = self.hidden[i-1]
                neurons[j].randomize_weights()
                neurons[j].randomize_bias()
            self.hidden_layers.append(L.Layer(neurons))

        neurons = []
        for i in range(self.ouptut):
            self.output_neurons.append(Neurons.OutputNeuron())
            self.output_neurons[i].previous_layer_size = self.hidden[len(hidden)-1]
            self.output_neurons[i].randomize_weights()
            self.output_neurons[i].randomize_bias()
        self.output_layer = L.Layer(neurons)

        return
        """

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
            np_weights = np.array(weights)
            np_values = np.array(values)
            np_x = np.dot(np_values,np_weights)  # values for the current layer
            self.hidden_layers[i].values = list(np_x)

        # for output layer:
        layer = self.hidden_layers[len(hidden)-1]
        np_weights = np.array(NeuralNetwork.get_weights(self.output_layer))
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
