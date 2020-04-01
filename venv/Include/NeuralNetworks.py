import Neurons
import Layer

class NeuralNetwork:
    def __init__(self, inp = 1, hidden = [1], output = 1, input_layer = Layer(), hidden_layers = [Layer()], output_layer = Layer()):
        self.input = inp
        self.hidden = hidden
        self.output = output
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.create(self)

    def create(self):
        neurons = []
        for i in range(self.inp):
            neurons.append(Neurons.InputNeuron())
        self.input_layer = Layer(neurons)

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
            self.hidden_layers.append(Layer(neurons))

        neurons = []
        for i in range(self.ouptut):
            self.output_neurons.append(Neurons.OutputNeuron())
            self.output_neurons[i].previous_layer_size = self.hidden[len(hidden)-1]
            self.output_neurons[i].randomize_weights()
            self.output_neurons[i].randomize_bias()
        self.output_layer = Layer(neurons)

        return

    def calculate_values(self):#forward propogation
        for i in range(self.hidden):
            for j in range(self.hidden[i]):
                layer
                if i == 0:
                    layer = self.input_layer
                else:
                    layer = self.hidden_layers[i-1]

