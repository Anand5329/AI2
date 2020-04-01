import Neurons
import gc

class Layer:
    def __init__(self, previous_layer_size = 1, size = 1):
        self.size = size
        self.biases = []
        self.values = []
        self.neurons = []
        for i in range(self.size):
            # TODO fix bug about weights not resetting. Acting like static field.
            self.neurons.append(Neurons.Neuron(previous_layer_size = previous_layer_size))
            self.neurons[i].randomize_weights()
            self.neurons[i].randomize_bias()
            self.biases.append(self.neurons[i].bias)
            self.values.append(self.neurons[i].value)
