import Neurons
import gc
import numpy as np

class Layer:

    def __init__(self, previous_layer_size = 1, size = 1, values = []):
        self.size = size
        self.previous_layer_size = previous_layer_size
        self.biases = []
        self.values = []
        self.neurons = []
        self.z = []
        self.weights = []
        self.error = []
        self.grad = {"W" : None, "b": None}
        if len(values) != size and len(values) != 0:
            print("Input size mismatch")
        for i in range(self.size):
            self.neurons.append(Neurons.Neuron(previous_layer_size = self.previous_layer_size))
            if len(values) == size:
                self.neurons[i].value = values[i]
        if self.previous_layer_size != 0:
            self.initialize_parameters()


    def iter_gen(self):
        for i in self.neurons:
            yield i

    def update_values(self):
        for n, v in zip(self.neurons, self.values):
            n.value = v
        return

    def update_weights(self):
        for w, n in zip(self.weights, self.neurons):
            n.weights = w
        return

    def update_biases(self):
        for b, n in zip(self.biases, self.neurons):
            n.bias = b
        return

    def initialize_parameters(self):
        self.weights = np.random.randn(self.size, self.previous_layer_size)/np.sqrt(self.previous_layer_size)
        self.biases = np.zeros((self.size,1))
        self.update_weights()
        self.update_biases()