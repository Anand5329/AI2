import Neurons
import gc
import numpy as np

class Layer:

    def __init__(self, previous_layer_size = 1, size = 1, values = []):
        self.size = size
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
            self.neurons.append(Neurons.Neuron(previous_layer_size = previous_layer_size))
            self.neurons[i].randomize_weights()
            self.neurons[i].randomize_bias()
            if len(values) == size:
                self.neurons[i].value = values[i]
            self.biases.append(self.neurons[i].bias)
            self.values.append(self.neurons[i].value)
            self.weights.append(self.neurons[i].weights)
        self.biases = np.array(self.biases)
        self.values = np.array(self.values)
        self.weights = np.array(self.weights)

        self.biases = self.biases.reshape(self.biases.shape[0],1)
        self.values = self.values.reshape(self.values.shape[0],1)


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
