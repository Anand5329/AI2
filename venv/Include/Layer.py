import Neurons
import gc

class Layer:
    def __init__(self, previous_layer_size = 1, size = 1, values = []):
        self.size = size
        self.biases = []
        self.values = []
        self.neurons = []
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
