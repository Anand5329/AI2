class Neuron:
    def __init__(self, bias=0, value=0, weights = [], next_layer_size = 0):
        self.bias = bias
        self.value = value
        self.weights = weights
        self.next_layer_size = next_layer_size

class InputNeuron(Neuron):
    def __init__(self):
        super().init(self)

class OutputNeuron(Neuron):
    def __init__(self):
        super().__init__(self)

class HiddenNeuron(Neuron):
    def __init__(self):
        super().__init__(self)