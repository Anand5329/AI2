class Neuron:
    def __init__(self, bias=0, value=0, weights = [], next_layer_size = 0):
        self.bias = bias
        self.value = value
        self.weights = weights
        self.next_layer_size = next_layer_size

    def __str__(self):
        return str(self.value) + '\n' + str(self.bias) + '\n' + str(self.next_layer_size)

class InputNeuron(Neuron):
    def __init__(self, bias=0, value=0, weights = [], next_layer_size = 0):
        super().__init__(bias,value,weights,next_layer_size)

class OutputNeuron(Neuron):
    def __init__(self, bias=0, value=0, weights = [], next_layer_size = 0):
        super().__init__(bias,value,weights,next_layer_size)

class HiddenNeuron(Neuron):
    def __init__(self, bias=0, value=0, weights = [], next_layer_size = 0):
        super().__init__(bias,value,weights,next_layer_size)