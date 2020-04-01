import random
class Neuron:
    def __init__(self, bias=0, value=0, weights = [], previous_layer_size = 0):
        self.bias = bias
        self.value = value
        self.weights = weights
        self.previous_layer_size = previous_layer_size

    def __str__(self):
        return str(self.value) + '\n' + str(self.bias) + '\n' + str(self.previous_layer_size)

    def randomize_weights(self):
        if self.previous_layer_size > 0:
            for i in range(self.previous_layer_size):
                self.weights.append(random.random())
            return True
        else:
            return False

    def randomize_bias(self):
        self.bias = random.random()
        return

class InputNeuron(Neuron):
    def __init__(self, bias=0, value=0, weights = [], previous_layer_size = 0):
        super().__init__(bias,value,weights,previous_layer_size)

class OutputNeuron(Neuron):
    def __init__(self, bias=0, value=0, weights = [], previous_layer_size = 0):
        super().__init__(bias,value,weights,previous_layer_size)

class HiddenNeuron(Neuron):
    def __init__(self, bias=0, value=0, weights = [], previous_layer_size = 0):
        super().__init__(bias,value,weights,previous_layer_size)