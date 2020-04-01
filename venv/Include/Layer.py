import Neurons

class Layer:
    def __init__(self, neurons = []):
        self.neurons = neurons
        self.biases = []
        self.values = []

        for i in range(len(neurons)):
            self.biases.append(neurons[i].bias)
            self.values.append(neurons[i].value)