import Neurons

class NeuralNetwork:
    def __init__(self, inp = 1, hidden = [1], output = 1, input_neurons = [], hidden_neurons = [[]], output_neurons = []):
        self.input = inp
        self.hidden = hidden
        self.output = output
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.create(self)

    def create(self):
        for i in range(self.inp):
            self.input_neurons.append(Neurons.InputNeuron())
            self.input_neurons[i].next_layer_size = self.hidden[0]

        for i in range(self.hidden):
            for j in range(self.hidden[i]):

                self.hidden_neurons[i].append(Neurons.HiddenNeuron())
                if i+1 < len(hidden):
                    self.hidden_neurons[i][j].next_layer_size = self.hidden[i+1]
                else:
                    self.hidden_neurons[i][j].next_layer_size = self.output

        for i in range(self.ouptut):
            self.output_neurons.append(Neurons.OutputNeuron())
            self.output_neurons[i].next_layer_size = 0
        return

