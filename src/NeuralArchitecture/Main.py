import src.NeuralArchitecture.NeuralNetworks as NN
import src.NeuralArchitecture.Neurons as N
import src.NeuralArchitecture.Layer as L
import numpy as np
import random

inputs = []
for i in range(784):
    inputs.append(random.uniform(0, 1))
ann = NN.NeuralNetwork(inp = 784, hidden = [16, 16], output = 10, inputs = inputs)
ann.forward_propagation()
needed_output = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
print(NN.NeuralNetwork.mean_squared_error(needed_output, ann.output_layer.values))
pass