import src.NeuralArchitecture.NeuralNetworks as NN
import src.NeuralArchitecture.Neurons as N
import src.NeuralArchitecture.Layer as L
import numpy as np
import random

inputs = [[]]
for j in range(0,100):
    inputs.append([])
    for i in range(784):
        inputs[j].append(random.uniform(0, 1))
ann = NN.NeuralNetwork(inp = 784, hidden = [16, 16], output = 10)
needed_output = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
outputs = []
for j in range(0,100):
    outputs.append(needed_output)
training_data = [inputs]
training_data.append(outputs)
ann.input_training_data(training_data)
ann.stochastic_GD(10)
print(ann.output_layer.values)
pass