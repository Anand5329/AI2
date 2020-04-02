import Neurons
import NeuralNetworks as NN
import numpy as np
import Layer as L

ann = NN.NeuralNetwork(inp = 5, hidden = [2, 2], output = 3, inputs = [1, 2, 3, 4, 5])
ann.forward_propagation()
pass