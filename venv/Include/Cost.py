import NeuralNetworks
import Layer
import math


def squared_mean_error(needed_output, existing_output):
    sum = 0
    for x, y in zip(needed_output.values, existing_output):
        sum = sum + (x - y)**2
    return sum / needed_output.size
