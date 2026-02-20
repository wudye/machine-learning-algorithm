import numpy as np

def sigmoid(matrix):
    """
    return sigmoid function 1 / (1 + exp(-x))
    """

    return 1 / (1 + np.exp(-matrix))