from .sigmoid import sigmoid

def sigmoid_gradient(matrix):
    """
    return sigmoid gradient function sigmoid(x) * (1 - sigmoid(x))
    """

    return sigmoid(matrix) * (1 - sigmoid(matrix))