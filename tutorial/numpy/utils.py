import numpy as np

def softmax(x, dim=-1):
    """
    softmax for one element of a vector: softmax(x_i) = e^x_i / sum_j(e^x_j)
    If softmax has multiple dimension, perform the operation on the last dimension (axis=-1).
    Standard reduction operations remove the dimension they reduce (keepdims=True will keep it).

    x: vector of size j
    return: vector with softmax applied to every element
    """
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True)) # max prevent overflow
    return e_x / np.sum(e_x, axis=dim, keepdims=True)