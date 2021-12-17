import numpy as np
from math import *


def sigmoid(x):
    return np.divide(1, (1 + np.power(e, -x)))


def sigmoid_prime(x):
    return np.multiply(sigmoid(x), np.subtract(1, sigmoid(x)))


def random_weights(sizes):
    return [xavier_initialization(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def zeros_weights(sizes):
    return [np.zeros((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]


def zeros_biases(list):
    return [np.zeros(i) for i in list]


def create_batches(data, labels, batch_size):
    return [ (data[i * batch_size : (i + 1) * batch_size], labels[i * batch_size : (i + 1) * batch_size])
             for i in range((len(data) // batch_size) + 1)
             if (i * batch_size != len(data))]

def add_elementwise(list1, list2):
    return [list1[i] + list2[i] for i in range(len(list1))]


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))