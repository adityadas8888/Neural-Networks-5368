# Kamangar, Farhad
# 1000-123-456
# 2019-09-22
# Assignment-01-02

import numpy as np
import pytest
from perceptron import Perceptron


def test_weight_dimension():
    input_dimensions = 4
    number_of_classes = 9
    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes)
    assert model.weights.ndim == 2 and \
           model.weights.shape[0] == number_of_classes and \
           model.weights.shape[1] == (input_dimensions + 1)


def test_weight_initialization():
    input_dimensions = 2
    number_of_classes = 5
    model = Perceptron(input_dimensions=2, number_of_classes=number_of_classes, seed=1)
    assert model.weights.ndim == 2 and model.weights.shape[0] == number_of_classes and model.weights.shape[
        1] == input_dimensions + 1
    weights = np.array([[1.62434536, -0.61175641, -0.52817175],
                        [-1.07296862, 0.86540763, -2.3015387],
                        [1.74481176, -0.7612069, 0.3190391],
                        [-0.24937038, 1.46210794, -2.06014071],
                        [-0.3224172, -0.38405435, 1.13376944]])
    np.testing.assert_allclose(model.weights, weights, rtol=1e-3, atol=1e-3)
    model.initialize_all_weights_to_zeros()
    assert np.array_equal(model.weights, np.zeros((number_of_classes, input_dimensions + 1)))


def test_predict():
    input_dimensions = 2
    number_of_classes = 2
    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    Y_hat = model.predict(X_train)
    assert np.array_equal(Y_hat, np.array([[1, 1, 1, 1], [1, 0, 1, 1]]))


def test_error_calculation():
    input_dimensions = 2
    number_of_classes = 2
    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    model.initialize_all_weights_to_zeros()
    error = []
    for k in range(20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        print(model.calculate_percent_error(X_train, Y_train))
        error.append(model.calculate_percent_error(X_train, Y_train))
    np.testing.assert_allclose(error,
                               [0.25, 0.5, 0.5, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0], rtol=1e-3, atol=1e-3)
