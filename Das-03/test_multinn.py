# Kamangar, Farhad
# 1000-123-456
# 2019-10-07
# Assignment-03-02

import numpy as np
import pytest
from multinn import MultiNN
import tensorflow as tf

def test_weight_and_biases_dimensions():
    input_dimension = 4
    number_of_layers=3
    number_of_nodes_in_layers_list=list(np.random.randint(3,high=15,size=(number_of_layers,)))
    multi_nn = MultiNN(input_dimension)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        multi_nn.add_layer(number_of_nodes_in_layers_list[layer_number], activation_function=multi_nn.sigmoid)
    previous_number_of_outputs=input_dimension
    # assert multi_nn.get_weights_without_biases(0).shape == (4, 12)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        assert multi_nn.get_weights_without_biases(layer_number).shape == (previous_number_of_outputs,
                                            number_of_nodes_in_layers_list[layer_number])
        previous_number_of_outputs=number_of_nodes_in_layers_list[layer_number]

def test_get_and_set_weight_and_biases():
    input_dimension = 4
    number_of_layers=3
    number_of_nodes_in_layers_list=list(np.random.randint(3,high=15,size=(number_of_layers,)))
    multi_nn = MultiNN(input_dimension)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        multi_nn.add_layer(number_of_nodes_in_layers_list[layer_number], activation_function=multi_nn.sigmoid)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        W=multi_nn.get_weights_without_biases(layer_number)
        W=np.random.randn(*W.shape)
        multi_nn.set_weights_without_biases(W,layer_number)
        b = multi_nn.get_biases(layer_number)
        b = np.random.randn(*b.shape)
        multi_nn.set_biases(b, layer_number)
        assert np.array_equal(W,multi_nn.get_weights_without_biases(layer_number))
        assert np.array_equal(b,multi_nn.get_biases(layer_number))

def test_get_and_set_weight_and_biases():
    input_dimension = 4
    number_of_layers=3
    number_of_nodes_in_layers_list=list(np.random.randint(3,high=15,size=(number_of_layers,)))
    multi_nn = MultiNN(input_dimension)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        multi_nn.add_layer(number_of_nodes_in_layers_list[layer_number], activation_function=multi_nn.sigmoid)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        W=multi_nn.get_weights_without_biases(layer_number)
        W=np.random.randn(*W.shape)
        multi_nn.set_weights_without_biases(W,layer_number)
        b = multi_nn.get_biases(layer_number)
        b = np.random.randn(*b.shape)
        multi_nn.set_biases(b, layer_number)
        assert np.array_equal(W,multi_nn.get_weights_without_biases(layer_number))
        assert np.array_equal(b,multi_nn.get_biases(layer_number))
def test_predict():
    np.random.seed(seed=1)
    input_dimension = 4
    number_of_samples=7
    number_of_layers=3
    number_of_nodes_in_layers_list=list(np.random.randint(3,high=15,size=(number_of_layers,)))
    number_of_nodes_in_layers_list[-1]=5
    multi_nn = MultiNN(input_dimension)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        multi_nn.add_layer(number_of_nodes_in_layers_list[layer_number], activation_function=multi_nn.sigmoid)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        W=multi_nn.get_weights_without_biases(layer_number)
        W=np.random.randn(*W.shape)
        multi_nn.set_weights_without_biases(W,layer_number)
        b = multi_nn.get_biases(layer_number)
        b = np.random.randn(*b.shape)
        multi_nn.set_biases(b, layer_number)
    X=np.random.randn(number_of_samples,input_dimension)
    Y=multi_nn.predict(X)
    assert np.allclose(Y.numpy(),np.array( \
        [[0.8325159, 0.66208966, 0.5367257, 0.78715318, 0.61613198],
         [0.84644084, 0.61953579, 0.66037588, 0.80119257, 0.5786144],
         [0.76684143, 0.66854621, 0.44115054, 0.79724872, 0.78264913],
         [0.79286549, 0.54751305, 0.53022196, 0.79516922, 0.60459471],
         [0.73018733, 0.59293959, 0.24156932, 0.68624113, 0.64357039],
         [0.77498083, 0.64774851, 0.39803303, 0.74039891, 0.69847998],
         [0.79041249, 0.6593798, 0.38533034, 0.80670202, 0.62567229]]),rtol=1e-3, atol=1e-3)

def test_predict_02():
    np.random.seed(seed=1)
    input_dimension = 4
    number_of_samples = 7
    number_of_layers = 2
    number_of_nodes_in_layers_list = list(np.random.randint(3, high=15, size=(number_of_layers,)))
    number_of_nodes_in_layers_list[-1] = 5
    multi_nn = MultiNN(input_dimension)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        multi_nn.add_layer(number_of_nodes_in_layers_list[layer_number], activation_function=multi_nn.linear)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = np.random.randn(*W.shape)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number)
        b = np.random.randn(*b.shape)
        multi_nn.set_biases(b, layer_number)
    X = 0.01*np.random.randn(number_of_samples, input_dimension)
    Y = multi_nn.predict(X).numpy()
    assert np.allclose(Y, np.array( \
        [[ 0.21554626, 5.74564102, 2.20448341,-2.34530146,-2.42393435],
     [ 0.1146438 , 5.67343621, 2.28304649,-2.49450112,-2.6851891 ],
     [ 0.14486076, 5.70363412, 2.28009343,-2.44228025,-2.67070783],
     [ 0.16074679, 5.71585438, 2.22879188,-2.41489906,-2.57137539],
     [ 0.1757518 , 5.77638577, 2.14903762,-2.36006981,-2.4708835 ],
     [ 0.14457726, 5.73368867, 2.23076791,-2.42033096,-2.63129168],
     [ 0.15044344, 5.71698563, 2.28266725,-2.41933965,-2.72953199]]), rtol=1e-3, atol=1e-3)

    def test_predict_02():
        np.random.seed(seed=1)
        input_dimension = 4
        number_of_samples = 7
        number_of_layers = 2
        number_of_nodes_in_layers_list = list(np.random.randint(3, high=15, size=(number_of_layers,)))
        number_of_nodes_in_layers_list[-1] = 5
        multi_nn = MultiNN(input_dimension)
        for layer_number in range(len(number_of_nodes_in_layers_list)):
            multi_nn.add_layer(number_of_nodes_in_layers_list[layer_number], activation_function=multi_nn.linear)
        for layer_number in range(len(number_of_nodes_in_layers_list)):
            W = multi_nn.get_weights_without_biases(layer_number)
            W = np.random.randn(*W.shape)
            multi_nn.set_weights_without_biases(W, layer_number)
            b = multi_nn.get_biases(layer_number)
            b = np.random.randn(*b.shape)
            multi_nn.set_biases(b, layer_number)
        X = 0.01 * np.random.randn(number_of_samples, input_dimension)
        Y = multi_nn.predict(X).numpy()
        assert np.allclose(Y, np.array( \
            [[0.21554626, 5.74564102, 2.20448341, -2.34530146, -2.42393435],
             [0.1146438, 5.67343621, 2.28304649, -2.49450112, -2.6851891],
             [0.14486076, 5.70363412, 2.28009343, -2.44228025, -2.67070783],
             [0.16074679, 5.71585438, 2.22879188, -2.41489906, -2.57137539],
             [0.1757518, 5.77638577, 2.14903762, -2.36006981, -2.4708835],
             [0.14457726, 5.73368867, 2.23076791, -2.42033096, -2.63129168],
             [0.15044344, 5.71698563, 2.28266725, -2.41933965, -2.72953199]]), rtol=1e-3, atol=1e-3)
def test_train():
    from tensorflow.keras.datasets import mnist
    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape and Normalize data
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    # np.random.shuffle(indices)
    number_of_samples_to_use = 500
    X_train = X_train[indices[:number_of_samples_to_use]]
    y_train = y_train[indices[:number_of_samples_to_use]]
    multi_nn = MultiNN(input_dimension)
    number_of_classes = 10
    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
    number_of_neurons_list = [50, 20, number_of_classes]
    for layer_number in range(len(activations_list)):
        multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
    for layer_number in range(len(multi_nn.weights)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = tf.Variable((np.random.randn(*W.shape) - 0.0) * 0.1, trainable=True)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number=layer_number)
        b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
        multi_nn.set_biases(b, layer_number)
    multi_nn.set_loss_function(multi_nn.cross_entropy_loss)
    percent_error = []
    for k in range(10):
        multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
        percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
    confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)
    assert np.allclose(percent_error, np.array( \
        [0.406,0.214,0.058,0.016,0.   ,0.   ,0.   ,0.   ,0.   ,0.   ]), rtol=1e-3, atol=1e-3)
    assert np.allclose(confusion_matrix, np.array( \
        [[50., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 66., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 52., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 50., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 52., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 39., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 45., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 52., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 39., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 55.]]), rtol=1e-3, atol=1e-3)
