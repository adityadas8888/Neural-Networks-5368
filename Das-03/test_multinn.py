# Kamangar, Farhad
# 1000-123-456
# 2019-10-07
# Assignment-03-02

import numpy as np
import pytest
from multinn import MultiNN
import tensorflow as tf
import keras.datasets

def test_weight_and_biases_dimensions():
    input_dimension = 4
    number_of_layers=3
    number_of_nodes_in_layers_list=list(np.random.randint(3,high=15,size=(number_of_layers,)))
    multi_nn = MultiNN(input_dimension)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        multi_nn.add_layer(number_of_nodes_in_layers_list[layer_number], activation_function=multi_nn.sigmoid)      ## number of nodes
    previous_number_of_outputs=input_dimension
    # assert multi_nn.get_weights_without_biases(0).shape == (4, 12)
    for layer_number in range(len(number_of_nodes_in_layers_list)):
        assert multi_nn.get_weights_without_biases(layer_number).shape == (previous_number_of_outputs,
                                            number_of_nodes_in_layers_list[layer_number])
        z=multi_nn.get_biases(layer_number).shape
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
        np.random.seed(seed=1)
        W=np.random.randn(*W.shape)                         ## random weight values are generated
        multi_nn.set_weights_without_biases(W,layer_number)     ##  random weights values are set up at the weights matrix according to the weights I created
        b = multi_nn.get_biases(layer_number)       
        b = np.random.randn(*b.shape)
        multi_nn.set_biases(b, layer_number)                    ## random biases are set according to the shape of the biases
    X=np.random.randn(number_of_samples,input_dimension)        
    Y=multi_nn.predict(X)

    assert np.allclose(Y.numpy(),np.array( \
        [[0.03266127, 0.50912841, 0.64450596, 0.9950739, 0.85501755],
         [0.10064564, 0.33718693, 0.30543574, 0.94555041, 0.85876801],
         [0.0723957, 0.37974684, 0.32749328, 0.96334569, 0.86322814],
         [0.08510464, 0.36674615, 0.49463606, 0.97151055, 0.79762128],
         [0.09168739, 0.3137653, 0.34286721, 0.96533277, 0.87458543],
         [0.03880329, 0.41823933, 0.33338152, 0.98297688, 0.86816211],
         [0.04737598, 0.4365602, 0.41720532, 0.9806787, 0.79618209]]),rtol=1e-3, atol=1e-3)

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
        np.random.seed(seed=1)
        W = np.random.randn(*W.shape)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number)
        b = np.random.randn(*b.shape)
        multi_nn.set_biases(b, layer_number)
    X = 0.01*np.random.randn(number_of_samples, input_dimension)
    Y = multi_nn.predict(X).numpy()
    # print(np.array2string(np.array(Y), separator=","))
    assert np.allclose(Y, np.array( \
        [[-0.40741104, -3.44566828, -1.76869339, 1.6163245, -2.54072028],
         [-0.43079142, -3.33179871, -1.68949031, 1.60929609, -2.56731804],
         [-0.40121922, -3.38003556, -1.72446422, 1.62425049, -2.53508997],
         [-0.39082312, -3.38185473, -1.70911191, 1.58223456, -2.57304599],
         [-0.32408109, -3.351902, -1.66647768, 1.55353972, -2.5740835],
         [-0.25208801, -3.46616221, -1.72734675, 1.53356758, -2.55233025],
         [-0.54751084, -3.30222443, -1.73142232, 1.72880885, -2.50151569]]), rtol=1e-3, atol=1e-3)


#  def test_train():
#     from keras.datasets import mnist
#     np.random.seed(seed=1)
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
#     # Reshape and Normalize data
#     X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
#     y_train = y_train.flatten().astype(np.int32)
#     input_dimension = X_train.shape[1]
#     indices = list(range(X_train.shape[0]))
#     # np.random.shuffle(indices)
#     number_of_samples_to_use = 500
#     X_train = X_train[indices[:number_of_samples_to_use]]
#     y_train = y_train[indices[:number_of_samples_to_use]]
#     multi_nn = MultiNN(input_dimension)
#     number_of_classes = 10
#     activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
#     number_of_neurons_list = [50, 20, number_of_classes]
#     for layer_number in range(len(activations_list)):
#         multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
#     for layer_number in range(len(multi_nn.weights)):
#         W = multi_nn.get_weights_without_biases(layer_number)
#         np.random.seed(seed=1)
#         W = tf.Variable((np.random.randn(*W.shape) - 0.0) * 0.1, trainable=True)
#         multi_nn.set_weights_without_biases(W, layer_number)
#         b = multi_nn.get_biases(layer_number=layer_number)
#         b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
#         multi_nn.set_biases(b, layer_number)
#     multi_nn.set_loss_function(multi_nn.cross_entropy_loss)
#     percent_error = []
#     for k in range(10):
#         multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
#         percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
#     confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)

#     # print(np.array2string(np.array(percent_error), separator=","))
#     # print(np.array2string(np.array(confusion_matrix), separator=","))
#     assert np.allclose(percent_error, np.array( \
#         [0.488,0.208,0.102,0.02 ,0.002,0.   ,0.   ,0.   ,0.   ,0.   ]), rtol=1e-3, atol=1e-3)
#     assert np.allclose(confusion_matrix, np.array( \
#         [[50., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 66., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 52., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 50., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 52., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 39., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 45., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 52., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 39., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 55.]]), rtol=1e-3, atol=1e-3)
