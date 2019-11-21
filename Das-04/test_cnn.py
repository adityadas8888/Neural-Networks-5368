# Das, Aditya
# 1001-675-672
# 2019-11_02
# Assignment-04-02

import pytest
import numpy as np
from cnn import CNN
import os


def test_add_input_layer():
    model = CNN()
    out=model.add_input_layer(shape=(256, 256, 3), name="input0")
    # no tests for this?
    assert True

def test_append_conv2d_layer():
    model = CNN()
    model.add_input_layer(shape=(256, 256, 3), name="input0")
    model.append_conv2d_layer(10, (3, 3), activation='relu')
    input = np.zeros((20, 256, 256, 3))
    out = model.predict(input)
    assert (out.shape == (20,256,256,10))
def test_append_maxpooling2d_layer():
    model = CNN()
    model.add_input_layer(shape=(256, 256, 3), name="input0")
    model.append_maxpooling2d_layer(pool_size=(2, 2), padding="same", strides=2, name='pooling')
    input = np.zeros((10, 256, 256, 3))
    out = model.predict(input)
    assert (out.shape == (10, 128, 128, 3))

def test_add_flatten_layer():
    model = CNN()
    model.add_input_layer(shape=(256, 256, 3), name="input0")
    model.append_flatten_layer(name='flatten')
    input = np.zeros((10, 256, 256, 3))
    out = model.predict(input)
    assert out.shape == (10, 256 * 256 * 3)

def test_append_dense_layer():
    model = CNN()
    model.add_input_layer(shape=(256*256*3), name="input0")
    model.append_dense_layer(num_nodes=100, activation='relu')
    input = np.zeros((10, 256*256*3))
    result = model.predict(input)
    assert result.shape == (10,100)

def test_get_weights_without_biases_1():
    my_cnn = CNN()
    input_size=np.random.randint(32,100)
    number_of_dense_layers=np.random.randint(2,10)
    my_cnn.add_input_layer(shape=input_size,name="input")
    previous_nodes=input_size
    for k in range(number_of_dense_layers):
        number_of_nodes = np.random.randint(3, 100)
        kernel_size= np.random.randint(3,9)
        my_cnn.append_dense_layer(num_nodes=number_of_nodes)
        actual = my_cnn.get_weights_without_biases(layer_number=k+1)
        assert actual.shape ==  (previous_nodes,number_of_nodes)
        previous_nodes=number_of_nodes

def test_get_weights_without_biases_2():
    my_cnn = CNN()
    image_size=(np.random.randint(32,100),np.random.randint(20,100),np.random.randint(3,10))
    number_of_conv_layers=np.random.randint(2,10)
    my_cnn.add_input_layer(shape=image_size,name="input")
    previous_depth=image_size[2]
    for k in range(number_of_conv_layers):
        number_of_filters = np.random.randint(3, 100)
        kernel_size= np.random.randint(3,9)
        my_cnn.append_conv2d_layer(num_of_filters=number_of_filters,
                                   kernel_size=(kernel_size,kernel_size),
                                   padding="same", activation='linear')

        actual = my_cnn.get_weights_without_biases(layer_number=k+1)
        assert actual.shape == (kernel_size,kernel_size,previous_depth,number_of_filters)
        previous_depth=number_of_filters
    actual = my_cnn.get_weights_without_biases(layer_number=0)
    assert actual is None
def test_get_weights_without_biases_3():
    my_cnn = CNN()
    image_size=(np.random.randint(32,100),np.random.randint(20,100),np.random.randint(3,10))
    number_of_conv_layers=np.random.randint(2,10)
    my_cnn.add_input_layer(shape=image_size,name="input")
    previous_depth=image_size[2]
    for k in range(number_of_conv_layers):
        number_of_filters = np.random.randint(3, 100)
        kernel_size= np.random.randint(3,9)
        my_cnn.append_conv2d_layer(num_of_filters=number_of_filters,
                                   kernel_size=(kernel_size,kernel_size),
                                   padding="same", activation='linear')

        actual = my_cnn.get_weights_without_biases(layer_number=k+1)
        assert actual.shape == (kernel_size,kernel_size,previous_depth,number_of_filters)
        previous_depth=number_of_filters
    actual = my_cnn.get_weights_without_biases(layer_number=0)
    assert actual is None
    pool_size = np.random.randint(2, 5)
    my_cnn.append_maxpooling2d_layer(pool_size=pool_size,padding="same",
                                     strides=2,name="pool1")
    actual=my_cnn.get_weights_without_biases(layer_name="pool1")
    assert actual is None
    my_cnn.append_flatten_layer(name="flat1")
    actual=my_cnn.get_weights_without_biases(layer_name="flat1")
    assert actual is None
    my_cnn.append_dense_layer(num_nodes=10)
    number_of_dense_layers = np.random.randint(2, 10)
    previous_nodes = 10
    for k in range(number_of_dense_layers):
        number_of_nodes = np.random.randint(3, 100)
        kernel_size = np.random.randint(3, 9)
        my_cnn.append_dense_layer(num_nodes=number_of_nodes)
        actual = my_cnn.get_weights_without_biases(layer_number=k+number_of_conv_layers+4 )
        # assert actual.shape == (previous_nodes, number_of_nodes)
        previous_nodes = number_of_nodes

def test_get_biases_1():
    my_cnn = CNN()
    input_size=np.random.randint(32,100)
    number_of_dense_layers=np.random.randint(2,10)
    my_cnn.add_input_layer(shape=input_size,name="input")
    previous_nodes=input_size
    for k in range(number_of_dense_layers):
        number_of_nodes = np.random.randint(3, 100)
        kernel_size= np.random.randint(3,9)
        my_cnn.append_dense_layer(num_nodes=number_of_nodes)
        actual = my_cnn.get_biases(layer_number=k+1)
        assert (actual.shape ==  (number_of_nodes,)) or (actual.shape ==  (number_of_nodes,1))
        previous_nodes=number_of_nodes

def test_get_biases_2():
    my_cnn = CNN()
    image_size=(np.random.randint(32,100),np.random.randint(20,100),np.random.randint(3,10))
    number_of_conv_layers=np.random.randint(2,10)
    my_cnn.add_input_layer(shape=image_size,name="input")
    previous_depth=image_size[2]
    for k in range(number_of_conv_layers):
        number_of_filters = np.random.randint(3, 100)
        kernel_size= np.random.randint(3,9)
        my_cnn.append_conv2d_layer(num_of_filters=number_of_filters,
                                   kernel_size=(kernel_size,kernel_size),
                                   padding="same", activation='linear')

        actual = my_cnn.get_biases(layer_number=k+1)
        assert (actual.shape == (number_of_filters,)) or (actual.shape == (number_of_filters,1))
        previous_depth=number_of_filters
    actual = my_cnn.get_biases(layer_number=0)
    assert actual is None

def test_set_weights_without_biases():
    my_cnn = CNN()
    image_size = (np.random.randint(32, 100), np.random.randint(20, 100), np.random.randint(3, 10))
    number_of_conv_layers = np.random.randint(2, 10)
    my_cnn.add_input_layer(shape=image_size, name="input")
    previous_depth = image_size[2]
    for k in range(number_of_conv_layers):
        number_of_filters = np.random.randint(3, 100)
        kernel_size = np.random.randint(3, 9)
        my_cnn.append_conv2d_layer(num_of_filters=number_of_filters,
                                   kernel_size=(kernel_size, kernel_size),
                                   padding="same", activation='linear')

        w = my_cnn.get_weights_without_biases(layer_number=k + 1)
        w_set=np.full_like(w,0.2)
        my_cnn.set_weights_without_biases(w_set,layer_number=k+1)
        w_get = my_cnn.get_weights_without_biases(layer_number=k + 1)
        assert w_get.shape == w_set.shape
        previous_depth = number_of_filters
    pool_size = np.random.randint(2, 5)
    my_cnn.append_maxpooling2d_layer(pool_size=pool_size, padding="same",
                                     strides=2, name="pool1")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=10)
    number_of_dense_layers = np.random.randint(2, 10)
    previous_nodes = 10
    for k in range(number_of_dense_layers):
        number_of_nodes = np.random.randint(3, 100)
        kernel_size = np.random.randint(3, 9)
        my_cnn.append_dense_layer(num_nodes=number_of_nodes)

        w = my_cnn.get_weights_without_biases(layer_number=k + number_of_conv_layers + 4)
        w_set = np.full_like(w, 0.8)
        my_cnn.set_weights_without_biases(w_set, layer_number=k + number_of_conv_layers + 4)
        w_get = my_cnn.get_weights_without_biases(layer_number=k + number_of_conv_layers + 4)
        assert w_get.shape == w_set.shape
        previous_nodes = number_of_nodes

def test_load_and_save_model():
    # Note: This test may take a long time to load the data
    my_cnn = CNN()
    my_cnn.load_a_model(model_name="VGG19")
    # my_cnn.append_dense_layer(num_nodes=10)
    w=my_cnn.get_weights_without_biases(layer_name="block5_conv4")
    assert w.shape == (3,3,512,512)
    w = my_cnn.get_weights_without_biases(layer_number=-1)
    assert w.shape == (4096,1000)
    my_cnn.append_dense_layer(num_nodes=10)
    path = os.getcwd()
    file_path=os.path.join(path,"my_model.h5")
    my_cnn.save_model(model_file_name=file_path)
    my_cnn.load_a_model(model_name="VGG16")
    w = my_cnn.get_weights_without_biases(layer_name="block4_conv1")
    assert w.shape == (3, 3, 256, 512)
    my_cnn.load_a_model(model_file_name=file_path)
    os.remove(file_path)
    w = my_cnn.get_weights_without_biases(layer_number=-1)
    assert w.shape == (1000,10)

def test_predict():
    # some of these may be duplicated
    X = np.float32([[0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5]])
    X = np.float32([[0.1, 0.2, 0.3, 0.4, 0.5, 0,0,0,0,0]])
    X = np.float32([np.linspace(0,10,num=10)])
    # X = np.float32([[0.1, 0.2]])
    my_cnn = CNN()
    my_cnn.add_input_layer(shape=(10,), name="input0")
    my_cnn.append_dense_layer(num_nodes=5, activation='linear', name="layer1")
    w = my_cnn.get_weights_without_biases(layer_name="layer1")
    w_set = np.full_like(w, 2)
    my_cnn.set_weights_without_biases(w_set, layer_name="layer1")
    b=my_cnn.get_biases(layer_name="layer1")
    b_set= np.full_like(b, 2)
    b_set[0]=b_set[0]*2
    my_cnn.set_biases(b_set, layer_name="layer1")

    # my_cnn.append_dense_layer(num_nodes=5, activation='linear', name="layer12")
    actual = my_cnn.predict(X)
    assert np.array_equal(actual,np.array([[104., 102., 102., 102., 102.]]))
def test_remove_last_layer():
    from tensorflow.keras.datasets import cifar10
    batch_size = 32
    num_classes = 10
    epochs = 100
    data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    number_of_train_samples_to_use = 100
    X_train = X_train[0:number_of_train_samples_to_use, :]
    y_train = y_train[0:number_of_train_samples_to_use]
    my_cnn=CNN()
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=(3,3),padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=10,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=2,activation="relu",name="dense2")
    out=my_cnn.predict(X_train)
    assert out.shape == (number_of_train_samples_to_use, 2)
    my_cnn.remove_last_layer()
    out = my_cnn.predict(X_train)
    assert out.shape==(number_of_train_samples_to_use,10)

