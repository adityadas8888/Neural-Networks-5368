# Das, Aditya
# 1001-675-762
# 2019-10-27
# Assignment-03-01

# using tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, activation_function):

        if len(self.weights)==0:
          self.weights.append(tf.random.normal((self.input_dimension,num_nodes)))
          self.biases.append(tf.random.normal((num_nodes,1)))
        else:
          self.weights.append(tf.random.normal((len(self.weights[-1][1]),num_nodes)))
          self.biases.append(tf.random.normal((num_nodes,1)))
        self.activations.append(activation_function)

    def get_weights_without_biases(self, layer_number):
        return self.weights[layer_number]
        


    def get_biases(self, layer_number):
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        self.weights[layer_number]=weights


    def set_biases(self, biases, layer_number):
        self.biases[layer_number]=biases

    def set_loss_function(self, loss_fn):
        self.loss = loss_fn

    def sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out

    def cross_entropy_loss(self, y, y_hat):

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        layers=len(self.weights)
        input_layer = X
        for i in range(layers):
            predicted = tf.matmul(input_layer,self.weights[i])+tf.transpose(self.biases[i])
            trained = self.activations[i](predicted) 
            input_layer=trained
        return trained

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8, regularization_coeff=1e-6):
        no_runs,remaining=self.chunker(X_train,batch_size) 
        for i in range(num_epochs):
            start = 0 
            end=batch_size 
            for j in range(no_runs):
                input_sliced = X_train[start:end,:] 
                target_sliced = y_train[start:end] 
                start=end 
                if i==no_runs-2 and remaining !=0:                                                                  # if the dataset size is not perfectly divisible, it changes the end index 
                    end = X_train.shape[0] 
                else:
                    end = (j+2)*batch_size
                # output=np.dot(self.weights,input_sliced)

                with tf.GradientTape() as tape:
                    predicted = self.predict(input_sliced)
                    loss=self.cross_entropy_loss(target_sliced,predicted)
                    loss_bias,loss_weight=tape.gradient(loss,[self.biases,self.weights])
                for i in range(len(self.weights)):
                    self.weights[i].assign_sub(alpha*loss_weight[i])
                    self.biases[i].assign_sub(alpha*loss_bias[i])


    def chunker(self,X,batch_size):
    # This function decides the number of batches to be made and if the dataset size is not perfectly divisible, it returns the remaining data after all 
    # batches are done and increases the run loop by one
        no_runs = int(X.shape[0]/batch_size)
        remaining = X.shape[0]%batch_size
        if remaining!=0:
            no_runs+=1
        return no_runs,remaining


    def calculate_percent_error(self, X, y):

        percent_error=0
        pred_y=self.predict(X)
        num_ofsamples,num_of_classes=np.shape(pred_y)
        print(num_ofsamples)
        target_class_predicted=np.argmax(pred_y,axis=1)
        commons=np.sum(np.where(target_class_predicted==y,0,1))
        # commons=sum(i != j for i, j in zip(target_class_predicted, y))
        
        percent_error=commons
        return percent_error/num_ofsamples
    
    
    def one_hot(self,Y,number_of_classes):
    # This function does one hot encoding on the input matrix
        Y = np.squeeze(np.eye(number_of_classes)[Y.reshape(-1)])
        return Y

    def calculate_confusion_matrix(self, X, y):
        
        X=self.predict(X)
        rows,number_of_classes=np.shape(X)
        y = self.one_hot(y,number_of_classes)
        confusion_matrix = np.zeros((number_of_classes,number_of_classes)) 
        print(y.shape)
        print(X.shape)
        for i in range(rows):
            actual_index = np.argmax(X[i,:], axis=0) 
            target_index = np.argmax(y[i,:], axis=0) 
            confusion_matrix[target_index][actual_index] += 1
        return confusion_matrix

if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape and Normalize data
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5   # to make it compatible with the 784 size 
    y_train = y_train.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))         ## making a list of all the indices of the array
    # np.random.shuffle(indices)
    number_of_samples_to_use = 500
    X_train = X_train[indices[:number_of_samples_to_use]]                   ## splicing the training data
    y_train = y_train[indices[:number_of_samples_to_use]]
    multi_nn = MultiNN(input_dimension)                             ## its sending 784 to the class
    number_of_classes = 10
    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
    number_of_neurons_list = [50, 20, number_of_classes]                ##?
    for layer_number in range(len(activations_list)):
        multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
    for layer_number in range(len(multi_nn.weights)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = tf.Variable((np.random.randn(*W.shape)) * 0.1, trainable=True)
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
    print("Percent error: ", np.array2string(np.array(percent_error), separator=","))
    print("************* Confusion Matrix ***************\n", np.array2string(confusion_matrix, separator=","))
