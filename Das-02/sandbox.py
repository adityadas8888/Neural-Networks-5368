# Das, Aditya
# 1001-675-762
# 2019-10-07
# Assignment-02-01

import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import math


def display_images(images):
	# This function displays images on a grid.
	# Farhad Kamangar Sept. 2019
	number_of_images=images.shape[0]
	number_of_rows_for_subplot=int(np.sqrt(number_of_images))
	number_of_columns_for_subplot=int(np.ceil(number_of_images/number_of_rows_for_subplot))
	for k in range(number_of_images):
		plt.subplot(number_of_rows_for_subplot,number_of_columns_for_subplot,k+1)
		plt.imshow(images[k], cmap=plt.get_cmap('gray'))
		# plt.imshow(images[k], cmap=pyplot.get_cmap('gray'))
	plt.show()

def display_numpy_array_as_table(input_array):
	# This function displays a 1d or 2d numpy array (matrix).
	# Farhad Kamangar Sept. 2019
	if input_array.ndim==1:
		num_of_columns,=input_array.shape
		temp_matrix=input_array.reshape((1, num_of_columns))
	elif input_array.ndim>2:
		print("Input matrix dimension is greater than 2. Can not display as table")
		return
	else:
		temp_matrix=input_array
	number_of_rows,num_of_columns = temp_matrix.shape
	plt.figure()
	tb = plt.table(cellText=np.round(temp_matrix,2), loc=(0,0), cellLoc='center')
	for cell in tb.properties()['child_artists']:
	    cell.set_height(1/number_of_rows)
	    cell.set_width(1/num_of_columns)

	ax = plt.gca()
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()
class Hebbian(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,transfer_function="Hard_limit",seed=None):
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self.transfer_function=transfer_function
        self._initialize_weights()
    def _initialize_weights(self):
        self.weights = []
        self.weights = np.random.randn(self.number_of_classes, self.input_dimensions + 1, )
    def initialize_all_weights_to_zeros(self):
        self.weights = []
        self.weights = np.zeros((self.number_of_classes, self.input_dimensions + 1,))

    # Transfer Functions
    def sigmoid(self, X):
        predicted_y = 1 / (1 + np.exp(-X))
        return predicted_y

    def linear(self, X):
        return X

    def hard_limit(self, X):
        predicted_y = np.where(X <= 0, 0, 1)
        return predicted_y

    def activation_function(self,X):
        if self.transfer_function == "Hard_limit":
            actual_y = self.hard_limit(X)
        elif self.transfer_function == "Sigmoid":
            actual_y = self.sigmoid(X)
        elif self.transfer_function == "Linear":
            actual_y = self.linear(X)
        else:
            print("Undefined Learning Rule")

        return actual_y

    def activation_function_predict(self, X):
        if self.transfer_function == "Hard_limit":
            actual_y = self.hard_limit(X)
            actual_y = np.argmax(actual_y, axis=0)
            actual_y = self.one_hot_encoder(actual_y)
        elif self.transfer_function == "Sigmoid":
            actual_y = self.sigmoid(X)
            actual_y=np.argmax(actual_y, axis=0)
            actual_y=self.one_hot_encoder(actual_y)
        elif self.transfer_function == "Linear":
            actual_y = self.linear(X)
            actual_y = np.argmax(actual_y, axis=0)
            actual_y = self.one_hot_encoder(actual_y)
        else:
            print("Undefined Learning Rule")

        return actual_y

        # Learning Functions



    def unsupervised_hebb(self, x, a, alpha):
        self.weights = self.weights + alpha * np.dot(a, x.T)

    def select_learning_rule(self,learning,X,y,a,alpha,gamma):
        if learning == "Delta":
            self.weights = self.weights + alpha*np.dot((y - a),X.T)
        elif learning == "Filtered":
            self.weights = ((1 - gamma) * self.weights) + (alpha * np.dot(y, X.T))
        elif learning == "Unsupervised_hebb":
            self.weights = self.weights + alpha * np.dot(a, X.T)
        else:
            print("undefined Learning Rule")



    def add_bias_to_x(self,X):
        X = np.insert(X, 0, 1, axis=0)
        return X

    def slice_batches(self,X,start,end):
        X[:, start:end]
        return X

    def one_hot_encoder(self,Y):
        Y = np.squeeze(np.eye(self.number_of_classes)[Y.reshape(-1)])
        Y=Y.transpose()
        return Y

    def predict(self, X):
        X=self.add_bias_to_x(X)
        X=np.dot(self.weights,X)
        predicted_y=self.activation_function(X)

        return predicted_y



    def print_weights(self):
        print(self.weights)

    def train(self, X, y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):
        #Adding bias to X
        X_with_bias=self.add_bias_to_x(X)
        dimension,total_number_of_samples=np.shape(X_with_bias)
        #Calculating batch size
        Flag=False
        num_of_batch_runs=math.ceil(total_number_of_samples/batch_size)
        batch_float=float(total_number_of_samples/batch_size)
        if num_of_batch_runs<batch_float:
            Flag=True
            num_of_batch_runs+=1

        # One-hot-encoding Y
        Y_encoded=self.one_hot_encoder(y)
        for j in range(num_epochs):
            start=batch_size
            end=batch_size
            for i in range(num_of_batch_runs):
                if i==num_of_batch_runs-1 and Flag==True:
                    X_sliced_with_bias=self.slice_batches(X_with_bias,(i*start),total_number_of_samples)
                    Y_sliced_and_encoded=self.slice_batches(Y_encoded,(i*start),total_number_of_samples)
                else:
                    X_sliced_with_bias=self.slice_batches(X_with_bias,(i*start),((i+1)*end))
                    Y_sliced_and_encoded = self.slice_batches(Y_encoded, (i * start),((i+1)*end))

                #Multiplying weights and input and passing through the transfer function
                X=np.dot(self.weights,X_sliced_with_bias)
                predicted_y=self.activation_function(X)
                self.select_learning_rule(learning,X_sliced_with_bias,Y_sliced_and_encoded,predicted_y,alpha,gamma)



    def calculate_percent_error(self,X, y):
        # Adding bias to X
        X_with_bias = self.add_bias_to_x(X)
        # One-hot-encoding Y with transpose
        Y_encoded = self.one_hot_encoder(y)

        #Multiply Weights with bias and pass it through transfer function
        X = np.dot(self.weights, X_with_bias)
        predicted_y=self.activation_function_predict(X)
        
        error=np.sum(np.invert(np.all(Y_encoded == predicted_y, axis=0)))

        #Calculate Percent error
        class_number,total_sample=np.shape(Y_encoded)
        percent_error=(error/total_sample)

        return percent_error


    def calculate_confusion_matrix(self,X,y):
        # Adding bias to X
        X_with_bias = self.add_bias_to_x(X)
        # One-hot-encoding Y with transpose
        Y_encoded = self.one_hot_encoder(y)
        class_number, total_sample = np.shape(Y_encoded)
        # Multiply Weights with bias and pass it through transfer function
        X = np.dot(self.weights, X_with_bias)
        predicted_y = self.activation_function_predict(X)

        confusion = np.zeros((class_number, class_number));
        for i in range(total_sample):
            actual = np.argmax(predicted_y[:, i], axis=0)
            target = np.argmax(Y_encoded[:, i], axis=0)
            confusion[target][actual] += 1
        # for i in range(total_sample):
        #     predicted_sliced = np.expand_dims(predicted_y[:, i], axis=1);
        #     target_sliced = np.expand_dims(Y_encoded[:, i], axis=1);
        #     predict_index = np.argmax(predicted_sliced);
        #     target_index = np.argmax(target_sliced)
        #     confusion[predict_index][target_index] += 1;
        #print(np.sum(confusion))
        return confusion



if __name__ == "__main__":

    # Read mnist data
    number_of_classes = 10
    number_of_training_samples_to_use = 700
    number_of_test_samples_to_use = 100
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_vectorized=((X_train.reshape(X_train.shape[0],-1)).T)[:,0:number_of_training_samples_to_use]
    y_train = y_train[0:number_of_training_samples_to_use]
    X_test_vectorized=((X_test.reshape(X_test.shape[0],-1)).T)[:,0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]
    number_of_images_to_view=16
    test_x=X_train_vectorized[:,0:number_of_images_to_view].T.reshape((number_of_images_to_view,28,28))
    #display_images(test_x)
    input_dimensions=X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit",seed=5)
    # model.initialize_all_weights_to_zeros()


    percent_error=[]
    for k in range (10):
        model.train(X_train_vectorized, y_train,batch_size=300, num_epochs=2, alpha=0.1,gamma=0.1,learning="Delta")
        percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
    print("******  Percent Error ******\n",percent_error)
    confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
    print(np.array2string(confusion_matrix, separator=","))
    model.print_weights()
