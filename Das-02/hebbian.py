# Das, Aditya
# 1001-675-762
# 2019-10-07
# Assignment-02-01

import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

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
        self.weights = np.random.randn(self.number_of_classes, self.input_dimensions + 1, );
    
    def initialize_all_weights_to_zeros(self):
        self.weights = []
        self.weights = np.zeros((self.number_of_classes, self.input_dimensions + 1,));

    def predict(self, X):

        X = np.dot(self.weights,np.insert(X, 0, 1, axis=0)); # added bias and multiplied weights
        predicted=self.activation_function(X);
        return predicted

    def activation_function(self,X):
        if self.transfer_function == "Hard_limit":
            X = np.where(X <= 0, 0, 1);
        elif self.transfer_function == "Sigmoid":
            X = 1 / (1 + np.exp(-X));
        elif self.transfer_function == "Linear":
            return X
        else:
            print("Invalid Learning Rule, Exiting!!!");
            exit();

        return X

    def predict_noduplicate(self, X):

        X=np.dot(self.weights,np.insert(X, 0, 1, axis=0));  # added bias and multiplied weights
        if self.transfer_function == "Hard_limit":
            predicted = self.one_hot(np.argmax(np.where(X <= 0, 0, 1), axis=0));
        elif self.transfer_function == "Sigmoid":
            predicted=self.one_hot(np.argmax(1 / (1 + np.exp(-X)), axis=0));
        elif self.transfer_function == "Linear":
            predicted = self.one_hot(np.argmax(X, axis=0));
        else:
            print("Invalid Learning Rule, Exiting!!!");
            exit();
        return predicted

    def one_hot(self,Y):
        Y = np.squeeze(np.eye(self.number_of_classes)[Y.reshape(-1)]);
        return Y.T

    def print_weights(self):
        print(self.weights);

    def chunker(self,X,batch_size):
        no_runs = int(X.shape[1]/batch_size);
        remaining = X.shape[1]%batch_size;
        if remaining!=0:
            no_runs+=1
        return no_runs,remaining;

    def train(self, X, Y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):           # definitely change this
        
        X_with_bias=np.insert(X,0,1,axis=0);
        no_runs,remaining=self.chunker(X,batch_size);

        Y_encoded=self.one_hot(Y)
        for i in range(num_epochs):
            start=batch_size
            end=batch_size
            for j in range(no_runs):
                if i==no_runs-1 and remaining!=0:
                    input_sliced=X_with_bias[:, (j*start):X_with_bias.shape[1]]
                    target_sliced=Y_encoded[:,(j*start):X_with_bias.shape[1]]
                else:
                    input_sliced = X_with_bias[:,(j*start):((j+1)*end)]
                    target_sliced = Y_encoded[:, (j * start):((j+1)*end)]

                output=np.dot(self.weights,input_sliced);
                predicted=self.activation_function(output);

                if learning == "Delta":
                    self.weights = self.weights + alpha*np.dot((target_sliced- predicted),input_sliced.T);
                elif learning == "Filtered":
                    self.weights = ((1 - gamma) * self.weights) + (alpha * np.dot(target_sliced, input_sliced.T));
                elif learning == "Unsupervised_hebb":
                    self.weights = self.weights + alpha * np.dot(predicted, input_sliced.T);
                else:
                    print("Invalid Learning Rule, Exiting!!!");
                    exit();

    def calculate_percent_error(self,X, Y):       # try to change this

        X = self.predict_noduplicate(X);
        Y = self.one_hot(Y);
        flag = 0;
        for i in range(X.shape[1]):
           predicted_sliced = np.expand_dims(X[:,i],axis=1);
           target_sliced = np.expand_dims(Y[:,i],axis=1);
           
           if(np.array_equal(predicted_sliced,target_sliced)):
               flag +=0
           else:
               flag+=1;
        return flag/Y.shape[1];

    def calculate_confusion_matrix(self,X,Y):

        X = self.predict_noduplicate(X);
        Y = self.one_hot(Y);
        confusion_matrix = np.zeros((Y.shape[0], Y.shape[0]));
        for i in range(Y.shape[1]):
            actual_index = np.argmax(X[:, i], axis=0);
            target_index = np.argmax(Y[:, i], axis=0);
            confusion_matrix[target_index][actual_index] += 1
        return confusion_matrix



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
    model.initialize_all_weights_to_zeros()
    percent_error=[]
    for k in range (10):
        model.train(X_train_vectorized, y_train,batch_size=300, num_epochs=2, alpha=0.1,gamma=0.1,learning="Delta")
        percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
    print("******  Percent Error ******\n",percent_error)
    confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
    print(np.array2string(confusion_matrix, separator=","))
    model.print_weights()
