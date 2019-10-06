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
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit" ,  "Sigmoid", "Linear".
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self.transfer_function=transfer_function
        self._initialize_weights()
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = [];
        self.weights = np.array(self.weights, dtype=np.float);
        self.weights = np.random.randn(self.number_of_classes,self.input_dimensions+1);

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = np.zeros([self.number_of_classes,self.input_dimensions+1], dtype = int);
    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]. This array is a numerical array.
        """
        X = np.insert(X,0,1,axis=0);                            # adds the bias to the matrix
        predicted = self.transfer_function(X);  #call to activation_function   
        return predicted

    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights);

    def train(self, X, Y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """

        no_runs,remaining = self.chunkify(X.shape[1],batch_size);                   # get the number of batches and the batch size remaining.

        X = np.insert(X,0,1,axis=0);                            # adds the bias to the matrix

        for i in range(num_epochs):  
           start = 0;
           end = batch_size-1;
           for j in range(no_runs+1):
               
               input_sliced = X[:,start:end];
               output = np.dot(self.weights,input_sliced);  # this multiplies the weights with the sliced input. Essentially giving the output.
               predicted = self.transfer_function(output);  #call to activation_function   
            #    calculate the error matrix Y- predicted by expanding Y
               ep = self.calculate_error(error,input_sliced);
               if(self.transfer_function=="Filtered"):
                   self.weights = (1-gamma)*self.weights+alpha*Y#*T;
               elif(self.transfer_function=="Delta"):
                   self.weights = self.weights+alpha*Y#ep;
               elif(self.transfer_function=="Unsupervised_hebb"):
                   self.weights = self.weights+alpha*Y#*A;      
               else:
                   print("invalid learning rule,exiting!!!");
                   exit();
       
               start = end;
               if(j==no_runs-1):
                   end =  X.shape[1]
               elif(j<no_runs):
                   end = ((j+2)*batch_size);
                






    def activation_function(self,X):

        if(self.transfer_function=="Hard_limit"):
            predicted = self.hard_limit(X);
            return X;
        elif(self.transfer_function=="Sigmoid"):
            predicted = self.sigmoid(X);
            return X;
        elif(self.transfer_function=="Linear"):
            return X;

        else:
            print("Invalid transfer function, exiting!!!!!");
            exit();

    def sigmoid(self,X):
                return 1/(1+np.exp(-X))
    
    def hard_limit(self,X):
                return np.where(X[:] <0, 0,1);

    def chunkify(self,dataset_size,batch_size):
        no_runs = int(dataset_size/batch_size);
        remaining = dataset_size%batch_size;
        return no_runs,remaining

    def calculate_error(self,X,Y):
        return X.dot(np.transpose(Y));

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        predicted= self.predict(X);
        flag = 0;
        for i in range(len(X[0])):                                                  ## this has to be changed.
           predicted_sliced = np.expand_dims(predicted[:,i],axis=1);
           target_sliced = np.expand_dims(Y[:,i],axis=1);
        #    print("predicted\n",predicted_sliced);
        #    print("target\n",target_sliced);
           if(np.array_equal(predicted_sliced,target_sliced)):
               flag +=0
           else:
               flag+=1;
        return (flag/len(X[0]));


    def calculate_confusion_matrix(self,X,y):
        """
        Given a desired (true) output as one hot and the predicted output as one-hot,
        this method calculates the confusion matrix.
        If the predicted class output is not the same as the desired output,
        then it is considered one error.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """
        



if __name__ == "__main__":

    # Read mnist data
    number_of_classes = 10
    number_of_training_samples_to_use = 700
    number_of_test_samples_to_use = 100

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(); ## splitting into traning and testing dataset

    X_train_vectorized=((X_train.reshape(X_train.shape[0],-1)).T)[:,0:number_of_training_samples_to_use]

    y_train = y_train[0:number_of_training_samples_to_use]                                                              # this is the target sample to be tested on.
    X_test_vectorized=((X_test.reshape(X_test.shape[0],-1)).T)[:,0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]


    print('xtest',X_test.shape);
    print('\nytest',y_train.shape);
    print('\nxtrain',X_train.shape);
    print('\nytrain',y_train.shape);
    print('\nxtest_vectorised',X_test_vectorized.shape);
    print('\nxtrain_vectorised',X_train_vectorized.shape);
    number_of_images_to_view=16
    test_x=X_train_vectorized[:,0:number_of_images_to_view].T.reshape((number_of_images_to_view,28,28))
    display_images(test_x)
    input_dimensions=X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                     transfer_function="Hard_limit",seed=5)
    model.initialize_all_weights_to_zeros()
    print("input dimensions",input_dimensions);
    percent_error=[]
    # for k in range (10):
    #     model.train(X_train_vectorized, y_train,batch_size=300, num_epochs=2, alpha=0.1,gamma=0.1,learning="Delta")
    #     percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
    # print("******  Percent Error ******\n",percent_error);
    # confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
    # print(np.array2string(confusion_matrix, separator=","))
