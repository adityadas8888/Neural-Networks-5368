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
        self.weights = [];
        self.weights = np.array(self.weights, dtype=np.float);
        self.weights = np.random.randn(self.number_of_classes,self.input_dimensions+1);

    def initialize_all_weights_to_zeros(self):
        self.weights = np.zeros([self.number_of_classes,self.input_dimensions+1], dtype = int);

    def predict(self, X):

        X = np.insert(X,0,1,axis=0);                            # adds the bias to the matrix
        output = np.dot(self.weights,X);         # predict the values by multiplying the trained weights
        predicted = self.activation_function(output);  #call to activation_function
        
        return predicted

    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights);

    def train(self, X, Y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):

        no_runs,remaining = self.chunkify(X.shape[1],batch_size);                   # get the number of batches and the batch size remaining.
        X = np.insert(X,0,1,axis=0);                            # adds the bias to the matrix
        Y = self.one_hot(Y);
        Y=Y.T;
        if remaining!=0:
            no_runs+=1

        for i in range(num_epochs):  
           start = 0;
           end = batch_size;
           for j in range(no_runs):
               
               input_sliced = X[:,start:end];
               output = np.dot(self.weights,input_sliced);  # this multiplies the weights with the sliced input. Essentially giving the output.
               predicted = self.activation_function(output);  #call to activation_function   
               target_sliced = Y [:,start:end];
               error = target_sliced-predicted;
               ep = self.calculate_error(error,input_sliced);

               if(learning=="Filtered"):
                   self.weights = (1-gamma)*self.weights+(alpha*target_sliced*input_sliced.T);
               elif(learning=="Delta"):
                   self.weights = self.weights+(alpha*ep);
               elif(learning=="Unsupervised_hebb"):
                   self.weights = self.weights+(alpha*predicted*input_sliced.T);      
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

    def activation_function_predict(self,X):

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


    def one_hot(self,X):
        return(np.squeeze(np.eye(10)[X.reshape(-1)]))

    def sigmoid(self,X):
                return 1/(1+np.exp(-X))
    
    def hard_limit(self,X):
                return np.where(X[:] <=0, 0,1);

    def chunkify(self,dataset_size,batch_size):
        no_runs = int(dataset_size/batch_size);
        remaining = dataset_size%batch_size;
        return no_runs,remaining

    def calculate_error(self,X,Y):
        return X.dot(np.transpose(Y));

    def calculate_percent_error(self,X, Y):

        Y = self.one_hot(Y);
        Y=Y.T;
        predicted= self.predict(X);
        flag=0
        if(self.transfer_function=='Hard_limit'):
            for i in range(len(X[0])):
                predicted_sliced = np.expand_dims(predicted[:,i],axis=1);
                target_sliced = np.expand_dims(Y[:,i],axis=1);
        #    print("predicted\n",predicted_sliced);
        #    print("target\n",target_sliced);
                if(np.array_equal(predicted_sliced,target_sliced)):
                    flag +=0
                else:
                    flag+=1;
            return (flag/len(X[0]));

        elif(self.transfer_function=='Linear' or self.transfer_function=='Sigmoid'):
            max_col = np.amax(predicted, axis=0)
            predicted=np.where(max_col==predicted,1,0)
            for i in range(len(X[0])):
                predicted_sliced = np.expand_dims(predicted[:,i],axis=1);
                target_sliced = np.expand_dims(Y[:,i],axis=1);
                if(np.array_equal(predicted_sliced,target_sliced)):
                    flag +=0
                else:
                    flag+=1;

            return (flag/len(X[0]));

    def calculate_confusion_matrix(self,X,Y):

        Y = self.one_hot(Y);                    ##did one hot encoding
        Y=Y.T;
        a,b = np.shape(Y);
        predicted = self.predict(X);    

        flag=0
        confusion = np.zeros((self.number_of_classes,self.number_of_classes));
        for i in range(b):

            predicted_sliced = np.expand_dims(predicted[:,i],axis=1);
            target_sliced = np.expand_dims(Y[:,i],axis=1);
            predict_index=np.argmax(predicted_sliced);
            target_index=np.argmax(target_sliced)
            confusion[target_index][predict_index]+=1;
        return confusion

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

    number_of_images_to_view=16
    test_x=X_train_vectorized[:,0:number_of_images_to_view].T.reshape((number_of_images_to_view,28,28))
    # display_images(test_x)
    input_dimensions=X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                     transfer_function="Hard_limit",seed=5)
    model.initialize_all_weights_to_zeros()
    # print("input dimensions",input_dimensions);
    percent_error=[]
    for k in range (10):
        model.train(X_train_vectorized, y_train,batch_size=300, num_epochs=2, alpha=0.1,gamma=0.1,learning="Delta")
        # print('x_test_vectorised',X_test_vectorized.shape);
        # print('y_test',y_test.shape);
        percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
    print("******  Percent Error ******\n",percent_error);
    confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
    print(np.array2string(confusion_matrix, separator=","))
