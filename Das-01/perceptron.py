# Das, Aditya
# 1001-675-672
# 2019-09-17
# Assignment-01-01
import numpy as np
import itertools
class Perceptron(object):
   def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
       if seed != None:
           np.random.seed(seed)
       self.input_dimensions = input_dimensions
       self.number_of_classes=number_of_classes
       self.weights = []
       self._initialize_weights()
   def _initialize_weights(self):
       self.weights = [];
       self.weights = np.array(self.weights, dtype=np.float);
       self.weights = np.random.randn(self.number_of_classes,self.input_dimensions+1);
   def initialize_all_weights_to_zeros(self):
       # self.weights = []
       self.weights = np.zeros([self.number_of_classes,self.input_dimensions+1], dtype = int);
       return self.weights;
   def predict(self, X):
       """ this function returns the predicted output after multiplying the weights with the input and then running it through an activation function"""
       X = np.insert(X,0,1,axis=0);
    #    print(self.weights,X.shape)
       # print("weights\n",self.weights.shape);
       output = np.dot(self.weights,X);
       # print(output.shape);                        """ multiplied the input with the weights"""
       predicted = np.where(output[:] <0, 0,1);              """ activation function"""
       return predicted;
   def print_weights(self):
       print(self.weights);
   def train(self, X, Y, num_epochs=10, alpha=0.001):
       X = np.insert(X,0,1,axis=0);
       for i in range(num_epochs):
           for j in range(len(X[0])):
            #    print(len(X[0]))
               input_sliced = np.expand_dims(X[:,j],axis =1);
               output = np.dot(self.weights,input_sliced);
               predicted_sliced = np.where(output[:] <0, 0,1);   
               target_sliced = np.expand_dims(Y[:,j],axis=1);

               print(target_sliced.shape,predicted_sliced.shape);
               error = target_sliced - predicted_sliced;
               ep = self.calculate_error(error,input_sliced);
               self.weights = self.weights + alpha*(ep);
   def calculate_error(self,X,Y):
       return X.dot(np.transpose(Y));
   def calculate_percent_error(self,X, Y):
       predicted= self.predict(X);
       flag = 0;
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
if __name__ == "__main__":
   input_dimensions = 2
   number_of_classes = 2
   model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
   X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                       [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
   # X_train = np.append(X_train, [np.ones(len(X_train[0]))], axis=0);  """ appended bias to the input"""
   print(X_train);                                                         #can be deleted.
   print(model.predict(X_train));                                       """ prints the predicted labels """ #can be deleted
   Y_train = np.array([[1, 0, 0, 1],
                       [0, 1, 1, 0]]);
   model.initialize_all_weights_to_zeros();
   print("****** Model weights ******\n",model.weights);
   print("****** Input samples ******\n",X_train);
   print("****** Desired Output ******\n",Y_train);
   percent_error=[]
   for k in range (20):
       model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
       percent_error.append(model.calculate_percent_error(X_train,Y_train))
   print("******  Percent Error ******\n",percent_error)
   print("****** Model weights ******\n",model.weights)