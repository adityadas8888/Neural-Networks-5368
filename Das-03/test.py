import numpy as np
import tensorflow as tf
weights = []
activation_function = ['multi_nn.sigmoid', 'multi_nn.sigmoid', 'multi_nn.linear'];
activations=[]
biases=[]
input_dimension=10;

def add_layer(num_nodes, activation_function):
    print('randomint')
    if len( weights)==0:
             weights.append(tf.random.normal( (input_dimension,num_nodes)));
             biases.append(tf.random.normal(1,num_nodes));
    else:
            temp,prev_num_nodes=np.shape( weights[len( weights)-1]);
            weights.append(tf.random.normal(prev_num_nodes,num_nodes));
            biases.append(tf.random.normal(1,prev_num_nodes));
    
    activations.append(activation_function);
    print(np.shape(weights[len(weights)-1]));
    
    # print('zeros')
    # if len( weights)==0:
    #             weights.append(np.zeros(( input_dimension,num_nodes)));
    #             biases.append(np.zeros((1,num_nodes)));
    # else:
    #             temp,prev_num_nodes=np.shape( weights[len( weights)-1]);
    #             weights.append(np.zeros((prev_num_nodes,num_nodes)));
    #             biases.append(np.zeros((1,prev_num_nodes)));
    # activations.append(activation_function);
    # print(np.shape(weights[len(weights)-1]));
    # print('random.rand')
    # if len( weights)==0:
    #             weights.append(np.random.rand( input_dimension,num_nodes));
    #             biases.append(np.random.rand(1,num_nodes));
    # else:
    #             temp,prev_num_nodes=np.shape( weights[len( weights)-1]);
    #             weights.append(np.random.rand(prev_num_nodes,num_nodes));
    #             biases.append(np.random.rand(1,prev_num_nodes));
    # activations.append(activation_function);
    # print(np.shape(weights[len(weights)-1]));
         
add_layer(5,activation_function[0]);
add_layer(6,activation_function[1]);
add_layer(9,activation_function[2]);
add_layer(4,activation_function[0]);

