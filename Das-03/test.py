# import numpy as np

# weights = []
# activation_function = ['multi_nn.sigmoid', 'multi_nn.sigmoid', 'multi_nn.linear'];
# activations=[]
# biases=[]
# input_dimension=10;
# def add_layer(num_nodes,activation_function):
#     if len(weights)==0:
#                 weights.append(np.zeros((input_dimension,num_nodes)));
#                 biases.append(np.zeros((num_nodes,1)));
#     else:
#                 temp,prev_num_nodes=np.shape(weights[len(weights)-1]);
#                 # print(temp);
#                 # print(prev_num_nodes);
#                 weights.append(np.zeros((prev_num_nodes,num_nodes)));
#                 biases.append(np.zeros((prev_num_nodes,1)));
#     activations.append(activation_function);
         
# add_layer(5,activation_function[0]);
# add_layer(6,activation_function[1]);
# add_layer(9,activation_function[2]);
# add_layer(4,activation_function[0]);

# print(weights);