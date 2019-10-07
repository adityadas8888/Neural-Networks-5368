import numpy as np
a=np.array([[0.5,0.5,0.3,0.4,0],
            [0.5,0.3,0.5,0.2,1],
            [0.2,0.4,0.2,0.3,0]]);
some = np.amax(a, axis=0);
print(some);
print(len(a[0]));
a=np.where(some==a,1,0);
print("a matrix\n",a);
b=np.array([[1,0,0,0,1],
            [0,0,1,0,0],
            [0,1,0,1,0]])
print("b matrix\n",b);


# confusion = np.zeros((3,3));
# for i in range(len(a[0])):

#     predicted_sliced = np.expand_dims(a[:,i],axis=1);
#     target_sliced = np.expand_dims(b[:,i],axis=1);
#     predict_index=np.argmax(predicted_sliced);
#     target_index=np.argmax(target_sliced)
#     confusion[predict_index][target_index]+=1;

# print("confusion\n",confusion);
