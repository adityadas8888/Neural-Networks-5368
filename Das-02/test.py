import numpy as np
batch_size = 30;
start = 0;
dataset=130
no_runs=int(dataset/batch_size);
X= np.ones(1300);
end = batch_size;
for j in range(no_runs+1):
    print("before, start",start);
    print("\nbefore, end",end);
    # print(X[start:end]);
    print("\n",len(X[start:end]));
    start = end;
    if(j==no_runs-1):
        end =  dataset;
        print("epoch",end);
    elif(j<no_runs):
        end = ((j+2)*batch_size);
        print("\nafter, start",start);
        print("\nafter, end",end);