import numpy as np
import time

#input variables
inputx = np.array([[1],[2],[0.5],[2]]).T
#target
targety = np.array([[0],[1],[0]]).T

def sigmoid(x):
    return  1 / (1 + np.exp(-x))

#list for saving the result
weight =[]
outputy=[]
mse=[]

start_time = time.time()

#Running for search random weight by num of iteration
for iteration in range(0,10000):
    w = np.random.random((4,3))
    weight.append(w)
    bias=np.array([[0],[1],[0]]).T
    y = sigmoid(np.dot(inputx,w)+bias)
    outputy.append(y)
    minus = np.subtract(targety, outputy[iteration])
    square = np.square(minus)
    error = square.mean()  
    mse.append(error)
    
print('The Smallest MSE For',iteration+1,'=', min(mse))
print("--- %s seconds ---" % (time.time() - start_time))

for i in range(0, iteration+1):
    if mse[i] == min(mse):
        index = i
print(weight[index])