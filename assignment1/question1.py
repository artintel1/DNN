import numpy as np

inputs = np.array([[1,1],[1,2],[2,-1],[2,0],[-1,2],[-2,1],[-1,-1],[-2,-2]],dtype=int)
labels = np.array([[0,0],[0,0],[0,1],[0,1],[1,0],[1,0],[1,1],[1,1]],dtype=int)
weights = np.zeros((2,2))
biases = np.zeros((2,1))
lr = 1

def activation(z):
    return np.where(z>0,1,0)

def predict(x,w,b):
    return activation(np.dot(w,x.transpose())+b)

while True:
    prediction = predict(inputs,weights,biases)
    error =  labels.transpose()-prediction
    weights += lr*(np.dot(error,inputs))
    # print("product:",np.dot(error,inputs))
    biases = (biases+lr*np.array([error.sum(1)]).transpose())
    if sum(abs(error).sum(0))==0:
        print("weights:\n",weights)
        print("biases:\n",biases)
        print("error:\n",error)
        break


inp = np.array([[2,-1],[-2,-2],[-1,-1]])
print("Checking:",predict(inp,weights,biases).transpose())
