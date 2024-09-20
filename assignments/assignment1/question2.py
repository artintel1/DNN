import numpy as np

inputs = np.array([[2,2], [1,-2], [-2,-2], [2,-1]],dtype=int)
labels = np.array([[0],[1],[0],[1]],dtype=int)
weights = np.zeros((1,2))
biases = np.zeros((1,1))
lr = 0.1

def activation(z):
    return np.where(z>0,1,0)

def predict(x,w,b):
    return activation(np.dot(w,x.transpose())+b)

prediction = predict(inputs,weights,biases)
print(predict(inputs,weights,biases))
print(labels.transpose()-prediction)
print(biases)
counter=0
while True:
    prediction = predict(inputs,weights,biases)
    error =  labels.transpose()-prediction
    print("error:\n",error)
    weights += lr*(np.dot(error,inputs))
    print("weights:\n",weights)
    # print("product:",np.dot(error,inputs))
    biases = (biases+lr*np.array([error.sum(1)]).transpose())
    counter+=1
    if sum(abs(error).sum(0))==0 or counter==5:
        print("weights:\n",weights)
        print("biases:\n",biases)
        print("error:\n",error)
        break


inp = np.array([[2,-1],[-2,-2],[-1,-1],[5,5],[10,-8]])
print("Checking:",predict(inp,weights,biases).transpose())


