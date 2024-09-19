import numpy as np

inputs = np.array([[1,1],[1,2],[2,-1],[2,0],[-1,2],[-2,1],[-1,-1],[-2,-2]],dtype=int)
labels = np.array([[0,0],[0,0],[0,1],[0,1],[1,0],[1,0],[1,1],[1,1]],dtype=int)
weights = np.array([[1,0],[0,1]])
biases = np.array([[1],[1]])

lr = 1
def activation(z):
    return np.where(z>0,1,0)

print("HHERE:",activation(np.array([0.5,0.6,-0.2])))
def predict(x,w,b):
    return activation(np.dot(w,x.transpose())+b)

print(predict(inputs,weights,biases))
error = predict(inputs,weights,biases)-labels.transpose()
print(predict(inputs,weights,biases)-labels.transpose())
print(lr*np.dot(error,inputs))

while True:
    prediction = predict(inputs,weights,biases)
    error =  -prediction+labels.transpose()
    weights += lr*(np.dot(error,inputs))
    # print("product:",np.dot(error,inputs))
    biases = (biases+lr*np.array([error.sum(1)]).transpose())
    print("prediction:",prediction)
    print("weights",weights)
    print("error",error)
    if sum(abs(error).sum(0))==0:
        print("weights:",weights)
        print("biases:",biases)
        print("error:",error)
        break


inp = np.array([[2,-1],[-2,-2],[-1,-1]])
print("Checking:",predict(inp,weights,biases).transpose())


