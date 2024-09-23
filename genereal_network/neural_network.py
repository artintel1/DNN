import numpy as np
from typing import List

class NeuralNetwork:
    def __init__(self, layers: List,input_size:int,lr=0.01) -> None:
        self.lr = lr
        self.layers = layers
        # self.weights = [np.ones((layers[i], layers[i + 1])).transpose() for i in range(len(layers) - 1)]
        self.weights = [(np.random.randint(-200,200,(layers[i], layers[i + 1]))/100).transpose() for i in range(len(layers) - 1)]
        self.biases = [np.zeros((layer, 1)) for layer in layers]
        self.states = [np.zeros((layers[i],1)) for i in range(len(layers)-1)]
        self.weights.insert(0,np.ones((input_size,self.layers[0])).transpose())
        self.states.insert(0,np.zeros((input_size,1)))
    def prepare_data(self, data):
        data = np.asarray(data).reshape(-1, 1)
        if data.ndim != 2 or data.shape[1] != 1:
            raise ValueError("Input must have shape (number_of_inputs, 1).")
        return data
    
    def activate(self, z):
        return np.where(z>0,z,0.01*z)
    def predict(self,x):
        a = self.prepare_data(x)
        for (w,b) in zip(self.weights,self.biases):
            z = np.dot(w,a)+b
            a = self.activate(z)
        return a
    
    def forward(self, input_data):
        input_data = self.prepare_data(input_data)
        a = input_data
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            # print("HI")
            self.states[i] = a
            z = np.dot(w,a)+b
            # print("Hello")
            a = self.activate(z)
        self.out = a
        return a
        
    def backward(self,labels):
        self.prepare_data(labels)
        print("labels:\n",labels)
        print("OUT:\n",self.out)
        o = self.out
        dz = (o-labels)*np.where(o>0,1,0.01)
        for W,b,a in zip(reversed(self.weights),reversed(self.biases),reversed(self.states)):
            dw = np.dot(dz,(a.transpose()))
            # print("Wth")
            dw_prev = np.dot(W.transpose(),dz)
            # next_gradient = np.dot(W.transpose(),gradient)
            W -= self.lr*dw
            b -= self.lr*dz
            dz = dw_prev*np.where(a>0,1,0.01)
            
