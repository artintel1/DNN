import numpy as np
import matplotlib.pyplot as plt
inputs = np.array([[1,1],[1,2],[2,-1],[2,0],[-1,2],[-2,1],[-1,-1],[-2,-2]],dtype=int)
labels = np.array([[0,0],[0,0],[0,1],[0,1],[1,0],[1,0],[1,1],[1,1]],dtype=int)
weights = np.zeros((2,2))
biases = np.zeros((2,1))
lr = 1

def activation(z):
    return np.where(z>0,1,0)

def predict(x,w,b):
    return activation(np.dot(w,x.transpose())+b)

note = """The number of epochs can be adjusted based on the model's error.
          However, batch size needs to be selected carefully. It should be
        chosen such that the number of input samples is divisible by the
          batch size, i.e., number_of_input_samples % batch_size = 0."""

print(note,"\n")
epochs = int(input("Enter number of epochs:"))
batch_size = int(input("Enter batch size:"))
if inputs.shape[0] % batch_size != 0:
    raise ValueError("The number of rows must be divisible by the batch size")

batched_inputs = inputs.reshape(-1, batch_size, inputs.shape[1])
batched_labels = labels.reshape(-1,batch_size,labels.shape[1])

for epoch in range(epochs):
    total_error = 0
    for b_inputs,b_labels in zip(batched_inputs,batched_labels):
        prediction = predict(b_inputs,weights,biases)
        error =  b_labels.transpose()-prediction
        print("b_labels:\n",b_labels)
        print("prediction:\n",prediction)
        print("error:\n",error)
        weights += lr*(np.dot(error,b_inputs))
        # print("product:",np.dot(error,inputs))
        biases = (biases+lr*np.array([error.sum(1)]).transpose())
        # biases = np.array([[0],[0]])
        total_error+=sum(abs(error).sum(0))
    if total_error==0:
        print("weights:\n",weights)
        print("biases:\n",biases)
        print("error:\n",error)
        break

inp = np.array([[2,-1],[-2,-2],[-1,-1]])
print("Checking:",predict(inp,weights,biases).transpose())

def plot_decision_boundary(inputs, labels, weights, biases):
    plt.figure(figsize=(10, 6))
    class_mapping = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    colors = ['red', 'green', 'blue', 'purple']
    for point, label in zip(inputs, labels):
        plt.scatter(point[0], point[1], c=colors[class_mapping[tuple(label)]], s=100)

    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = predict(grid_points, weights, biases).transpose()
    predicted_classes = np.array([class_mapping[tuple(p)] for p in predictions]).reshape(xx.shape)
    plt.contourf(xx, yy, predicted_classes, alpha=0.3, levels=4, cmap='RdYlBu')
    plt.colorbar(ticks=[0, 1, 2, 3], label='Class')
    plt.title('Perceptron Decision Boundary with Four Classes')
    plt.xlabel('X1')
    plt.ylabel('X2')
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {label}', 
                          markerfacecolor=colors[class_mapping[label]], markersize=10) for label in class_mapping]
    plt.legend(handles=handles)
    plt.grid(True)
    plt.show()

plot_decision_boundary(inputs=inputs,labels=labels,weights=weights,biases=biases)