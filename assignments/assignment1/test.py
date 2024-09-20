# import numpy as np
# import time

# # Generate two 200x200 matrices with random values
# matrix_a = np.random.rand(300, 300)
# matrix_b = np.random.rand(300, 300)

# # Function to multiply matrices using loops
# def loop_matrix_multiplication(a, b):
#     result = np.zeros((300, 300))
#     for i in range(300):
#         for j in range(300):
#             for k in range(300):
#                 result[i, j] += a[i, k] * b[k, j]
#     return result

# # Measure time for loop-based matrix multiplication
# start_loop = time.time()
# result_loop = loop_matrix_multiplication(matrix_a, matrix_b)
# end_loop = time.time()

# # Measure time for vectorized matrix multiplication
# start_vec = time.time()
# result_vec = np.dot(matrix_a, matrix_b)
# end_vec = time.time()

# # Convert time to milliseconds
# time_loop = (end_loop - start_loop) * 1000
# time_vec = (end_vec - start_vec) * 1000

# # Display results
# print(f"Time taken for loop-based multiplication: {time_loop:.6f} milliseconds")
# print(f"Time taken for vectorized multiplication: {time_vec:.6f} milliseconds")
# print(f"Time difference (loop - vectorized): {time_loop - time_vec:.6f} milliseconds")
# print(result_vec[0][0:4])
# print(result_loop[0][0:4])

# import numpy as np

# # Original input matrix
# inputs = np.array([[1, 1], [1, 2], [2, -1], [2, 0],
#                    [-1, 2], [-2, 1], [-1, -1], [-2, -2]], dtype=int)

# # Desired batch size
# batch_size = 2  # Change this to any batch size

# # Total number of rows in the input
# num_rows = inputs.shape[0]

# # Check if batch size is compatible
# if num_rows % batch_size != 0:
#     raise ValueError("The number of rows must be divisible by the batch size")

# # Reshape the input matrix
# batched_matrix = inputs.reshape(-1, batch_size, inputs.shape[1])

# print(type(batched_matrix[0]))

import numpy as np
import matplotlib.pyplot as plt

# Define inputs and labels
inputs = np.array([[1,1],[1,2],[2,-1],[2,0],[-1,2],[-2,1],[-1,-1],[-2,-2]],dtype=int)
labels = np.array([[0,0],[0,0],[0,1],[0,1],[1,0],[1,0],[1,1],[1,1]],dtype=int)

# Initialize weights and biases
weights = np.ones((2,2))
biases = np.ones((2,1))
lr = 1

# Activation function
def activation(z):
    return np.where(z>0,1,0)

# Prediction function
def predict(x, w, b):
    return activation(np.dot(w, x.transpose()) + b)

note = """The number of epochs can be adjusted based on the model's error.
          However, batch size needs to be selected carefully. It should be
          chosen such that the number of input samples is divisible by the
          batch size, i.e., number_of_input_samples % batch_size = 0."""

print(note, "\n")
epochs = int(input("Enter number of epochs:"))
batch_size = int(input("Enter batch size:"))

if inputs.shape[0] % batch_size != 0:
    raise ValueError("The number of rows must be divisible by the batch size")

# Reshape inputs and labels into batches
batched_inputs = inputs.reshape(-1, batch_size, inputs.shape[1])
batched_labels = labels.reshape(-1, batch_size, labels.shape[1])

# Training the perceptron
for epoch in range(epochs):
    total_error = 0
    for b_inputs, b_labels in zip(batched_inputs, batched_labels):
        prediction = predict(b_inputs, weights, biases)
        error = b_labels.transpose() - prediction
        weights += lr * (np.dot(error, b_inputs))
        biases += lr * np.array([error.sum(1)]).transpose()
        total_error += sum(abs(error).sum(0))
    
    if total_error == 0:
        print("weights:\n", weights)
        print("biases:\n", biases)
        print("error:\n", error)
        break

# Test predictions
inp = np.array([[2,-1],[-2,-2],[-1,-1]])
print("Checking:", predict(inp, weights, biases).transpose())

def plot_decision_boundary(inputs, labels, weights, biases):
    plt.figure(figsize=(10, 6))
    
    # Define a unique integer for each class tuple
    class_mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }
    colors = ['red', 'green', 'blue', 'purple']
    
    # Plot input points with their corresponding labels
    for point, label in zip(inputs, labels):
        label_tuple = tuple(label)
        plt.scatter(point[0], point[1], c=colors[class_mapping[label_tuple]], s=100, label=f'Class {label_tuple}' if label_tuple not in [tuple(lbl) for lbl in inputs[:np.where(inputs == point)[0][0]]] else "")
    
    # Generate a grid to plot the decision boundary
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # Compute predictions for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = predict(grid_points, weights, biases).transpose()
    
    # Convert predictions into a single integer label for color mapping
    predicted_classes = np.array([class_mapping[tuple(p)] for p in predictions]).reshape(xx.shape)
    
    # Draw decision boundary using integer class labels
    plt.contourf(xx, yy, predicted_classes, alpha=0.3, levels=4, cmap='RdYlBu')
    plt.colorbar(ticks=[0, 1, 2, 3], label='Class')
    
    # Labels and title
    plt.title('Perceptron Decision Boundary with Four Classes')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper right', markerscale=0.7)
    plt.grid(True)
    plt.show()

# Call the function to plot the decision boundary
plot_decision_boundary(inputs, labels, weights, biases)