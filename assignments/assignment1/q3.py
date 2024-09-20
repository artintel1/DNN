import numpy as np
import matplotlib.pyplot as plt

# Data Reading
attributes = np.array([[1, 1], [1, 2], [2, -1], [2, 0], [-1, 2], [-2, 1], [-1, -1], [-2, -2]])
labels = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 1]])
colors = ['C1', 'C1', 'C2', 'C2', 'C3', 'C3', 'C4', 'C4']
no_of_iteration = 4
weights = np.array([[0, 0], [0, 0]])
bias = np.array([[1], [1]])


def activate_perceptron(arg_inputs, arg_weights, arg_bias):
    output = np.dot(arg_weights, arg_inputs[np.newaxis].transpose()) + arg_bias
    return np.where(output > 0, 1, 0)


def train_perceptron(arg_inputs, arg_weights, arg_bias, arg_error):
    arg_weights += np.dot(arg_error, arg_inputs[np.newaxis])
    arg_bias += arg_error
    return arg_weights, arg_bias


def plot_decision_boundary(plt, arg_input, arg_weight, arg_bias):
    x_min, x_max = arg_input[:, 0].min() - 1, arg_input[:, 0].max() + 1
    y_min, y_max = arg_input[:, 1].min() - 1, arg_input[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    for row in range(arg_weight.shape[0]):
        output = np.dot(np.c_[xx.ravel(), yy.ravel()], arg_weight[row]) + arg_bias[row]
        Z = np.where(output > 0, 1, 0)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
    plt.grid(color='C5', linestyle=':')
    plt.axis('square')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.scatter(arg_input[:, 0], arg_input[:, 1], marker='o', color=colors, edgecolor='k', s=100)
    plt.title("Vectors and decision boundary")


for iteration in range(no_of_iteration):
    print("epoch", iteration, "\n==================================")
    for row in range(attributes.shape[0]):
        output = activate_perceptron(attributes[row], weights, bias)
        error = labels[row][np.newaxis].T - output
        weights, bias = train_perceptron(attributes[row], weights, bias, error)
        print("Vector", row + 1, "-----------------------------")
        print("input:", attributes[row], "target:", labels[row])
        # print("target:", labels[row])
        print("weights:\n", weights)
        print("bias:\n", bias)
        print("output:\n", output)
        print("error:\n", error)

output_predicted = np.dot(attributes, weights.T) + bias.T
output_predicted = np.where(output_predicted > 0, 1, 0)
accuracy = np.mean(output_predicted == labels)
print("Accuracy", round(accuracy * 100, 2), "%")
plot_decision_boundary(plt, attributes, weights, bias)
plt.show()
