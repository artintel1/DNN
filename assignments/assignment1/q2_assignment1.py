import numpy as np
import matplotlib.pyplot as plt

# Data Reading
attributes = np.array([[2, 2], [1, -2], [-2, -2], [2, -1]])
labels = np.array([0, 1, 0, 1])
no_of_iteration = 5
weights = []
bias = 0
learning_rate = 0.5


def init_perceptron(arg_attribute_array):
    global weights
    global bias
    global learning_rate
    weights = np.zeros(arg_attribute_array.shape[1])
    # weights = np.full(arg_attribute_array.shape[1], 0.3)
    # weights = np.array([0.3, -0.1])
    bias = 0
    learning_rate = 1


def activate_perceptron(arg_inputs, arg_weights, arg_bias, func="step"):
    output = np.dot(arg_inputs, arg_weights) + arg_bias
    if func == "step":
        return np.where(output > 0, 1, 0)
    elif func == "sign":
        return np.where(output > 0, 1, -1)
    elif func == "sigmoid":
        return 1 / (1 + np.exp(-output))
    else:
        return None


def train_perceptron(arg_inputs, arg_weights, arg_bias, arg_learning_rate, arg_error):
    update = arg_learning_rate * arg_error
    arg_weights += update * arg_inputs
    arg_bias += update
    return arg_weights, arg_bias


def plot_decision_boundary(arg_attributes, arg_labels):
    plt.scatter(arg_attributes[:, 0], arg_attributes[:, 1], c=arg_labels, marker='o', edgecolor='k', s=100)
    x_min, x_max = arg_attributes[:, 0].min() - 1, arg_attributes[:, 0].max() + 1
    y_min, y_max = arg_attributes[:, 1].min() - 1, arg_attributes[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = activate_perceptron(np.c_[xx.ravel(), yy.ravel()], weights, bias, "step")
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.grid(color='C5', linestyle=':')
    plt.axis('square')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.title("Vectors and decision boundary")
    plt.show()


init_perceptron(attributes)
for iteration in range(no_of_iteration):
    print("epoch", iteration, "\n==================================")
    for index, inputs in enumerate(attributes):
        output = activate_perceptron(inputs, weights, bias, "step")
        error = labels[index] - output
        weights, bias = train_perceptron(inputs, weights, bias, learning_rate, error)
        print(inputs, labels[index], weights, bias, output, error, sep='\t\t')

output_predicted = activate_perceptron(attributes, weights, bias, "step")
accuracy = np.mean(output_predicted == labels)
print("Accuracy", round(accuracy * 100, 2), "%")
plot_decision_boundary(attributes, labels)
