import random as random
import numpy as np
import matplotlib.pyplot as plt

weights = np.array([[-4.0, -1.0], [1.0, -1.5]])
bias = np.array([[1.5], [0.0]])
points = np.array([[0, 1.5], [0, 0]])


def plot_otho_line(plt, arg_weight, arg_points):
    for row in range(arg_weight.shape[0]):
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        slope = weights[row][1] / weights[row][0]
        slope1 = -(1 / slope)
        plt.axline(arg_points[row], slope=slope1, color=color, linestyle="--", linewidth=2)


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


class1 = np.array([[1, 1], [1, 2]])
class2 = np.array([[2, -1], [2, 0]])
class3 = np.array([[-1, 2], [-2, 1]])
class4 = np.array([[-1, -1], [-2, -2]])
input = np.concatenate((class1, class2, class3, class4), axis=0)
plot_decision_boundary(plt, input, weights, bias)
plt.grid(True, color="grey", linewidth="1.4", linestyle=":")
plt.scatter(class1[:, 0], class1[:, 1], c="red", marker='o', edgecolor='k', s=200, label='Class-1')
plt.scatter(class2[:, 0], class2[:, 1], c="yellow", marker='s', edgecolor='k', s=200, label='Class-2')
plt.scatter(class3[:, 0], class3[:, 1], c="green", marker='^', edgecolor='k', s=200, label='Class-3')
plt.scatter(class4[:, 0], class4[:, 1], c="blue", marker='*', edgecolor='k', s=200, label='Class-4')
plt.title("Input Vector Plot with tentative Decision Boundary")
plot_otho_line(plt, weights, points)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.legend()
plt.grid(linestyle='--')
plt.axis('square')
plt.show()
