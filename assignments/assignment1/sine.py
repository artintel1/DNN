import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(0)

# Generate 20 points along the x-axis
x = np.linspace(0, 2 * np.pi, 20)

# Generate the sine wave
y = np.sin(x)

# Add random noise to the sine wave
noise = np.random.normal(0, 0.1, size=y.shape)
y_noisy = y + noise

# Fit a polynomial of degree 19 (n-1) to pass through all points
degree = len(x) - 1
coefficients = np.polyfit(x, y_noisy, degree)

# Generate fitted values using the polynomial
polynomial = np.poly1d(coefficients)
y_fit = polynomial(x)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Original Sine Wave', color='blue', linewidth=2)
plt.scatter(x, y_noisy, label='Noisy Data', color='red')
plt.plot(x, y_fit, label='Interpolated Polynomial', color='green', linewidth=2)
plt.title('Polynomial Interpolation Through Noisy Data Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

# Print the polynomial coefficients
print("Polynomial coefficients:")
print(coefficients)
