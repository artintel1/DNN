import numpy as np
import time

# Generate two 200x200 matrices with random values
matrix_a = np.random.rand(300, 300)
matrix_b = np.random.rand(300, 300)

# Function to multiply matrices using loops
def loop_matrix_multiplication(a, b):
    result = np.zeros((300, 300))
    for i in range(300):
        for j in range(300):
            for k in range(300):
                result[i, j] += a[i, k] * b[k, j]
    return result

# Measure time for loop-based matrix multiplication
start_loop = time.time()
result_loop = loop_matrix_multiplication(matrix_a, matrix_b)
end_loop = time.time()

# Measure time for vectorized matrix multiplication
start_vec = time.time()
result_vec = np.dot(matrix_a, matrix_b)
end_vec = time.time()

# Convert time to milliseconds
time_loop = (end_loop - start_loop) * 1000
time_vec = (end_vec - start_vec) * 1000

# Display results
print(f"Time taken for loop-based multiplication: {time_loop:.6f} milliseconds")
print(f"Time taken for vectorized multiplication: {time_vec:.6f} milliseconds")
print(f"Time difference (loop - vectorized): {time_loop - time_vec:.6f} milliseconds")
print(result_vec[0][0:4])
print(result_loop[0][0:4])