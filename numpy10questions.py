import numpy as np

#question1
# Create a NumPy array of random numbers
arr = np.random.rand(10)

# Min-max normalization
normalized_arr = (arr - arr.min()) / (arr.max() - arr.min())

print("Original array:", arr)
print("Normalized array:", normalized_arr)

#question2
# Generate a 5x5 matrix with random integers between 0 and 9
matrix = np.random.randint(0, 10, (5, 5))
print("Original matrix:\n", matrix)

# Replace all occurrences of a specific value (e.g., 5) with another value (e.g., -1)
specific_value = 5
replacement_value = -1
matrix[matrix == specific_value] = replacement_value

print("Modified matrix:\n", matrix)

#question3
# Create two NumPy arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

# Element-wise operations
addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b

print("Addition:", addition)
print("Subtraction:", subtraction)
print("Multiplication:", multiplication)
print("Division:", division)

#question4
# Solve the system of equations:
# 3x + 4y = 7
# 5x + 2y = 8

coefficients = np.array([[3, 4], [5, 2]])
constants = np.array([7, 8])

solution = np.linalg.solve(coefficients, constants)
print("Solution (x, y):", solution)

#question5
# Create a 3x3 matrix and a 1D array
matrix_3x3 = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
array_1d = np.array([10, 20, 30])

# Broadcasting: add the 1D array to each row of the matrix
result = matrix_3x3 + array_1d

print("3x3 Matrix:\n", matrix_3x3)
print("1D Array:", array_1d)
print("Result of broadcasting addition:\n", result)

#question6
# Create a 3x3 identity matrix
identity_matrix = np.eye(3)
print("3x3 Identity Matrix:\n", identity_matrix)

#question7
# Perform matrix multiplication between two 2D arrays
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
matrix_product = np.dot(A, B)
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix multiplication result:\n", matrix_product)

#question8
# Calculate the dot product and cross product of two vectors
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

dot_product = np.dot(vec1, vec2)
cross_product = np.cross(vec1, vec2)

print("Vector 1:", vec1)
print("Vector 2:", vec2)
print("Dot product:", dot_product)
print("Cross product:", cross_product)

#question9
# Generate a NumPy array of 20 random integers between 0 and 9
random_integers = np.random.randint(0, 10, 20)
unique_elements = np.unique(random_integers)

print("Random integers array:", random_integers)
print("Unique elements:", unique_elements)

#question10
def inverse_matrix(matrix):
    return np.linalg.inv(matrix)

# Example usage:
sample_matrix = np.array([[1, 2], [3, 4]])
inv = inverse_matrix(sample_matrix)
print("Original matrix:\n", sample_matrix)
print("Inverse matrix:\n", inv)