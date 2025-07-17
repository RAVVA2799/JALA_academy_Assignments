import numpy as np



#question1
# Create a 3x3 array with random integers between 0 and 10
arr = np.random.randint(0, 11, size=(3, 3))

# Calculate sum, mean, and standard deviation
arr_sum = np.sum(arr)
arr_mean = np.mean(arr)
arr_std = np.std(arr)

print("Array:\n", arr)
print("Sum:", arr_sum)
print("Mean:", arr_mean)
print("Standard Deviation:", arr_std)


#question2
# Create a 1D array of 10 elements
arr_1d = np.arange(10)

# Compute the cumulative sum
cumulative_sum = np.cumsum(arr_1d)

print("1D Array:", arr_1d)
print("Cumulative Sum:", cumulative_sum)

#question3
# Generate two 2x3 arrays with random integers between 1 and 10
array1 = np.random.randint(1, 11, size=(2, 3))
array2 = np.random.randint(1, 11, size=(2, 3))

# Element-wise operations
addition = array1 + array2
subtraction = array1 - array2
multiplication = array1 * array2
division = array1 / array2

print("Array 1:\n", array1)
print("Array 2:\n", array2)
print("Addition:\n", addition)
print("Subtraction:\n", subtraction)
print("Multiplication:\n", multiplication)
print("Division:\n", division)

#question4
# Create a 4x4 identity matrix
identity_matrix = np.eye(4)
print("4x4 Identity Matrix:\n", identity_matrix)

#question5
a = np.array([5, 10, 15, 20, 25])
result = a / 5
print("Original array:", a)
print("Array after division by 5:", result)


#question6
arr_1d_12 = np.arange(12)
reshaped_matrix = arr_1d_12.reshape(3, 4)
print("1D Array of 12 elements:", arr_1d_12)
print("Reshaped to 3x4 matrix:\n", reshaped_matrix)

#question7
matrix_3x3 = np.random.randint(0, 10, size=(3, 3))
flattened_array = matrix_3x3.flatten()
print("3x3 Matrix:\n", matrix_3x3)
print("Flattened 1D Array:", flattened_array)

#question8
# Generate two 3x3 matrices with random integers between 1 and 10
mat1 = np.random.randint(1, 11, size=(3, 3))
mat2 = np.random.randint(1, 11, size=(3, 3))

# Stack horizontally
h_stack = np.hstack((mat1, mat2))

# Stack vertically
v_stack = np.vstack((mat1, mat2))

print("Matrix 1:\n", mat1)
print("Matrix 2:\n", mat2)
print("Horizontally Stacked:\n", h_stack)
print("Vertically Stacked:\n", v_stack)

#question9
# Concatenate two arrays of different sizes along a new axis
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5])

concatenated = np.stack((arr1, np.pad(arr2, (0, arr1.size - arr2.size), 'constant')), axis=0)
print("Array 1:", arr1)
print("Array 2:", arr2)
print("Concatenated along new axis:\n", concatenated)

#question10
# Create a 3x2 matrix
matrix_3x2 = np.random.randint(1, 11, size=(3, 2))
print("Original 3x2 Matrix:\n", matrix_3x2)

# Transpose the matrix
transposed = matrix_3x2.T
print("Transposed Matrix:\n", transposed)

# Reshape the transposed matrix to have 3 rows and 2 columns
reshaped = transposed.reshape(3, 2)
print("Reshaped to 3x2 Matrix:\n", reshaped)

#question11
arr_15 = np.arange(15)
extracted_elements = arr_15[2:11:2]
print("Original 1D Array:", arr_15)
print("Extracted elements (positions 2 to 10, step 2):", extracted_elements)


#question12
matrix_5x5 = np.random.randint(1, 21, size=(5, 5))
sub_matrix = matrix_5x5[1:4, 2:5]
print("5x5 Matrix:\n", matrix_5x5)
print("Extracted Sub-matrix (rows 1-3, columns 2-4):\n", sub_matrix)

#question13
arr_replace = np.array([5, 12, 7, 15, 3, 11, 8])
arr_replace[arr_replace > 10] = 10
print("Original array with values > 10 replaced by 10:", arr_replace)

#question14
arr_fancy = np.arange(8)
selected_elements = arr_fancy[[0, 2, 4, 6]]
print("Original array:", arr_fancy)
print("Elements at positions [0, 2, 4, 6]:", selected_elements)

#question15
arr_reverse = np.arange(10)
reversed_arr = arr_reverse[::-1]
print("Original array:", arr_reverse)
print("Reversed array:", reversed_arr)

#question16
matrix = np.random.randint(1, 10, size=(3, 3))
row_array = np.array([1, 2, 3])
result = matrix + row_array
print("3x3 Matrix:\n", matrix)
print("1x3 Array:", row_array)
print("Result after broadcasting addition:\n", result)

#question17
arr_5 = np.arange(5)
scalar = 4
multiplied = arr_5 * scalar
print("Original array:", arr_5)
print("Array after multiplication by scalar (4):", multiplied)

#question18
matrix_3x3 = np.random.randint(1, 10, size=(3, 3))
col_vector = np.array([[2], [4], [6]])
result = matrix_3x3 - col_vector
print("3x3 Matrix:\n", matrix_3x3)
print("3x1 Column Vector:\n", col_vector)
print("Result after broadcasting subtraction:\n", result)

#question19
# Create a 3D array of shape (2, 3, 4) with random integers
array_3d = np.random.randint(1, 10, size=(2, 3, 4))
scalar = 5
broadcasted_result = array_3d + scalar

print("Original 3D Array:\n", array_3d)
print("Scalar to add:", scalar)
print("Result after broadcasting addition:\n", broadcasted_result)

#question20
arr_a = np.random.randint(1, 10, size=(4, 1))
arr_b = np.random.randint(1, 10, size=(1, 5))
broadcasted_sum = arr_a + arr_b

print("Array of shape (4, 1):\n", arr_a)
print("Array of shape (1, 5):\n", arr_b)
print("Result after broadcasting addition:\n", broadcasted_sum)

#question21
matrix_2d = np.random.randint(1, 16, size=(3, 3))
sqrt_matrix = np.sqrt(matrix_2d)
print("Original 2D Array:\n", matrix_2d)
print("Square root of each element:\n", sqrt_matrix)

#question22
arr1 = np.random.randint(1, 10, size=5)
arr2 = np.random.randint(1, 10, size=5)
dot_product = np.dot(arr1, arr2)
print("Array 1:", arr1)
print("Array 2:", arr2)
print("Dot Product:", dot_product)

#question23
arr1 = np.random.randint(1, 10, size=5)
arr2 = np.random.randint(1, 10, size=5)
comparison = arr1 > arr2
print("Array 1:", arr1)
print("Array 2:", arr2)
print("Element-wise comparison (arr1 > arr2):", comparison)

#question24
array_2d = np.random.randint(1, 10, size=(3, 4))
doubled_array = array_2d * 2
print("Original 2D Array:\n", array_2d)
print("Doubled Array:\n", doubled_array)

#question25
arr_100 = np.random.randint(1, 101, size=100)
even_sum = np.sum(arr_100[arr_100 % 2 == 0])
print("1D Array of 100 random integers:", arr_100)
print("Sum of all even numbers:", even_sum)

#question26
matrix_3x3 = np.random.randint(1, 10, size=(3, 3))
determinant = np.linalg.det(matrix_3x3)
print("3x3 Matrix:\n", matrix_3x3)
print("Determinant:", determinant)

#question27
matrix_2x2 = np.random.randint(1, 10, size=(2, 2))
inverse_matrix = np.linalg.inv(matrix_2x2)
identity_check = np.dot(matrix_2x2, inverse_matrix)

print("2x2 Matrix:\n", matrix_2x2)
print("Inverse Matrix:\n", inverse_matrix)
print("Product (should be identity):\n", identity_check)

#question28
matrix_2x2_eig = np.random.randint(1, 10, size=(2, 2))
eigenvalues, eigenvectors = np.linalg.eig(matrix_2x2_eig)
print("2x2 Matrix:\n", matrix_2x2_eig)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

#question29
# Solve the system of equations:
# 2x + 3y = 5
# 4x + 6y = 10

coefficients = np.array([[2, 3], [4, 6]])
constants = np.array([5, 10])

solution = np.linalg.solve(coefficients, constants)
print("Solution for x and y:", solution)

#question30
# Perform SVD on a 3x3 matrix and reconstruct the original matrix
svd_matrix = np.random.randint(1, 10, size=(3, 3))
U, S, Vt = np.linalg.svd(svd_matrix)
S_matrix = np.zeros((3, 3))
np.fill_diagonal(S_matrix, S)
reconstructed = np.dot(U, np.dot(S_matrix, Vt))

print("Original 3x3 Matrix:\n", svd_matrix)
print("U Matrix:\n", U)
print("Singular Values:", S)
print("Vt Matrix:\n", Vt)
print("Reconstructed Matrix:\n", reconstructed)
