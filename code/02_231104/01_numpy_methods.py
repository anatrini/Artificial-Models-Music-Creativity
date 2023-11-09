############################################################################
############################################################################
############################################################################
### Artificial Models for Music Creativity: 
### Lesson 2 - 4.11.2023 Alessandro Anatrini



# Introduction 1/2: NumPy array

# NumPy, which stands for Numerical Python, is a library for the Python programming language, 
# adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays

# A NumPy array is a grid of values, all of the same type, and is indexed by a tuple of non-negative integers
# The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension

# NumPy arrays are more efficient than lists for a few reasons:
# Memory: A NumPy array is more memory-efficient than a Python list because it takes less space in memory
# This is because a NumPy array is densely packed in memory due to its homogeneous type, 
# while Python lists are arrays of pointers to objects, even when all of them are of the same type

# Speed: You often don’t need to iterate over the elements of a NumPy array using a loop, which can be slow in Python
# Instead, you can use operations that apply to the array as a whole (element-wise operations), which are implemented internally in C, making them much faster.

# Convenience: With NumPy, you can perform operations on entire arrays without the need for Python for loops
# These operations are faster and require less code, making your programs more readable and easier to understand

# NumPy arrays cannot be directly used for GPU computation like PyTorch tensors can
# They also lack the automatic differentiation features of PyTorch tensors, which are crucial for training machine learning models
# But for many scientific computing tasks, NumPy is more than sufficient and is a very powerful tool to have in your toolkit

# Automatic differentiation is a key feature in many machine learning libraries, including PyTorch 
# It’s a way of automatically computing the derivatives of the function that you’re optimizing 
# In the context of machine learning, this function is often a loss function, and the parameters of this function are the weights of your model
# When you’re training a model, you want to find the parameters that minimize the loss function
# To do this, you need to know the gradient of the loss function with respect to the parameters, which tells you how to update the parameters to reduce the loss
# Computing these gradients can be quite complex, especially for large models with many parameters
# This is where automatic differentiation comes in. You simply define the forward pass of your model (i.e., how to go from the inputs to the outputs), and PyTorch takes care of computing the gradients for you
# This is done using the backward() function in PyTorch. When you call backward() on a tensor, PyTorch computes the gradient of that tensor with respect to all the other tensors that it depends on (and that have requires_grad=True)
# These gradients are then stored in the .grad attribute of those other tensors
# This feature is what allows us to easily implement and train complex models in PyTorch
# It abstracts away the details of the backpropagation algorithm, letting us focus on designing our models
# It’s a powerful tool in the machine learning practitioner’s toolbox!

import numpy as np

# Create a rank 1 array from a Python list
a = np.array([1, 2, 3])
print("a:", a)

# Create a rank 2 array from a nested Python list
b = np.array([[1, 2, 3], [4, 5, 6]])
print("b:", b)


# Create a 3x3 array of zeros
zeros = np.zeros((3, 3))
print("Zeros:\n", zeros)

# Create a 3x3 array of ones
ones = np.ones((3, 3))
print("Ones:\n", ones)


# Create an array with values from 0 to 4
sequence = np.arange(5)
print("Sequence:", sequence)


# Create an array with 5 values evenly spaced between 0 and 1
linspace = np.linspace(0, 1, 5)
print("Linspace:", linspace)


# Reshape the sequence array to a 2x2 array
sequence = np.arange(16)
reshaped = sequence.reshape((4, 4))
print("Reshaped:\n", reshaped)


# Create a 2x2 array of random values
random_values = np.random.rand(2, 2)
print("Random values:\n", random_values)


# np.dot()
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

dot_product = np.dot(a, b)
print("Dot product:\n", dot_product)


# np.transpose()
a = np.array([[1, 2], [3, 4]])

transpose = np.transpose(a)
print("Transpose:\n", transpose)


# np.linalg.inv()
a = np.array([[1, 2], [3, 4]])

inverse = np.linalg.inv(a)
print("Inverse:\n", inverse)


# np.mean()
a = np.array([[1, 2], [3, 4]])

mean = np.mean(a)
print("Mean:", mean)


# np.std()
a = np.array([[1, 2], [3, 4]])

std_dev = np.std(a)
print("Standard Deviation:", std_dev)


# np.var()
a = np.array([[1, 2], [3, 4]])

variance = np.var(a)
print("Variance:", variance)


# np.sum()
a = np.array([[1, 2], [3, 4]])

sum = np.sum(a)
print("Sum:", sum)


# np.max(), np.min()
a = np.array([[1, 2], [3, 4]])

max_value = np.max(a)
min_value = np.min(a)
print("Max:", max_value)
print("Min:", min_value)