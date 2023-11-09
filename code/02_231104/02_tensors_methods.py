############################################################################
############################################################################
############################################################################
### Artificial Models for Music Creativity: 
### Lesson 2 - 4.11.2023 Alessandro Anatrini



# Introduction 2/2: Tensor

# Yt Video: What's a Tensor?
# https://www.youtube.com/watch?v=f5liqUk0ZTw

# A tensor is a generalization of vectors and matrices to potentially higher dimensions 
# Internally, PyTorch represents tensors as multi-dimensional arrays, similar to the numpy.ndarray
# The name “tensor” comes from the mathematical term used to describe a similar concept.

#Here’s a simple way to understand tensors:
#A 0-dimensional tensor is just a number, or scalar
#A 1-dimensional tensor is an array of numbers, or a vector
#A 2-dimensional tensor is a matrix, which is an array of vectors
#A 3-dimensional tensor can be visualized as a cube of numbers, where each layer of the cube is a matrix
#This pattern continues into higher dimensions
#In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters

# Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators
# In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data

#Tensors are also optimized for automatic differentiation
# If you’re familiar with the backpropagation algorithm for training neural networks, you know that we need to compute gradients of certain quantities
# These gradients are used to update the parameters of the model
# PyTorch uses tensors to make this process more efficient

# Please refer to browser based pytorch tool mm for matrix and tensors visualisation
# https://pytorch.org/blog/inside-the-matrix/


############################################################################
############################################################################
############################################################################

import torch

# Exercise 1: Creating and Manipulating Tensors
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
result = tensor1 + tensor2
print("Exercise 1 - Result:", result)

# Exercise 2: Mathematical Calculations between scalar tensors
x = torch.tensor(5.0)
y = torch.tensor(3.0)
addition = x + y
multiplication = x * y
print("Exercise 2 - Sum:", addition)
print("Exercise 2 - Multiplication:", multiplication)

# Exercise 3: Matrix Operations
matrix1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
matrix2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
#result_matrix = torch.matmul(matrix1, matrix2)
result_matrix = matrix1 @ matrix2
print("Exercise 3 - Matrix Product:")
print(result_matrix)

# Exercise 4: Tensor Size - Returns the size of the tensor
print("Exercise 4 - Tensor Size:")
print("Size of tensor1:", tensor1.size())
print("Size of matrix1:", matrix1.size())

# Exercise 5: Squeeze and Unsqueeze
tensor3 = torch.tensor([[[1, 2, 3]]])
print("Exercise 5 - Squeeze and Unsqueeze:")
print("Original tensor3:", tensor3)
print("Squeezed tensor3:", tensor3.squeeze()) # Removes dimensions of size 1 from the tensor
print("Unsqueezed tensor3:", tensor3.unsqueeze(dim=0)) # Adds a dimension of size 1 to the tensor

# Exercise 6: Reshape - Reshapes the tensor to the specified size
print("Exercise 6 - Reshape:")
print("Reshaped tensor1:", tensor1.reshape(3, 1))

# Exercise 7: Item - Returns the value of a one-element tensor as a standard Python number
# If you have a one-element tensor, for example by aggregating all values of a tensor into one value, 
# you can convert it to a Python numerical value using item()
print("Exercise 7 - Item:")
print("Item of x:", x.item())

# Exercise 8a: To List - Converts the tensor to a list
print("Exercise 8a - To List:")
print("List of tensor1:", tensor1.tolist())

# Exercise 8b - Converts the tensor to a numpy array
# .numpy() converts the tensor to a numpy array 
# This is a very useful method because numpy supports a wider range of operations than PyTorch tensors
# Also, many other libraries such as matplotlib accept numpy arrays as inputs.
# If you want to plot the values of a tensor, you would typically convert it to a numpy array first
# The numpy array and the original tensor share the same underlying memory, which means changes to one will affect the other
# This is a feature designed for memory efficiency, but you need to be careful about unintended side effects
print("Exercise 8b - Convert to Numpy:")
print("Numpy array of tensor1:", tensor1.numpy())

# Exercise 9: Zeroes and Ones 
print("Exercise 9 - Zeroes and Ones:")
print("Zero tensor:", torch.zeros((2, 3))) # Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size
print("One tensor:", torch.ones((2, 3))) # Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size

# Exercise 10: Arange - Returns a 1-D tensor of size (end - start) / step with values from the interval
print("Exercise 10 - Arange:")
print("Arange tensor:", torch.arange(start=0, end=5, step=1))

# Exercise 11: Linspace - Returns a one-dimensional tensor of steps equally spaced points between start and end
print("Exercise 11 - Linspace:")
print("Linspace tensor:", torch.linspace(start=0.1, end=1, steps=10))

# Exercise 12: Eye (Identity Matrix) - Returns a 2-D tensor with ones on the diagonal and zeros elsewhere (identity matrix)
print("Exercise 12 - Eye (Identity Matrix):")
print("Eye tensor:", torch.eye(3))

# Exercise 13: Fill - Fills tensor with a certain value
tensor4 = torch.empty(3, 2)
tensor4.fill_(5)
print("Exercise 13 - Fill (Fills tensor with a certain value):")
print("Filled tensor:", tensor4)

# Exercise 14: Random numbers - Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
print("Exercise 14 - Random numbers:")
print("Random tensor:", torch.rand(3, 2))

# Exercise 15: View
# Returns a new tensor with the same data but of a different shape 
# The returned tensor shares the same data and must have the same number of elements, but may have a different size 
# For a tensor to be viewed, the new view size must be compatible with its original size and stride
tensor5 = torch.arange(10)
print("Exercise 15 - View:")
print("Original tensor5:", tensor5)
print("Reshaped tensor5 using view:", tensor5.view(2, 5))

# Exercise 17: Concatenate - Concatenates the given sequence of tensors in the given dimension
tensor6 = torch.tensor([1, 2, 3])
tensor7 = torch.tensor([4, 5, 6])
print("Exercise 17 - Concatenate:")
print("Concatenated tensor:", torch.cat((tensor6, tensor7)))

# Exercise 18: Stack - Concatenates a sequence of tensors along a new dimension
print("Exercise 18 - Stack:")
print("Stacked tensor:", torch.stack((tensor6, tensor7)))

# Exercise 19: Split - Splits the tensor into chunks
tensor8 = torch.arange(10)
print("Exercise 19 - Split:")
print("Split tensor:", torch.split(tensor8, 2))

# Exercise 20: Chunk - Splits a tensor into a specific number of chunks
print("Exercise 20 - Chunk:")
print("Chunked tensor:", torch.chunk(tensor8, 5))