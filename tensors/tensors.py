'''
Tensors are specialized data structures that are similar to arrays & matrices.
In PyTorch, tensors are used to encode the inputs and outputs of a model,
as well as the model's parameters.

Tensors are similar to NumPy's ndarrys, except that tensors can run on GPUs
or ther hardware accelerators.

Tensors and NumPy arrays can often share the same underlying memory
eliminating the need to copy data.

Tensors are optimized for automatic differentiation.
'''


import torch
import numpy as np


seed = 42
torch.manual_seed(seed)


''' INITIALIZING A TENSOR '''

# From a list
data = [[1, 2], [3, 4]]
print(f'data list: \n {data} \n')
x_data = torch.tensor(data)
print(f'Tensor from list: \n {x_data} \n')

# From a numpy array
arr = np.array(data)
x_np = torch.from_numpy(arr)
print(f'Tensor from numpy array: \n {x_np} \n')

# From another tensor
x_ones = torch.ones_like(x_data)  # reatins the properties of x_data
print(f'Ones Tensor:\n {x_ones} \n')

# overrides the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f'Random Tensor: \n {x_rand} \n')

# With random or constant values
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f'Random Tensor with shape {shape}: \n {rand_tensor} \n')
print(f'Ones Tensor with shape {shape}: \n {ones_tensor} \n')
print(f'Zeros Tensor with shape {shape}: \n {zeros_tensor} \n')

# Notice that shape can be a tuple or not!
assert torch.equal(torch.ones(2, 3), ones_tensor)

''' ATTRIBUTES OF A TENSOR
Attributes of a tensor describe their shape datatype, and the device on which
they are stored.
'''

tensor = torch.rand(3, 4)
print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}')


''' OPERATIONS ON TENSORS
There are over 100 operations: arithmetic, linear algebra, matrix ops,
indexing, slicing, sampling, etc.
See: https://pytorch.org/docs/stable/torch.html

Each of these operations can be run on a GPU at typically higher speeds
than on a CPU. Although by default, tensors are created on the CPU.
So you will need to explicitly move tensor to the GPU using the `.to`
method provided a GPU is available.

Copying large tensors across devices can be expensive in terms of time
and memory!
'''

# Move tensor to GPU (on Mac M2)
if torch.backends.mps.is_available():
    tensor = tensor.to('mps')
print(f'Device tensor is updated to: {tensor.device}')

# Standard numpy-like slicing
tensor = torch.ones(3, 4)
print(f'Tensor: \n {tensor} \n')
print(f'First row: {tensor[0]}')
print(f'First column: {tensor[:, 0]}')
print(f'Last column: {tensor[..., -1]}')
tensor[:, 1] = 0
print(f'Set 2nd column to zeros: \n {tensor} \n')

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor, tensor])
print(f'Concat 4 tensors (along dim = 0): \n {t1} \n')
t2 = t1 = torch.cat([tensor, tensor, tensor, tensor], dim=1)
print(f'Concat 4 tensors (along dim = 1): \n {t2} \n')

# Matrix multiplication
y1 = tensor @ tensor.T  # tensor.T is transpose
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)  # need to define y3 before using it in the next line
torch.matmul(tensor, tensor.T, out=y3)
print(f'Matrix multiplication tensor @ tensor.T: \n {y1} \n')
assert torch.equal(y1, y2)
assert torch.equal(y1, y3)

# Element-wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(z1)
torch.mul(tensor, tensor, out=z3)
assert torch.equal(z1, z2)
assert torch.equal(z1, z3)
print(f'Element-wise product tensor * tensor: \n {z1} \n')

# Single-element tensor
agg = tensor.sum()
agg_item = agg.item()
print(f'Sum of tensor: \n {agg} \n')
print(f'tensor.sum().item(): {agg_item}, type: {type(agg_item)}')

# In place operations (denoted by a _ suffix, e.g. x.copy_(y) will change x)
print(f'tensor: {tensor} \n')
tensor.add_(5)
print(f'tensor.add_(5): \n {tensor} \n')


''' BRIDGE WITH NUMPY
Tensors on the CPU and NumPy arrays can share their underlying memory locations
and changing one will change the other.
'''

# Tensor to numpy array
t = torch.ones(5)
print(f't: \n {t} \n')
n = t.numpy()
print(f'n: \n {n} \n')
# a change in the tensor reflects in the numpy array
t.add_(1)
print(f't: \n {t} \n')
print(f'n: \n {n} \n')


# Numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)
# a change in the numpy array reflects in the tensor
np.add(n, 1, out=n)
print(f'n: \n {n} \n')
print(f't: \n {t} \n')

# However, writing over the array does not change the tensor
n = np.ones(5)
t = torch.from_numpy(n)
n = n + 1
print(f'n: \n {n} \n')
print(f't: \n {t} \n')
