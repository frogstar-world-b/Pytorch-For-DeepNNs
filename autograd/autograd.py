''' AUTOMATIC DIFFERENTIATION WITH TORCH.AUTOGRAD

torch.autograd is torch's built-in differentiation engine. It supports
automatic computation of gradient of any computational graph.

We will begin by considering the simplest one-layer NN, with inuput x,
parameters w and b, and some loss function.


x --->  *  --->  +  ---> z   ----> cross-entropy  ---->  loss

        ^        ^                        ^
        |        |                        |

        w        b                        y

In this network, w and b are parameters that we need to optimize. Thus,
we need to be able to compute the gradients of the loss function with
respect to those variables. In order to do that:
1- We can set the `requires_grad` property of those tesnors when creating them
2- We can do that later using the `x.require_grad(True)` method.

Leaves: x, w, b
Roots(s): z, loss
'''


import torch


# Write the code the defines the computational graph above
x = torch.ones(5)  # expecte input
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(loss)


# To construct the computational graph, a function of the class Function
# is applied to the tensor. It computes the function in the forward
# direction and calculates its derivatives in the backwarwd propagation step.
# A reference to the backward prop function is stored in `grad_fn`:
print(f'Gradient function for z = {z.grad_fn}')
print(f'Gradient function for loss = {loss.grad_fn}')


''' COMPUTING GRADIENTS
In our example, to optimize the parameters in the NN, we need to compute
the derivatives of the loss function with respect to the parameters:
- partial d loss / dw
- partial d loss/ dw
under some fixed values of x and y.
To compute those derivatives, we call `loss.backward()`, and then retrieve
the values from `w.grad`, and `b.grad`.
'''

loss.backward()
print(w.grad)
print(b.grad)


# NOTE 1:
# We can only obtain `grad` for the leaf nodes of the computational graph,
# which have `requires_grad` set to `True`.
# NOTE 2:
# We can only perform grad calculations using `backward` ONCE on a given graph
# for performance reasons. If we need to do several `backward` calls on the
# same graph, we need to pass `retain_graph=True` to the `backward` call.


''' DISABLING GRADIENT TRACKING '''

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# Another way to achieve the same result is to use the `detach` method on
# the tensor
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)


''' FORWARD AND BACKWARD PASSES

In the DAG, leaves are input tensors, and roots are the output tensors.
By tracing the graph from roots to leaves, you can automatically compute the
gradients using the chain rule.

In a FORWARD pass, autograd does two things simultaneously:
- run the requested operations to compute the resulting tensor
- maintain the operation's gradient function in the DAG

The BACKWARD pass kicks off when `.backward()` is called on the DAG root.
`autograd` then:
- computes the gradient from each .grad_fn
- accumulates them in the respective tensor's `.grad` attribute
- using the chain rule, propagates all the way to the leaf tensors.

'''

input_tensor = torch.eye(4, 5, requires_grad=True)
output_tensor = (input_tensor + 1).pow(2).t()
print(f'Input tensor: \n{input_tensor}')
print(f'Output tensor: \n{output_tensor}')

output_tensor.backward(torch.ones_like(output_tensor), retain_graph=True)
print(f'First call \n{input_tensor.grad}')

output_tensor.backward(torch.ones_like(output_tensor), retain_graph=True)
print(f'Second call \n{input_tensor.grad}')

# zero the gradients in-place
input_tensor.grad.zero_()
output_tensor.backward(torch.ones_like(output_tensor), retain_graph=True)
print(f'Call after zeroing gradients \n{input_tensor.grad}')

# Notice that when we call `backward` for the second time with the same
# argument the value of the gradient is different. This happens because when
# doing backward propagation, PyTorch accumulates the gradients, i.e. the value
# of computed gradients is added to the grad property of all leaf nodes of
# computational graph. If you want to compute the proper gradients, you need to
# zero out the grad property before. In real-life training an optimizer helps
# us to do this.

# NOTE: Previously, we were calling `backward()` function without parameters.
# This is essentially equivalent to calling `backward(torch.tensor(1.0))`,
# which is a useful way to comute the gradients in case of a scalar-valued
# function, such as loss during NN training.
