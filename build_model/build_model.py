import torch
from torch import nn


''' SPECIFY DEVICE '''

# Get the device for training
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

print(f'Using {device} device.\n')


''' DEFINE THE NN CLASS

We define our NN by
- subclassing nn.Module
- initializing the network layers in __init__
- every nn.Module subclass implements the operation on input data in the
  `forward` method

To use the model, we pass it the input data. This executes the model's
`forward`, along with some background operations.
'''


class My_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        return logits


# Create an insance of NN, and move it into device
model = My_NN().to(device)
print(model)

# Expected out:
# My_NN(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear_relu_stack): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#   )
# )


# Call the UNTRAINED model on an input (one channel, 28 x 28 pixels)
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probabilities = nn.Softmax(dim=1)(logits)
y_pred = pred_probabilities.argmax(dim=1)
print(f'Logits: \n {logits} \n')
print(f'Predicted probabilities: \n {pred_probabilities} \n')
print(f'Predicted class: {y_pred.item()} \n')


''' MODEL LAYERS

We'll break down the model layers in My_NN. For this illustration, we will
use a sample minibatch of 3 images of size 28x28.

check out torch.nn API docs: https://pytorch.org/docs/stable/nn.html
'''
print('Example using a sample minibatch of 3 images of size 28x28:')
input_images = torch.rand(3, 28, 28)
print(f'Minibatch images size: {input_images.size()}')

# nn.Flatten converts each 2D 28x28 image into a contiguous array of 784 pixel
# values. The minibatch dimension (at dim=0) is maintained.

flatten = nn.Flatten()
flat_images = flatten(input_images)
print(f'Size of flattened minibatch: {flat_images.size()}')

# nn.Linear is a linear layer that applies linear transformation on the input
# using stored wights and biases
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
hidden1 = layer1(flat_images)
print(f'First hidden layer size: {hidden1.size()} \n')

# nn.ReLU is used to introduce non-linearty in the model
# see https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
print(f'Before ReLU: \n {hidden1}\n')
hidden1 = nn.ReLU()(hidden1)
print(f'After ReLU: \n {hidden1}\n')

# nn.Sequential is an ordered container of models.
seq_models = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10))  # output layer

logits = seq_models(input_images)
print(f'Logits of images: \n {logits}')

# nn.Softmax:
# The last linear layer returns raw logits [-infty, infty].
# nn.Softmax scales those to [0, 1] representing the models predicted
# probabilities for each class.
# dim parameter indicates the dimension along which the values must sum to 1

softmax = nn.Softmax(dim=1)
pred_prob = softmax(logits)

print(f'Predicted probabilities of images for each class: \n {pred_prob}\n')

# Model parameters:
# subclassng nn.Module automatically tracks all fields defined inside your
# model object, and makes all parameters accessible using the parameters()
# or named_parameters() methods.

print(f'Model structure: {model}\n')
for name, param in model.named_parameters():
    print(f'Layer: {name} | Size: {param.size} | Values: {param[:2]} \n')
