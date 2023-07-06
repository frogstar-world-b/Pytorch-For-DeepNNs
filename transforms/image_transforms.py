''' All TorchVision dataset have two parameters:
- transform: to modify features
- target_transform: to modify lables

Both accept callables containing the transformation logic.

Full list of possible image transforms is here:
https://pytorch.org/vision/stable/transforms.html
'''

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda


'''
The FashionMNIST features are in PIL Image format, and the labels are integers.
For training, we need the features as normalized tensors, and the labels as
one-hot encoded tensors. To make these transformations, we use ToTensor and
Lambda.

ToTensor()
Converts a PIL image o NumPy ndarray into a FloatTensor and scales the image's
pixel intensity values in the range [0., 1.].

Lambda Transforms
Applies a user-defined lambda function. Below we define a function to turn the
integer labels into a one-hot encoded tensor. First, a zero tensor of size 10
is created, where 10 is the number of labels in FashionMNIST. Then scatter_ is
called, which assigns a value=1 on the index geven the label y.
'''


ds = datasets.FashionMNIST(root='data',
                           train=True,
                           download=True,
                           transform=ToTensor(),
                           target_transform=Lambda(lambda y: torch.zeros(10,
                           dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

print(ds[0])

# Each sample of the dataset has two tensors, for features and target
assert len(ds[0]) == 2
# The normalized pixel values should be in [0., 1.]
assert torch.all((ds[0][0]) <= 1 & (ds[0][0] >= 0))
# There should be 10 one-hot encoded values in the target tensor
assert len(ds[0][1]) == 10
