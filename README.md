# Pytorch-For-DeepNN

In a typical machine learning workflow, one has to deal with preprocessing data, constructing models, optimizing parameters, and storing the finalized models. This guide presents a comprehensive machine learning pipeline implemented in PyTorch, based on tutorials available on (pytorch.org)[https://pytorch.org/tutorials/beginner/basics/intro.html] -- except the examples are expanded and  executable scripts are used here.  On the plus side, they are full of docstring nuggets that dive into key motivations and considerations for each piece of code.

Here are the topics covered,

1. [quickstart](quickstart/): Demonstrations of how to build a feed-forward neural network, conduct various flavors of regularization, and save/load your models.
2. [tensors](tensors/): Includes initializing tensors, operations on tensor, bridging with Numpy
3. [datasets_dataloaders](datasets_dataloaders/): Data preprocessing using `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`
4. [transforms](transforms/): How to perform some manipulation of the data and make it suitable for training
5. [build_model](build_model/): Build a neural network to classify images in the FashionMNIST dataset
6. [autograd](autograd/): How to compute (or even disable) PyTorch's a built-in differentiation engine `torch.autograd`
7. [optimization](optimization/): Optimizing model parameters
8. [warmstart](warmstart/): Warm-starting a model using parameters from a different model in PyTorch 
