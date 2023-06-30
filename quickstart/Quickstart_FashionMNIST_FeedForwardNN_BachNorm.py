'''
BATCH NORMALIZATION :
Batch Norm is just another network layer that gets inserted between
a hidden layer and the next hidden layer. Its job is to take the outputs
from the first hidden layer and normalize them before passing them on as
the input of the next hidden layer.

"linear layer alters the distribution of the data linearly (very
predictably) and not by that much so there is no need to batchnorm
immediately after the linear layer. relu is non linear and it can mess up
the distribution and make it less normal-like, that's why you often use
batchnorm after relu." -Ankur

https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739
https://e2eml.school/batch_normalization.html
https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/

Batch Norm layer also has parameters of its own:
- Two non-learnable parameters (Mean Moving Average and Variance Moving
  Average) are saved as part of the ‘state’ of the Batch Norm layer.
- Two learnable parameters called beta and gamma (scale and shift).

Batch Norm keeps a running count of the Exponential Moving Average (EMA)
of the mean and variance. During training, it simply calculates this EMA
but does not do anything with it. At the end of training, it simply saves
this value as part of the layer’s state, for use during the Inference phase.

The Moving Average calculation uses a scalar ‘momentum’. This is
a hyperparameter that is used only for Batch Norm moving averages and
should not be confused with the momentum that is used in the Optimizer.

BatchNorm behaves differently during training and evaluation/inference
- During training: BatchNorm keeps track of the Moving Average parameters
  of each feature map during training. Thus different batches are
  are normalized using different Moving Average parameters.
- During evaluation/inference: BatchNorm uses the most recent Moving Average
  parameters

BENEFITS OF BATCHNORM:
1- Use Large Learning Rates:
   BatchNorm makes the network more stable during training. This may require
   the use of much larger than normal learning rates, that in turn may further
   speed up the learning process.
2- Less Sensitive to Weight Initialization:
   BatchNorm can make training deep networks less sensitive to the choice of
   weight initialization method.
3- Don’t Use With Dropout:
   Batch normalization offers some regularization effect, reducing
   generalization error, perhaps no longer requiring the use of dropout for
   regularization. Further, it may not be a good idea to use batch
   normalization and dropout in the same network. The reason is that the
   statistics used to normalize the activations of the prior layer may become
   noisy given the random dropping out of nodes during the dropout procedure.



BATCH SIZE
The batch size used during training can also influence the regularization
effect of BatchNorm. Smaller batch sizes tend to introduce more noise in
the batch statistics, which can have a regularizing effect. In contrast,
larger batch sizes provide more accurate statistics, which can reduce
regularization.

'''


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils import plot_train_test_metrics


seed = 42
torch.manual_seed(seed)


'''
1. WORKING WITH THE DATA

 PyTorch provides two data primitives that allow for using
 pre-loaded datasets as well as your own data:
 - torch.utils.data.Dataset
   stores the samples and their corresponding labels
 - torch.utils.data.DataLoader
   wraps an iterable around the Dataset to enable easy access to the samples
   supports automatic batching, sampling, shuffling & multiprocess data loading
'''

# Download training data from open datasets
train_data = datasets.FashionMNIST(root='data',
                                   train=True,
                                   download=True,
                                   transform=ToTensor())

# Download test data from datasets
test_data = datasets.FashionMNIST(root='data',
                                  train=False,
                                  download=True,
                                  transform=ToTensor())


batch_size = 64


# Create data loaders
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


'''
2. CREATING A MODEL

- To accelerate operations, move to GPU or MPS if avaiable
- Define a neural network: create a class that inherits from nn.Module
- Define the layers: do so in the __init__ function
- Specify how data will pass through the network: in the forward function
'''

# Specify device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


# Define model
class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10)
        )

    def forward(self, X):
        x = self.flatten(X)
        logits = self.linear_relu_stack(x)
        return logits


model = MyNN().to(device)
print(model)


'''
3. LOSS FUNCTION AND OPTIMIZER

- Loss functions: https://pytorch.org/docs/stable/nn.html#loss-functions
- Optimization algorithms: https://pytorch.org/docs/stable/optim.html

'''
# Specify loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


'''
4. MODEL TRAINING AND EVALUATION

- Data will be fed using in batches
- Model feeds forward: makes predictions on the data
  and obtains prediction error
- Model backpropegates the prediction error to adjust model parameters
'''


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    correct = 0
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Get prediction and prediction error
        pred = model(X)
        # Increment the number of correct predictions
        correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
        # Calculate the loss between the predicted and true label
        loss = loss_fn(pred, y)
        total_loss += loss

        # Perform backpropegation by computing the gradients of of the model
        # parameters with respect to the loss
        loss.backward()
        # Update the model's parameters based on the computed gradients
        optimizer.step()
        # Reset the gradients of the parameters to zero for the next iteration
        optimizer.zero_grad()

        # Print some metrics every 100 batches
        if batch % 100 == 0:
            loss = loss.item()
            current = (batch + 1) * len(X)
            accuracy = correct / current
            print(
                f'Train Loss: {loss:>5f}, Train accuracy: {accuracy:>5f} [{current:>5d}/{size:>5d}]')

    avg_train_loss = total_loss / num_batches
    overall_accuracy = correct / size
    print(
        f'\nAvg Train Loss: {avg_train_loss:>5f}, Train Accuracy: {overall_accuracy:>5f}')

    return(avg_train_loss, overall_accuracy)


def test(test_dataloader, model, loss_fn, test_type='Test'):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, test_correct = 0, 0
    # Set the model to evaluation mode because BatchNorm layers behave
    # differently during training and evaluation.
    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_correct += (pred.argmax(dim=1) ==
                             y).type(torch.float).sum().item()
    avg_test_loss = test_loss / num_batches
    ovarall_test_accuracy = test_correct / size
    print(
        f'Avg {test_type} Loss: {avg_test_loss:>5f}, {test_type} Accuracy: {ovarall_test_accuracy:>5f}')

    return(avg_test_loss, ovarall_test_accuracy)


# train and evaluate
epochs = 20
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for t in range(epochs):
    print(f'\nEpoch {t+1}\n--------------------')
    # train
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    train_losses.append(train_loss.item())
    train_accuracies.append(train_acc)
    # test
    test_loss, test_acc = test(test_dataloader, model, loss_fn)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
print('\nAll Done!!\n')


# Plot train and test metrics
plot_train_test_metrics(epochs, train_losses, test_losses,
                        train_accuracies, test_accuracies,
                        type_plot='BatchNorm')
