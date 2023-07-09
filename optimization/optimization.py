''' STEPS:
1- Load and transform the data as tensors
2- Process with DataLoader to ready for modeling
3- Define the NN class
4- Set values of hyperparameters (e.g. epochs, batch_size, learning_rate)
5- Define the loss function
6- Define the optimizer algorithm
7- Optimization loop, which includes Train and test loops
'''


# Prerequisite code

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data)
test_dataloader = DataLoader(test_data)


class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = MyNN()
print(model)


''' HYPERPARAMETERS
- Number of epochs: the number of times to iterate over the dataset
- Batch Size, Learning Rate: number of data samples propegated through the
  network before the parameters are updated
- Learning rate: how much to update the model prameters at each batch/epoch.
  Smaller values yield slow learning speed, while large ones may result in
  unpredictable behavior during training.
'''

learning_rate = 1e-3
batch_size = 64
epochs = 5

''' OPTIMIZATION LOOP

Optimization is the process of adjusting model parameters to reduce the error
in each training step.

Each iteration of the optimization loop is called an epoch and contains two
main parts:
1- The train loop - iterate over the training dataset and try to convrge to
   optimal parameters
2- The validation/test loop - iterate over a test set to check model
   performance

Inside the training loop, optimization happens in three steps:
- Call `optimizer.zero_grad() to reset the gradients of the model parameters.
  Gradients by default add up; so to prevent double-counting, we explicitly
  zero them at each iteration.
- Backpropagate the prediction loss with a call to `loss.backward()`.
  PyTorch deposits the gradients of the loss w.r.t. each parameter.
- Once we have our gradients, we call the `optimizer.step() to adjust the
  parameters by the gradients collected in the backward pass.
'''

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer: initialized by registering model parameters entended for training,
#            and passing the learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


'''OPTIMIZATION LOOK IMPLEMENTATION'''


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # compute predictions and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropegate
        loss.backward()  # get gradients
        optimizer.step()  # adjust parameter values
        optimizer.zero_grad()  # zero gradients to prevent double-counting

        # print key iteration metrics
        if batch % 100 == 0:
            loss, current_sample = loss.item(), (batch + 1) * len(X)
            print(f'Loss: {loss:>5f} [{current_sample:>4d}/{size:>5d}]')


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = 0
    # ensure no gradients are computed during test mode
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_class = pred.argmax(1)
            correct += (pred_class == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
    print(f'Test Metrics: \n Accuracy: {100*correct:>0.1f}%, Avg Loss: {test_loss:>5f}\n')


train_loop(train_dataloader, model, loss_fn, optimizer)
test_loop(test_dataloader, model, loss_fn)
