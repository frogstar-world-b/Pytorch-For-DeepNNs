'''
A simple feed-forward neural network (FNN) or multi-layer perceptron (MLP).
It consists of an input layer, two hidden layers, and an output layer.
We use the FashionMNIST dataset with 10 output classes.

Input Layer:
- The input layer does not have any explicit definition in the code.
  Instead, the input data is flattened using nn.Flatten() before being passed
  to the first hidden layer.
- The input size is assumed to be 28x28 (as it is specified in the
  nn.Linear(28 * 28, 512) layer), which corresponds to the size of the
  FashionMNIST images.
Hidden Layers:
- There are two hidden layers with 512 units (neurons) each.
- The activation function used in both hidden layers is the rectified linear
  unit (ReLU) activation function (nn.ReLU()).
- The first hidden layer takes the flattened input as its input and produces
  512 output features.
    The second hidden layer takes the 512 features from the first hidden layer
    as its input and also produces 512 output features
Output Layer:
- The output layer consists of a linear layer (nn.Linear(512, 10)), which takes
  the 512 features from the second hidden layer and produces logits for the 10
  output classes.
- The logits represent the raw predicted values before applying any activation
    function.

The forward method defines the forward pass of the model. It applies the
operations of flattening the input, passing it through the linear and
activation layers of the hidden layers (self.linear_relu_stack), and returning
the logits as the output of the model.

Overall, this architecture is a basic FNN/MLP with two hidden layers and ReLU
activation, designed for the FashionMNIST dataset with 10 output classes.

'''


import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils import plot_some_train_images, plot_train_test_metrics

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

for X, y in test_dataloader:
    print(f"Shape of X[N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")


print(f'Unique class labels: {torch.unique(train_data.targets)}')


# Plot some images in the training data
plot_some_train_images(train_data)


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
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, X):
        x = self.flatten(X)
        logits = self.linear_relu_stack(x)
        return logits


model = MyNN().to(device)
print(model)


# A NOTE ON LOGITS TO PROBABILITIES

# Probabilites can be obtained from logits using softmax activation:
# import torch.nn.functional as F
# probabilities = F.softmax(logits, dim=1)

# However, when using the softmax function for training,
# it is often more suitable to use the cross-entropy loss function
# (torch.nn.CrossEntropyLoss), which combines the softmax operation
# with the computation of the loss.


'''
3. LOSS FUNCTION AND OPTIMIZER

- Loss functions: https://pytorch.org/docs/stable/nn.html#loss-functions
- Optimization algorithms: https://pytorch.org/docs/stable/optim.html
'''

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


# The difference in handling the loss between the train and test functions is
# because in training, the gradients need to be calculated and updated in each
# batch, whereas in testing, only the loss and accuracy metrics are calculated
# for evaluation purposes. Therefore, the accumulation of loss is not necessary
# in the train function.

# The training process is conducted over several epochs.
# During each epoch, the model learns parameters to make better predictions.
# We print the model’s accuracy and loss at each epoch; we’d like to see the
# accuracy increase and the loss decrease with every epoch.


def test(test_dataloader, model, loss_fn, test_type='Test'):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, test_correct = 0, 0
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
print('\nModel training and testing are Done!!\n')


# Plot train and test metrics
plot_train_test_metrics(epochs, train_losses, test_losses,
                        train_accuracies, test_accuracies,
                        type_plot='Original')

'''
5. SAVING THE MODEL

A common way to save a model is to serialize the internal state dictionary
(containing the model parameters).

'''

saved_model_name = 'original_model.pth'
torch.save(model.state_dict(), saved_model_name)
print(f'Saved PyTorch model state to {saved_model_name}')

time.sleep(3)  # Pause for 3 seconds to give time to write to disk

'''
6. LOADING THE MODEL

Loading the model involves recreating the model class structure and loading
the state dictionary into it.
'''

loaded_model = MyNN().to(device)
loaded_model.load_state_dict(torch.load(saved_model_name))


'''
7. MAKING PREDICTIONS FROM A LOADED MODEL
'''

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

loaded_model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    pred_class = pred_class = classes[pred[0].argmax().item()]
    actual_class = classes[y]

print(f'Example prediction for the first item in the test set:')
print(f'Actual: {actual_class}, Predicted: {pred_class}\n')
