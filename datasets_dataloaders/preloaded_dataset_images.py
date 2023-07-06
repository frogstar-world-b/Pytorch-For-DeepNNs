''' Code for processing data samples can get messy and hard to maintain.
Ideally, we want our dataset code to be decoupled from our model training
code for better readability and modularity.

`torch.utils.data.DataLoader and torch.utils.data.Dataset are two
fundamental data building blocks that enable using pre-loaded dataset as
well as using your own data.

- Dataset stores the samples and their corresonding labels.
- Dataloader wraps an iterable around the Dataset to enable easy access
  to the samples.

  See a full list of pre-loaded PyTorch datasets here:
  https://pytorch.org/vision/stable/datasets.html
'''

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


''' LOADING A DATASET
Let's work with Fashion-MNIST dataset from TorchVision.
- training: 60,000
- test: 10,000
- Each example comprises a 28x28 grayscale image and a label,
  from one to 10 classes
'''

# root: is the path to where the train/test data are stored
# train: boolean (train or test)
# download: boolean (if True, downloads data if not available at root)
# transform and target_transform: specify the feature and label transformation
training_data = datasets.FashionMNIST(root='data',
                                      train=True,
                                      download=True,
                                      transform=ToTensor())


test_data = datasets.FashionMNIST(root='data',
                                  train=False,
                                  download=True,
                                  transform=ToTensor())

''' ITERATING AND VISUALIZATING THE DATASET '''
labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(labels_map[label])
    plt.axis('off')
plt.show()

''' PREPARING A DATASET FOR TRAINING WITH DATALOADERS
The Dataset retrieves the dataset's features and labels one at a time.
While training a model, we typically want to pass samples in "minibatches",
reshuffle the data at every epoch to reduce model overfitting, and use Python's
`multiprocessing` to speed up data retrieval.

DataLoader is an iterable that abstracts this complexity in an easy API.
'''

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=True)

# Iterate through the DataLoader
train_features, train_labels = next(iter(train_dataloader))
print(f'Feature batch shape: {train_features.size()}')
print(f'Labels batch shape: {train_labels.size()}')
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap='gray')
plt.title(f'Label: {labels_map[label.item()]}')
plt.axis('off')
plt.show()
