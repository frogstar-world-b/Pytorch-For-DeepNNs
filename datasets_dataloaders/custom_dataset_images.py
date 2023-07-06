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

import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


''' CREATING A CUSTOM DATASET FOR YOUR FILES
A custom Dataset class must implement three functions:
__init__
__len__
__getitem__

In the implementation below, MNIST images are stored in a directory
img_dir, and their labels are stored seperately in a CSV file labels.csv.


The labels_file (labels.csv) looks like this:

tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9

'''


class CustomImageDataset(Dataset):
    def __init__(self, labels_file, img_dir,
                 transform=None, target_transform=None):
        self.img_labels = pd.read_csv(labels_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # pull the image using its index in the labels file
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # read_image reads a JPEG or PNG file into a 3D RGB or grayscale tensor
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def main():
    img_dir = 'custom/img_dir'
    labels_file = 'custom/labels.csv'
    data = CustomImageDataset(labels_file, img_dir)
    print(f'len(data): \n {len(data)} \n')
    fig = plt.figure(figsize=(8, 8))
    for i in range(4):
        img, label = data[i]
        fig.add_subplot(2, 2, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(label)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
