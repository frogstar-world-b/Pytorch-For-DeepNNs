import matplotlib.pyplot as plt
import os


def plot_some_train_images(train_data):
    # Plot some images in the training data
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    plt.subplots_adjust(hspace=.5)  # adjusts vertical spacing
    for i in range(16):
        image, label = train_data[i]
        axs[i // 4, i % 4].imshow(image.squeeze(), cmap='gray')
        axs[i // 4, i % 4].set_title(f'Image Label: {label}')
    fig.suptitle('FashionMNIST - First 16 Images')
    # Create the 'plots' folder if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    # Save the figure to the 'plots' folder
    plt.savefig('plots/first_16_images.png')
    plt.show()


def plot_train_test_metrics(epochs, train_losses, test_losses,
                            train_accuracies, test_accuracies,
                            type_plot='Original'):

    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.ylim(0.1, 0.8)
    plt.xticks(epochs_range)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy')
    plt.ylim(0.73, 1)
    plt.xticks(epochs_range)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/train_test_loss_accuracy_{type_plot}.png')
    plt.show()
