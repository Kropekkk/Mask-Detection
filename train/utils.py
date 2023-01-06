import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

def save_model(model, optimizer):
    """
    Saves models
    """
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'last_model.pth')

def save_loss(train_loss, test_loss):
    """
    Plots train and test loss and saves it
    """
    plt.figure(figsize=(16,25))
    plt.plot(train_loss, label="Train loss")
    plt.plot(test_loss, label="Test loss")
    plt.legend()
    plt.savefig("train_loss.png")
    plt.close('all')