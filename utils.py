import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_transform():
    return transforms.Compose([
        transforms.ToTensor(), 
    ])

def save_model(model, optimizer):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'outputs/last_model.pth')

def save_train_loss(train_loss):
    plt.figure(figsize=(16,8))
    plt.plot(train_loss, label="train_loss")
    plt.title("train loss")
    plt.savefig("train_loss.png")
    plt.close('all')