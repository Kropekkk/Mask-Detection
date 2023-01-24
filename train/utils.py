import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

def save_model(model):
    """
    Saves models
    """

    torch.save(obj=model.state_dict(),
             f='last_model.pth')

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

def plot_image(img, annotation, scale_h, scale_w, save = False, confidence = 0.5):
    """
    Draw rectangles on original image
    """

    img_np = np.asarray(img)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for i, box in enumerate(annotation['boxes']):
        xmin, ymin, xmax, ymax = box

        if annotation['scores'][i] > confidence:
            if annotation['labels'][i] == 1:
                cv2.rectangle(img_rgb, (int(xmin*scale_w), int(ymin*scale_h)), (int(xmax*scale_w), int(ymax*scale_h)), (0, 255, 0), 2)
            elif annotation['labels'][i] == 2:
                cv2.rectangle(img_rgb, (int(xmin*scale_w), int(ymin*scale_h)), (int(xmax*scale_w), int(ymax*scale_h)), (0, 0 ,255), 2)
   
    if save:
        cv2.imwrite('prediction.png', img_rgb)

    return img_rgb

def plot_image_tensor(img_tensor, annotation, save = False):
    """
    Draws rectangles on detected boxes from given tensor of an image.

    Saves and returns plotted image.
    """
    img = torch.permute(img_tensor, (1,2,0))
    img_np = img.to('cpu').numpy()
    img_rgb = cv2.cvtColor(img_np*255, cv2.COLOR_RGB2BGR)
    
    for box in annotation['boxes']:
        xmin, ymin, xmax, ymax = box
        
        cv2.rectangle(img_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 2)
    
    if save:
        cv2.imwrite('prediction.png', img_rgb)

    return img_rgb