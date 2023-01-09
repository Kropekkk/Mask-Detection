import torch
from train.config import (NUM_CLASSES, RESIZE)
from train.model import get_model_instance_segmentation
from train.utils import get_train_transform, plot_image
from PIL import Image
import io
import cv2

convert_tensor = get_train_transform(RESIZE)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(NUM_CLASSES)
checkpoint = torch.load('train/last_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()


def transform_to_tensor(image_bytes):
    """
    Transforms image bytes to tensor
    """
    image = Image.open(io.BytesIO(image_bytes))

    return convert_tensor(image).unsqueeze(0), image

def predict(image_bytes):
    """
    Predict label for given image
    """

    tensor, image = transform_to_tensor(image_bytes)
    preds = model(tensor)

    scale_h = image.size[1]/RESIZE[0]
    scale_w = image.size[0]/RESIZE[1]

    PATH = 'static/prediction.png'
    cv2.imwrite(PATH, plot_image(image, preds[0], scale_h, scale_w))

    return PATH