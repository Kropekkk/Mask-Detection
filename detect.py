from PIL import Image
from io import BytesIO
from torchvision import transforms
from model import get_model_instance_segmentation
from config import (NUM_CLASSES)
import torch

def read(file):
    image = Image.open(BytesIO(file))
    return image

def predict(image: Image.Image):
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(image)
    simg = img.unsqueeze(0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(NUM_CLASSES)
    checkpoint = torch.load('models/last_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    preds = model(simg)

    return preds