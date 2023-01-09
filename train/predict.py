import torch
from config import (NUM_CLASSES, RESIZE)
from model import get_model_instance_segmentation
from utils import get_train_transform, plot_image
from PIL import Image
import sys

if len(sys.argv)>1:
    path = sys.argv[1]
    img = Image.open(path)
    convert_tensor = get_train_transform(RESIZE)
    tensor = convert_tensor(img).unsqueeze(0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(NUM_CLASSES)
    checkpoint = torch.load('last_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    preds = model(tensor)

    scale_h = img.size[1]/RESIZE[0]
    scale_w = img.size[0]/RESIZE[1]

    plot_image(img, preds[0], scale_h, scale_w, True)
else:
    print("Provide path to make predictions")