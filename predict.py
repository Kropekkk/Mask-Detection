import torch
from config import (NUM_CLASSES, TRAIN_DIR, ANN_DIR)
from model import get_model_instance_segmentation
import cv2
from utils import collate_fn, get_train_transform
#from dataset import CustomDataset

from PIL import Image
from torchvision import transforms

img = Image.open("test1.png")
convert_tensor = transforms.ToTensor()
img = convert_tensor(img)
simg = img.unsqueeze(0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model_instance_segmentation(NUM_CLASSES)
checkpoint = torch.load('models/last_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

data_transform = get_train_transform()
#dataset = CustomDataset(data_transform, TRAIN_DIR, ANN_DIR)
#data_loader = torch.utils.data.DataLoader(
#dataset, batch_size=4, collate_fn=collate_fn)

#for imgs, annotations in data_loader:
#        imgs = list(img.to(device) for img in imgs)
#        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#        break



#preds = model(imgs)
preds = model(simg)

print("=======================")
print(preds)

def plot_image(img_tensor, annotation):
    img = torch.permute(img_tensor, (1,2,0))
    img_np = img.to('cpu').numpy()

    img_np1 = img_np*255

    im_rgb = cv2.cvtColor(img_np1, cv2.COLOR_RGB2BGR)
    
    for box in annotation['boxes']:
        xmin, ymin, xmax, ymax = box
        
        cv2.rectangle(im_rgb, (int(xmin),int(ymin)),(int(xmax),int(ymax)), (0,255,0),2)
    
    cv2.imwrite('prediction.png', im_rgb)



plot_image(img,preds[0])

print("Save predicition.png")