import os
from PIL import Image
import torch
from bs4 import BeautifulSoup

class CustomDataset(object):
    def __init__(self, transforms, train_dir, ann_dir, height, width):
        self.transforms = transforms
        self.train_dir = train_dir
        self.ann_dir = ann_dir
        self.resize_height = height
        self.resize_width = width

    def resize(self, xmin, ymin, xmax, ymax, height, width):
        """
        Converts annotations to be compatible with resized image
        """
        xmin = int((self.resize_width/width) * xmin)
        ymin = int((self.resize_height/height) * ymin)
        xmax = int((self.resize_width/width) * xmax)
        ymax = int((self.resize_height/height) * ymax)

        return [xmin, ymin, xmax, ymax]

    def generate_box(self, obj, height, width):
        xmin = int(obj.find('xmin').text)
        ymin = int(obj.find('ymin').text)
        xmax = int(obj.find('xmax').text)
        ymax = int(obj.find('ymax').text)

        return self.resize(xmin, ymin, xmax, ymax, height, width)

    def generate_label(self, obj):
        if obj.find('name').text == "with_mask":
            return 1
        elif obj.find('name').text == "mask_weared_incorrect":
            return 2
        return 0

    def generate_target(self, image_id, file):
        with open(file) as f:
            data = f.read()
            soup = BeautifulSoup(data, 'xml')
            objects = soup.find_all('object')
            img_width = int(soup.find('width').text)
            img_height = int(soup.find('height').text)

            boxes = []
            labels = []

            for i in objects:
                boxes.append(self.generate_box(i, img_height, img_width))
                labels.append(self.generate_label(i))

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            img_id = torch.tensor([image_id])

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = img_id
            
            return target

    def __getitem__(self, idx):
        file_image = 'maksssksksss'+ str(idx) + '.png'
        file_label = 'maksssksksss'+ str(idx) + '.xml'
        
        img_path = os.path.join(self.train_dir, file_image)
        label_path = os.path.join(self.ann_dir, file_label)
        
        img = Image.open(img_path).convert("RGB")
        target = self.generate_target(idx, label_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(list(sorted(os.listdir(self.train_dir))))