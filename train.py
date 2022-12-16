import torch
from utils import collate_fn, get_train_transform, save_model, save_train_loss
from dataset import CustomDataset
from model import get_model_instance_segmentation
from config import (NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, TRAIN_DIR, ANN_DIR)

if __name__ == '__main__':
  data_transform = get_train_transform()
  dataset = CustomDataset(data_transform, TRAIN_DIR, ANN_DIR)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  model = get_model_instance_segmentation(NUM_CLASSES)
  model.to(device)

  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

  train_loss = []

  for epoch in range(NUM_EPOCHS):
    model.train() 
    epoch_loss = 0
    for imgs, annotations in dataloader:
      imgs = list(img.to(device) for img in imgs)
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
      loss_dict = model([imgs[0]], [annotations[0]])
      losses = sum(loss for loss in loss_dict.values())        

      optimizer.zero_grad()
      losses.backward()
      optimizer.step() 
      epoch_loss += losses
    
    train_loss.append(float(epoch_loss))

  save_model(model, optimizer)
  save_train_loss(train_loss)