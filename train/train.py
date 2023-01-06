import torch
from utils import collate_fn, get_train_transform, save_model, save_train_loss
from dataset import CustomDataset
from model import get_model_instance_segmentation
from config import (NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, TRAIN_DIR, ANN_DIR, RESIZE)

if __name__ == '__main__':
  data_transform = get_train_transform(RESIZE)
  dataset = CustomDataset(data_transform, TRAIN_DIR, ANN_DIR, RESIZE)
  train_size = int(len(dataset) * 0.8)
  test_size = len(dataset) - train_size
  train_data, test_data = torch.utils.data.random_split(dataset,[train_size,test_size])

  print(f"Train dataset lenght: {len(train_data)}   Test data: {len(test_data)} ")
  
  train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
  test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  model = get_model_instance_segmentation(NUM_CLASSES)
  model.to(device)

  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

  train_loss_results = []
  test_loss_results = []

  for epoch in range(NUM_EPOCHS):
    model.train() 
    train_loss = 0
    test_loss = 0

    for imgs, annotations in train_dataloader:
      imgs = list(img.to(device) for img in imgs)
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
      train_loss_dict = model(imgs, annotations)
      train_losses = sum(loss for loss in train_loss_dict.values())     

      optimizer.zero_grad()
      train_losses.backward()
      optimizer.step() 
      train_loss += train_losses
    
    for imgs, annotations in test_dataloader:
      imgs = list(img.to(device) for img in imgs)
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        
      with torch.no_grad():
        test_loss_dict = model(imgs, annotations)
        test_losses = sum(loss for loss in test_loss_dict.values())
        test_loss+= test_losses

    test_loss_results.append(test_loss)
    train_loss_results.append(train_loss)

    print(f"Train loss: {train_loss}    Test loss: {test_loss}")
    
  save_model(model, optimizer)
  save_train_loss(train_loss_results)