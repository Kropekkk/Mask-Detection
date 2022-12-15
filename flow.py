from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

def train(model: FastRCNNPredictor,
          dataloader: torch.utils.data.DataLoader,
          device: torch.device):
  EPOCHS = 10
  model.to(device)
      
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.01,
                                  momentum=0.9, weight_decay=0.0005)

  len_dataloader = len(dataloader)

  for epoch in range(EPOCHS):
      model.train()
      i = 0    
      epoch_loss = 0
      for imgs, annotations in dataloader:
          i += 1
          imgs = list(img.to(device) for img in imgs)
          annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
          loss_dict = model([imgs[0]], [annotations[0]])
          losses = sum(loss for loss in loss_dict.values())        

          optimizer.zero_grad()
          losses.backward()
          optimizer.step() 
          epoch_loss += losses
      print(epoch_loss)