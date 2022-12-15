from torchvision import transforms

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_transform():
    return transforms.Compose([
        transforms.ToTensor(), 
    ])