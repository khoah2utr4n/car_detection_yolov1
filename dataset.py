import os
import pandas as pd
import torch
from PIL import Image
import config
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_dataloader():
    train_dataset = CarDataset(
        csv_file=f'{config.DATASET_DIR}/train.csv',
        imgs_dir=config.TRAINING_IMAGES_DIR,
        labels_dir=config.TRAINING_LABELS_DIR,
        num_grids=config.NUM_GRIDS,
        num_boxes=config.NUM_BOXES,
        transform=config.TRANSFORM
    )
    val_dataset = CarDataset(
        csv_file=f'{config.DATASET_DIR}/val.csv',
        imgs_dir=config.VAL_IMAGES_DIR,
        labels_dir=config.VAL_LABELS_DIR,
        num_grids=config.NUM_GRIDS,
        num_boxes=config.NUM_BOXES,
        transform=config.TRANSFORM
    )
    test_dataset = CarDataset(
        csv_file=f'{config.DATASET_DIR}/test.csv',
        imgs_dir=config.TESTING_IMAGES_DIR,
        labels_dir='',
        num_grids=config.NUM_GRIDS,
        num_boxes=config.NUM_BOXES,
        transform=config.TRANSFORM
    )
    
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        pin_memory=config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        pin_memory=config.PIN_MEMORY
    )
    return train_loader, val_loader, test_loader


class CarDataset(Dataset):
    def __init__(self, csv_file, imgs_dir, labels_dir, num_grids, num_boxes, transform=None):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.S = num_grids
        self.B = num_boxes
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.imgs_dir, self.annotations.iloc[index]['img'])
        label_path = os.path.join(self.labels_dir, self.annotations.iloc[index]['label'])
        label_matrix = torch.zeros((self.S, self.S, 5*self.B))
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for label in f.readlines():
                    x, y, w, h = [
                        float(x)
                        for x in label.replace("\n", "").split()
                    ]
                    boxes.append([x, y, w, h])

        for box in boxes:
            x, y, w, h = box
            j, i = int(x * self.S), int(y * self.S)
            x, y = x * self.S - j, y * self.S - i
            if label_matrix[i, j, 4] == 0:
                label_matrix[i, j, 4] = 1 
                box_coord = torch.tensor([x, y, w, h])
                label_matrix[i, j, 0:4] = box_coord   
        
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img, boxes = self.transform(img, boxes)
            
        return img, label_matrix


if __name__ == '__main__':
    train_data = CarDataset(
        csv_file='data/train.csv',
        imgs_dir=config.TRAINING_IMAGES_DIR,
        labels_dir=config.TRAINING_LABELS_DIR,
        num_grids=7,
        num_boxes=2,
        transform=config.TRANSFORM
    )
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(train_features.shape, train_labels.shape)