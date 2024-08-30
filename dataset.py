import os
import pandas as pd
import torch
from PIL import Image
import config
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2


def get_dataloader():
    train_dataset = CarDataset(
        csv_file=f'{config.DATASET_DIR}/train.csv',
        imgs_dir=config.TRAINING_IMAGES_DIR,
        labels_dir=config.TRAINING_LABELS_DIR,
        num_grids=config.NUM_GRIDS,
        num_boxes=config.NUM_BOXES,
        num_classes=config.NUM_CLASSES
    )
    val_dataset = CarDataset(
        csv_file=f'{config.DATASET_DIR}/val.csv',
        imgs_dir=config.VAL_IMAGES_DIR,
        labels_dir=config.VAL_LABELS_DIR,
        num_grids=config.NUM_GRIDS,
        num_boxes=config.NUM_BOXES,
        num_classes=config.NUM_CLASSES
    )
    test_dataset = CarDataset(
        csv_file=f'{config.DATASET_DIR}/test.csv',
        imgs_dir=config.TESTING_IMAGES_DIR,
        labels_dir='',
        num_grids=config.NUM_GRIDS,
        num_boxes=config.NUM_BOXES,
        num_classes=config.NUM_CLASSES
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
    def __init__(self, csv_file, imgs_dir, labels_dir, num_grids, num_boxes, num_classes):
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.S = num_grids
        self.B = num_boxes
        self.C = num_classes
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.imgs_dir, self.annotations.iloc[index]['img'])
        label_path = os.path.join(self.labels_dir, self.annotations.iloc[index]['label'])
        label_matrix = torch.zeros((self.S, self.S, 5*self.B + self.C))
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for label in f.readlines():
                    class_idx, x, y, w, h = [
                        float(x)
                        for x in label.replace("\n", "").split()
                    ]
                    boxes.append([int(class_idx), x, y, w, h])

        for box in boxes:
            class_idx, x, y, w, h = box
            j, i = int(x * self.S), int(y * self.S)
            x, y = x * self.S - j, y * self.S - i
            if label_matrix[i, j, 4] == 0:
                label_matrix[i, j, 4] = 1 
                label_matrix[i, j, 5+class_idx] = 1
                box_coord = torch.tensor([x, y, w, h])
                label_matrix[i, j, 0:4] = box_coord   
        
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        return img, label_matrix
    
    def transform(self, img):
        transform = v2.Compose([
            v2.ToImage(),  
            v2.ToDtype(torch.float32),  
            v2.Resize((448, 448)),  
        ])
        return transform(img)