import os
import torch
from torchvision.transforms import v2

DATASET_DIR = 'data'
TRAINING_IMAGES_DIR = 'data/training_images'
VAL_IMAGES_DIR = 'data/validation_images'
TESTING_IMAGES_DIR = 'data/testing_images'

TRAINING_LABELS_DIR = 'data/training_labels'
VAL_LABELS_DIR = 'data/validation_labels'

CSV_FILEPATH = 'data/train_solution_bounding_boxes (1).csv'
ALL_FILENAMES = os.listdir(TRAINING_IMAGES_DIR)

TRANSFORM = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32),
    v2.Resize((448, 448)),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

NUM_GRIDS = 7
NUM_BOXES = 2
IMAGE_SIZE = 448

PIN_MEMORY = True
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'